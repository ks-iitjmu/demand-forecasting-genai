"""Write plain-language executive summaries of the forecast with an LLM.

The LLM never calculates. Python computes every fact from SQL (totals, ranges, changes,
departments, events, accuracy) and formats each number; the LLM only turns the facts into
prose. Then every number in the summary is checked against the facts; if some are not
supported, the LLM gets one retry naming them, and a summary that still contains
unsupported numbers is stored with a warning instead of being presented as reliable.

Scopes: "all" (the four California stores together) and each store.
Output: SQLite table `summaries` (text, facts, model, verification result).
Usage:  python summaries/generate.py [--force]     (--force rewrites cached summaries)
"""
import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB, LLM_MODEL, LLM_PROVIDER, RESULTS  # noqa: E402
from core.llm import chat  # noqa: E402

DEPT_NAMES = {"FOODS_1": "Foods 1", "FOODS_2": "Foods 2", "FOODS_3": "Foods 3", "HOBBIES_1": "Hobbies 1",
              "HOBBIES_2": "Hobbies 2", "HOUSEHOLD_1": "Household 1", "HOUSEHOLD_2": "Household 2"}

SYSTEM = """You write short executive summaries of a retail sales forecast for store managers.
Use ONLY the facts provided. Copy every number exactly as it is written in the facts, with its
unit; never calculate, round differently, or introduce a number that is not in the facts.
Write natural, fluent English for a busy manager: do not repeat the wording of the fact labels.
Write 110 to 170 words of plain prose in three short paragraphs:
1. the headline: expected units and revenue for the period, with the likely range, and the change;
2. what drives it: the departments that change most, and events or SNAP days in the period;
3. how far to trust it: the forecast accuracy compared with the naive forecast, and the range,
   which is conservative (the sum of daily ranges), so describe it as a cautious outer bound.
No headings, no bullet points, no recommendations that the facts do not support."""


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def day(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{MONTHS[int(m) - 1]} {int(d)}"


def event_label(name: str) -> str:
    """M5 names events like 'MemorialDay' or 'NBAFinalsStart'; make them readable."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).replace("  ", " ")


def backtest_wape(store: str | None, model: str) -> float:
    """Mean WAPE over the backtest periods, for one store or all stores together."""
    bt = pd.read_parquet(RESULTS / "backtest_predictions.parquet")
    bt = bt[bt["model"] == model]
    if store is not None:
        bt = bt[bt["store_id"] == store]
    per_fold = bt.groupby("fold").apply(lambda g: (g["yhat"] - g["y"]).abs().sum() / g["y"].sum(),
                                        include_groups=False)
    return float(per_fold.mean())


def money(v: float) -> str:
    return f"${v / 1e6:.2f} million" if abs(v) >= 1e6 else f"${v / 1e3:.0f} thousand"


def pct(v: float) -> str:
    return f"{v * 100:+.1f}%"


def facts_for(con, store: str | None) -> dict:
    where, label = ("", "the four California stores") if store is None else (f"AND store_id = '{store}'", f"store {store}")
    f = pd.read_sql(f"SELECT * FROM forecasts WHERE 1=1 {where}", con)
    origin = f["origin"].iloc[0]
    start, end = f["date"].min(), f["date"].max()
    last28 = pd.read_sql(f"""SELECT SUM(units) AS units, SUM(revenue) AS revenue FROM daily_sales
                             WHERE date > date('{origin}', '-28 days') AND date <= '{origin}' {where}""", con).iloc[0]
    last_year = pd.read_sql(f"""SELECT SUM(units) AS units FROM daily_sales
                                WHERE date >= date('{start}', '-364 days') AND date <= date('{end}', '-364 days') {where}""",
                            con).iloc[0]
    dept_now = f.groupby("dept_id")["units"].sum()
    dept_before = pd.read_sql(f"""SELECT dept_id, SUM(units) AS units FROM daily_sales
                                  WHERE date > date('{origin}', '-28 days') AND date <= '{origin}' {where}
                                  GROUP BY dept_id""", con).set_index("dept_id")["units"]
    change = (dept_now / dept_before - 1).sort_values()
    events = f.loc[f["event_name"] != "", ["date", "event_name"]].drop_duplicates()
    snap_days = int(f.drop_duplicates("date")["snap"].sum())
    wape, base = backtest_wape(store, "ensemble"), backtest_wape(store, "seasonal_naive")
    cal = pd.read_sql("SELECT held_out_coverage FROM forecast_calibration", con).iloc[0, 0]

    units, lo, hi = f["units"].sum(), f["units_lo"].sum(), f["units_hi"].sum()
    return {
        "scope": label,
        "period": f"{day(start)} to {day(end)}, {end[:4]} (28 days)",
        "expected units": f"{units:,.0f}",
        "conservative range of units (sum of the daily 80% ranges; the true range of the total is narrower)": f"{lo:,.0f} to {hi:,.0f}",
        "expected revenue": money(f["revenue"].sum()),
        "conservative range of revenue (same basis)": f"{money(f['revenue_lo'].sum())} to {money(f['revenue_hi'].sum())}",
        "change in units vs the previous 28 days": pct(units / last28["units"] - 1),
        "change in units vs the same weeks last year": pct(units / last_year["units"] - 1),
        "fastest-growing departments": ", ".join(
            f"{DEPT_NAMES.get(d, d)} ({pct(v)})" for d, v in change[::-1].head(2).items() if v > 0) or "none",
        "fastest-declining departments": ", ".join(
            f"{DEPT_NAMES.get(d, d)} ({pct(v)})" for d, v in change.head(2).items() if v < 0) or "none",
        "events": "; ".join(f"{event_label(r.event_name)} ({day(r.date)})" for r in events.itertuples()) or "none",
        "SNAP (food-stamp) days": str(snap_days),
        f"forecast error in backtests for {label} (WAPE)": f"{wape * 100:.1f}%",
        f"error of a naive repeat-last-week forecast for {label}": f"{base * 100:.1f}%",
        "actual days inside the forecast range in held-out backtests": f"{cal * 100:.0f}%",
    }


# ------------------------------------------------------------------- number check
NUMBER = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*(?:million|thousand))?%?", re.I)


def values_in(text: str) -> list[tuple[str, float]]:
    out = []
    for m in NUMBER.findall(text):
        s = m.strip()
        num = float(re.sub(r"[^\d.\-]", "", s.replace(",", "")) or "nan")
        if "million" in s.lower():
            num *= 1e6
        elif "thousand" in s.lower():
            num *= 1e3
        out.append((s, num))
    return out


def unsupported(summary: str, facts: dict) -> list[str]:
    """Numbers in the summary that match no fact. Dates, years and small counts are allowed."""
    allowed = [v for fact in facts.values() for _, v in values_in(str(fact))]
    allowed += [v for k in facts for _, v in values_in(k)]
    bad = []
    for text, v in values_in(summary):
        if v != v:                                   # NaN: not a number after all
            continue
        if (abs(v) <= 31 and float(v).is_integer()) or 2011 <= abs(v) <= 2016:
            continue                                 # day numbers, small counts, years
        if not any(abs(abs(v) - abs(a)) <= max(0.051, 0.005 * abs(a)) for a in allowed):
            bad.append(text)
    return bad


def complete(text: str) -> bool:
    """A finished summary ends with a full sentence and has a reasonable length."""
    return text.rstrip().endswith((".", "!", "?")) and len(text.split()) >= 70


def problems(text: str, facts: dict) -> list[str]:
    issues = [f"numbers not in the facts: {', '.join(b)}" for b in [unsupported(text, facts)] if b]
    if not complete(text):
        issues.append("the summary is incomplete or too short")
    return issues


def write_summary(facts: dict) -> dict:
    fact_text = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    user = f"Facts:\n{fact_text}\n\nWrite the summary."
    text = chat(SYSTEM, user, max_tokens=4096)
    issues, retried = problems(text, facts), False
    if issues:
        retried = True
        retry = (f"{user}\n\nYour previous summary had problems: {'; '.join(issues)}. Write it again: "
                 "complete, 110 to 170 words, using only numbers exactly as written in the facts."
                 f"\n\nPrevious summary:\n{text}")
        text = chat(SYSTEM, retry, max_tokens=4096)
    bad = unsupported(text, facts)
    return {"text": text, "verified": int(not bad and complete(text)),
            "unsupported_numbers": json.dumps(bad + ([] if complete(text) else ["(incomplete)"])),
            "retried": int(retried)}


def main(force: bool):
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS summaries (
        scope TEXT PRIMARY KEY, text TEXT, verified INTEGER, unsupported_numbers TEXT, retried INTEGER,
        facts TEXT, facts_hash TEXT, provider TEXT, model TEXT, created TEXT)""")
    stores = [r[0] for r in con.execute("SELECT DISTINCT store_id FROM forecasts ORDER BY store_id")]
    for scope in ["all"] + stores:
        facts = facts_for(con, None if scope == "all" else scope)
        h = hashlib.sha1(json.dumps(facts, sort_keys=True).encode()).hexdigest()[:16]
        row = con.execute("SELECT facts_hash FROM summaries WHERE scope = ?", (scope,)).fetchone()
        if row and row[0] == h and not force:
            print(f"{scope:5s} unchanged facts, keeping the existing summary")
            continue
        print(f"{scope:5s} writing with {LLM_PROVIDER}/{LLM_MODEL} ...", flush=True)
        s = write_summary(facts)
        con.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (scope, s["text"], s["verified"], s["unsupported_numbers"], s["retried"], json.dumps(facts),
                     h, LLM_PROVIDER, LLM_MODEL, datetime.now(timezone.utc).isoformat(timespec="seconds")))
        con.commit()
        status = "verified" if s["verified"] else f"UNVERIFIED numbers: {s['unsupported_numbers']}"
        print(f"      {status}{' (after one retry)' if s['retried'] else ''}")
    print()
    for scope, text, verified in con.execute("SELECT scope, text, verified FROM summaries ORDER BY scope = 'all' DESC, scope"):
        print(f"=== {scope} {'' if verified else '(contains unverified numbers)'}\n{text}\n")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rewrite summaries even if their facts are unchanged")
    main(ap.parse_args().force)
