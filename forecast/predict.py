"""Forecast the 28 days after the last day with sales, for every store and department.

1. Prophet and XGBoost are trained on all known days; the ensemble is their mean (the best
   model in the backtest).
2. The 80% intervals are calibrated: the backtest showed the ensemble's raw intervals cover
   about 77%, so each side is widened by the smallest factor that reaches 80% on the backtest
   forecasts. Calibrating and checking on the same data would flatter the result, so the
   honest coverage is estimated by leave-one-fold-out: the factor is fitted on two backtest
   periods and checked on the third, for each period in turn.
3. Revenue = forecast units x each series' price per unit over its last 28 days, adjusted by
   the planned price change (the price index is known in advance).

Output: SQLite tables `forecasts` and `forecast_calibration`; results/forecast.csv
Usage:  python forecast/predict.py        (run forecast/backtest.py first)
"""
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB, HORIZON, INTERVAL_TARGET, RESULTS  # noqa: E402
from forecast.backtest import load_features  # noqa: E402
from forecast.models import KEYS, prophet_model, xgboost_model  # noqa: E402


def widen(p: pd.DataFrame, s: float) -> tuple[np.ndarray, np.ndarray]:
    lo = p["yhat"] - s * (p["yhat"] - p["lo"]).clip(lower=0)
    hi = p["yhat"] + s * (p["hi"] - p["yhat"]).clip(lower=0)
    return lo.clip(lower=0).to_numpy(), hi.to_numpy()


def coverage(p: pd.DataFrame, s: float) -> float:
    lo, hi = widen(p, s)
    return float(((p["y"] >= lo) & (p["y"] <= hi)).mean())


def fit_scale(p: pd.DataFrame, target: float) -> float:
    """Smallest widening factor whose coverage reaches the target."""
    for s in np.arange(0.5, 5.0, 0.01):
        if coverage(p, s) >= target:
            return round(float(s), 2)
    return 5.0


def calibrate(target: float) -> dict:
    bt = pd.read_parquet(RESULTS / "backtest_predictions.parquet")
    ens = bt[bt["model"] == "ensemble"]
    if ens.empty:
        sys.exit("No ensemble forecasts in the backtest: run  python forecast/backtest.py  first")
    folds = sorted(ens["fold"].unique())
    held_out = {}
    for f in folds:                                     # fit on the other folds, check on this one
        s = fit_scale(ens[ens["fold"] != f], target)
        held_out[int(f)] = coverage(ens[ens["fold"] == f], s)
    return {"target": target, "scale": fit_scale(ens, target),
            "raw_coverage": round(coverage(ens, 1.0), 3),
            "held_out_coverage": round(float(np.mean(list(held_out.values()))), 3),
            "held_out_by_fold": {k: round(v, 3) for k, v in held_out.items()}}


def unit_prices(df: pd.DataFrame, last_known: pd.Timestamp) -> pd.DataFrame:
    """Recent price per unit per series and the price index it was earned at."""
    recent = df[(df["date"] > last_known - pd.Timedelta(days=HORIZON)) & (df["date"] <= last_known)]
    g = recent.groupby(KEYS).agg(revenue=("revenue", "sum"), units=("units", "sum"),
                                 base_index=("price_index", "mean")).reset_index()
    g["unit_price"] = g["revenue"] / g["units"]
    return g[KEYS + ["unit_price", "base_index"]]


def main():
    df = load_features()
    last_known = df.loc[df["is_future"] == 0, "date"].max()
    print(f"Training on all data up to {last_known.date()}; forecasting "
          f"{(last_known + pd.Timedelta(days=1)).date()} to {(last_known + pd.Timedelta(days=HORIZON)).date()}")
    p = prophet_model(df, last_known, HORIZON)
    x = xgboost_model(df, last_known, HORIZON)
    f = p.merge(x, on=KEYS + ["date"], suffixes=("_prophet", "_xgboost"))
    f["yhat"] = (f["yhat_prophet"] + f["yhat_xgboost"]) / 2
    f["lo"] = (f["lo_prophet"] + f["lo_xgboost"]) / 2
    f["hi"] = (f["hi_prophet"] + f["hi_xgboost"]) / 2

    cal = calibrate(INTERVAL_TARGET)
    f["units_lo"], f["units_hi"] = widen(f, cal["scale"])
    print(f"Interval calibration: raw coverage {cal['raw_coverage']:.0%} -> widened by x{cal['scale']}; "
          f"coverage on held-out backtest periods {cal['held_out_coverage']:.0%} (target {INTERVAL_TARGET:.0%})")

    fut = df[df["is_future"] == 1][KEYS + ["date", "cat_id", "price_index", "is_event", "event_name", "snap"]]
    f = f.merge(fut, on=KEYS + ["date"]).merge(unit_prices(df, last_known), on=KEYS)
    price = f["unit_price"] * (f["price_index"].fillna(f["base_index"]) / f["base_index"])
    out = pd.DataFrame({
        "store_id": f["store_id"], "dept_id": f["dept_id"], "cat_id": f["cat_id"],
        "date": f["date"].dt.strftime("%Y-%m-%d"),
        "units": f["yhat"].round(1), "units_lo": f["units_lo"].round(1), "units_hi": f["units_hi"].round(1),
        "revenue": (f["yhat"] * price).round(2), "revenue_lo": (f["units_lo"] * price).round(2),
        "revenue_hi": (f["units_hi"] * price).round(2),
        "units_prophet": f["yhat_prophet"].round(1), "units_xgboost": f["yhat_xgboost"].round(1),
        "is_event": f["is_event"], "event_name": f["event_name"], "snap": f["snap"],
        "origin": last_known.strftime("%Y-%m-%d"), "model": "ensemble (Prophet + XGBoost)",
    }).sort_values(KEYS + ["date"])

    con = sqlite3.connect(DB)
    out.to_sql("forecasts", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_forecasts ON forecasts (store_id, dept_id, date)")
    pd.DataFrame([{"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in cal.items()}}]
                 ).to_sql("forecast_calibration", con, if_exists="replace", index=False)
    con.commit()
    totals = pd.read_sql("""
        SELECT f.store_id, ROUND(SUM(f.units)) AS units_forecast,
               (SELECT SUM(units) FROM daily_sales d WHERE d.store_id = f.store_id
                  AND d.date > date(f.origin, '-28 days') AND d.date <= f.origin) AS units_last_28d,
               ROUND(SUM(f.revenue) / 1e3) AS revenue_forecast_k
        FROM forecasts f GROUP BY f.store_id ORDER BY f.store_id""", con)
    con.close()
    totals["change_vs_last_28d"] = (totals["units_forecast"] / totals["units_last_28d"] - 1).map("{:+.1%}".format)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "forecast.csv", index=False)
    (RESULTS / "interval_calibration.json").write_text(json.dumps(cal, indent=2))
    print(f"\nWrote {len(out)} forecasts to the SQL table `forecasts` and results/forecast.csv\n")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
