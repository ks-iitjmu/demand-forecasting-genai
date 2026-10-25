"""Demand forecasting dashboard.   streamlit run app/dashboard.py

Reads only from the SQLite database and results/ (read-only); run the pipeline first:
load_m5 -> build_features -> backtest -> predict -> summaries.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB, RESULTS  # noqa: E402

DEPT_NAMES = {"FOODS_1": "Foods 1", "FOODS_2": "Foods 2", "FOODS_3": "Foods 3", "HOBBIES_1": "Hobbies 1",
              "HOBBIES_2": "Hobbies 2", "HOUSEHOLD_1": "Household 1", "HOUSEHOLD_2": "Household 2"}
MODEL_NAMES = {"seasonal_naive": "Seasonal naive (baseline)", "prophet": "Prophet", "xgboost": "XGBoost",
               "ensemble": "Ensemble (Prophet + XGBoost)"}
BLUE, CYAN, GREY, LINE = "#3B82F6", "#22D3EE", "#8B95AD", "#1C2740"

st.set_page_config(page_title="Demand Forecast", page_icon=":material/insights:", layout="wide")


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()


@st.cache_data
def backtest() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "backtest_predictions.parquet")


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13))
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE, rangemode="tozero")
    return fig


def wape(df: pd.DataFrame) -> float:
    return float((df["yhat"] - df["y"]).abs().sum() / df["y"].sum())


# --------------------------------------------------------------------------- data
if not DB.exists() or not (RESULTS / "backtest_predictions.parquet").exists():
    st.error("No data yet. Run the pipeline first: load_m5, build_features, backtest, predict, summaries.")
    st.stop()

fc_all = query("SELECT * FROM forecasts")
stores = sorted(fc_all["store_id"].unique())
depts = sorted(fc_all["dept_id"].unique())
origin = fc_all["origin"].iloc[0]

# ------------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
store = st.sidebar.selectbox("Store", ["All stores"] + stores)
dept_label = {DEPT_NAMES.get(d, d): d for d in depts}
chosen = [dept_label[x] for x in st.sidebar.multiselect("Departments", list(dept_label), default=list(dept_label))]
measure = st.sidebar.radio("Measure", ["Units", "Revenue"], horizontal=True)
history_days = int(st.sidebar.select_slider("History shown", ["56 days", "91 days", "182 days", "365 days"],
                                            value="91 days").split()[0])
if not chosen:
    st.warning("Choose at least one department.")
    st.stop()

sel_stores = stores if store == "All stores" else [store]
value, lo_col, hi_col = ("units", "units_lo", "units_hi") if measure == "Units" else ("revenue", "revenue_lo", "revenue_hi")
fc = fc_all[fc_all["store_id"].isin(sel_stores) & fc_all["dept_id"].isin(chosen)]
marks = ",".join("?" * len(sel_stores)), ",".join("?" * len(chosen))
hist = query(f"""SELECT date, store_id, dept_id, units, revenue FROM daily_sales
                 WHERE store_id IN ({marks[0]}) AND dept_id IN ({marks[1]})
                   AND date > date(?, '-{int(max(history_days, 392))} days') AND date <= ?""",
             tuple(sel_stores) + tuple(chosen) + (origin, origin))
fmt = (lambda v: f"{v:,.0f}") if measure == "Units" else (lambda v: f"${v / 1e3:,.0f}k")

# ------------------------------------------------------------------------- header
start, end = fc["date"].min(), fc["date"].max()
st.title("Demand forecast")
scope = "four California Walmart stores" if store == "All stores" else f"store {store}"
dept_note = "all departments" if len(chosen) == len(depts) else f"{len(chosen)} of {len(depts)} departments"
st.caption(f"Next 28 days ({start} to {end}) for {scope}, {dept_note}. Data from the M5 dataset; "
           f"the last day with sales is {origin}.")

h = hist.copy()
h["date"] = pd.to_datetime(h["date"])
last28 = h[h["date"] > pd.Timestamp(origin) - pd.Timedelta(days=28)][value].sum()
f_start = pd.Timestamp(start)
last_year = h[(h["date"] >= f_start - pd.Timedelta(days=364)) &
              (h["date"] <= f_start - pd.Timedelta(days=364 - 27))][value].sum()
total = fc[value].sum()
bt = backtest()
bt_sel = bt[bt["store_id"].isin(sel_stores) & bt["dept_id"].isin(chosen)]

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Expected {measure.lower()}, next 28 days", fmt(total),
          f"{total / last28 - 1:+.1%} vs previous 28 days" if last28 else None)
k2.metric("Change vs same weeks last year", f"{total / last_year - 1:+.1%}" if last_year else "n/a")
k3.metric("Conservative range", f"{fmt(fc[lo_col].sum())} – {fmt(fc[hi_col].sum())}",
          help="Sum of the daily 80% ranges; the true range of the 28-day total is narrower.")
ens = wape(bt_sel[bt_sel["model"] == "ensemble"])
base = wape(bt_sel[bt_sel["model"] == "seasonal_naive"])
k4.metric("Forecast error in backtests (WAPE)", f"{ens:.1%}", f"{ens / base - 1:+.0%} vs naive baseline",
          delta_color="inverse", help="Weighted absolute percentage error over three 28-day backtest periods.")

tab_fc, tab_sum, tab_acc, tab_data = st.tabs(["Forecast", "Executive summary", "Model accuracy", "Data and method"])

# ----------------------------------------------------------------------- forecast
with tab_fc:
    shown = h[h["date"] > pd.Timestamp(origin) - pd.Timedelta(days=history_days)].groupby("date")[value].sum()
    f = fc.groupby("date")[[value, lo_col, hi_col]].sum()
    f.index = pd.to_datetime(f.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f.index.tolist() + f.index[::-1].tolist(),
                             y=f[hi_col].tolist() + f[lo_col][::-1].tolist(), fill="toself",
                             fillcolor="rgba(59,130,246,0.18)", line=dict(width=0), hoverinfo="skip",
                             name="80% range (daily)"))
    fig.add_trace(go.Scatter(x=shown.index, y=shown.values, name="Actual", line=dict(color=GREY, width=1.6)))
    fig.add_trace(go.Scatter(x=f.index, y=f[value], name="Forecast", line=dict(color=BLUE, width=2.4)))
    fig.add_vline(x=pd.Timestamp(origin), line=dict(color=LINE, width=1, dash="dot"))
    events = fc.loc[fc["event_name"] != "", ["date", "event_name"]].drop_duplicates()
    for r in events.itertuples():
        fig.add_vline(x=pd.Timestamp(r.date), line=dict(color=CYAN, width=1, dash="dot"))
        fig.add_annotation(x=pd.Timestamp(r.date), y=1, yref="paper", text=r.event_name, showarrow=False,
                           textangle=-90, xanchor="left", yanchor="top", font=dict(size=11, color=CYAN))
    st.plotly_chart(style(fig), width="stretch")
    st.caption("Grey: actual daily sales. Blue: ensemble forecast with its calibrated 80% daily range. "
               "Dotted cyan lines: events in the forecast period.")

    st.subheader("By department")
    d_now = fc.groupby("dept_id")[[value]].sum()
    d_before = h[h["date"] > pd.Timestamp(origin) - pd.Timedelta(days=28)].groupby("dept_id")[value].sum()
    table = pd.DataFrame({"Department": [DEPT_NAMES.get(d, d) for d in d_now.index],
                          f"Forecast {measure.lower()}": d_now[value].values,
                          "Previous 28 days": d_before.reindex(d_now.index).values})
    table["Change"] = table[f"Forecast {measure.lower()}"] / table["Previous 28 days"] - 1
    table = table.sort_values("Change", ascending=False)
    money_fmt = "%,.0f" if measure == "Units" else "$%,.0f"
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        f"Forecast {measure.lower()}": st.column_config.NumberColumn(format=money_fmt),
        "Previous 28 days": st.column_config.NumberColumn(format=money_fmt),
        "Change": st.column_config.NumberColumn(format="percent")})

# ------------------------------------------------------------------------ summary
with tab_sum:
    scope_key = "all" if store == "All stores" else store
    s = query("SELECT * FROM summaries WHERE scope = ?", (scope_key,))
    if s.empty:
        st.info("No summary yet. Run  python summaries/generate.py")
    else:
        row = s.iloc[0]
        if len(chosen) != len(depts):
            st.caption("The summary covers all departments of the selected store, not only the filtered ones.")
        if row["verified"]:
            st.success("Every number in this summary was checked against the facts below.", icon=":material/verified:")
        else:
            st.warning(f"This summary contains numbers that could not be verified: {row['unsupported_numbers']}", icon=":material/warning:")
        text_col, _ = st.columns([5, 2])                  # a comfortable reading width
        text_col.markdown(row["text"].replace("$", "\\$"))
        st.caption(f"Written by {row['provider']}/{row['model']} on {row['created'][:10]} from the facts below; "
                   "the model never calculates, it only writes.")
        with st.expander("Facts the summary was written from"):
            facts = json.loads(row["facts"])
            st.dataframe(pd.DataFrame(facts.items(), columns=["Fact", "Value"]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------- accuracy
with tab_acc:
    st.markdown("Each model forecast three separate 28-day periods using only earlier data, "
                "as if each period were the future (a rolling-origin backtest).")
    rows = []
    for m, g in bt_sel.groupby("model"):
        per_fold = g.groupby("fold").apply(wape, include_groups=False)
        err = g["yhat"] - g["y"]
        rows.append({"Model": MODEL_NAMES.get(m, m), "WAPE": per_fold.mean(), "WAPE spread across periods": per_fold.std(),
                     "Bias": err.sum() / g["y"].sum(),
                     "80% range coverage": ((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean(), "_order": list(MODEL_NAMES).index(m)})
    acc = pd.DataFrame(rows).sort_values("_order").drop(columns="_order")
    if True:
        fig = go.Figure(go.Bar(x=acc["WAPE"], y=acc["Model"], orientation="h", text=[f"{v:.1%}" for v in acc["WAPE"]],
                               textposition="outside",
                               marker_color=[GREY if "baseline" in m else (BLUE if "Ensemble" in m else "#2B4C8C") for m in acc["Model"]]))
        fig.update_layout(xaxis_title="WAPE (lower is better)", yaxis=dict(autorange="reversed"))
        fig.update_xaxes(tickformat=".0%", range=[0, acc["WAPE"].max() * 1.18])
        st.plotly_chart(style(fig, 280), width="stretch")
        st.dataframe(acc, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in acc.columns if c != "Model"})
    st.caption("Raw ranges; the final forecast's ranges are widened to cover 80% (checked on held-out periods).")

    st.subheader("Backtest forecasts against actual sales")
    label_to_model = {v: k for k, v in MODEL_NAMES.items()}
    model = label_to_model[st.selectbox("Model", list(label_to_model), index=3)]
    g = bt_sel[bt_sel["model"] == model].groupby("date")[["y", "yhat"]].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["y"], name="Actual", mode="lines", line=dict(color=GREY, width=1.6)))
    fig.add_trace(go.Scatter(x=g.index, y=g["yhat"], name="Forecast", mode="lines", line=dict(color=BLUE, width=2.2)))
    st.plotly_chart(style(fig, 340), width="stretch")

# --------------------------------------------------------------------------- data
with tab_data:
    st.download_button("Download the forecast (CSV)", fc.to_csv(index=False), file_name=f"forecast_{scope_key}.csv",
                       mime="text/csv")
    st.markdown(f"""
**Data.** M5 Forecasting (Kaggle): daily Walmart sales, January 2011 to {origin}, for the four California
stores. Forecasts are made per store and department (28 series); revenue is units x the planned shelf price.

**Features.** Built in SQLite with window functions: sales 4, 5 and 52 weeks earlier, 7- and 28-day averages
ending 28 days earlier, calendar, events, SNAP days and a price index.

**Models.** Prophet (one per series) and XGBoost (one global model) are averaged; both beat a naive
repeat-last-week baseline in the backtest. The 80% ranges are calibrated on the backtest.

**Summaries.** Python computes every fact; the LLM only writes the text, and every number in it is checked
against the facts.
""")
