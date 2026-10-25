"""Rolling-origin backtest: forecast three separate 28-day periods as if each were the future.

For fold k (k = 3, 2, 1), the forecast origin is the last known day minus 28 x k days; each
model trains only on data up to its origin and forecasts the next 28 days for all 28 series.
The ensemble is the mean of Prophet and XGBoost.

Metrics (pooled over all series and days of a fold, then averaged over folds):
  WAPE      sum |actual - forecast| / sum actual        (lower is better)
  bias      sum (forecast - actual) / sum actual        (+ over-forecast, - under-forecast)
  RMSE      root mean squared error in units
  coverage  share of actual values inside the 80% interval (ideal: about 0.80)

Output: results/backtest_predictions.parquet, backtest_metrics.csv, backtest_summary.md
Usage:  python forecast/backtest.py [--models seasonal_naive prophet xgboost]
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_FOLDS, DB, HORIZON, RESULTS  # noqa: E402
from forecast.models import KEYS, MODELS  # noqa: E402


def load_features() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql("SELECT * FROM features ORDER BY store_id, dept_id, date", con, parse_dates=["date"])
    con.close()
    return df


def metrics(p: pd.DataFrame) -> pd.Series:
    err = p["yhat"] - p["y"]
    return pd.Series({"wape": err.abs().sum() / p["y"].sum(), "bias": err.sum() / p["y"].sum(),
                      "rmse": float(np.sqrt((err ** 2).mean())),
                      "coverage": ((p["y"] >= p["lo"]) & (p["y"] <= p["hi"])).mean()})


def main(models: list[str], folds: int):
    df = load_features()
    last_known = df.loc[df["is_future"] == 0, "date"].max()
    actual = df[df["is_future"] == 0][KEYS + ["date", "units"]].rename(columns={"units": "y"})
    preds = []
    for k in range(folds, 0, -1):
        origin = last_known - pd.Timedelta(days=HORIZON * k)
        print(f"\nFold {folds - k + 1}: origin {origin.date()}, forecasting "
              f"{(origin + pd.Timedelta(days=1)).date()} to {(origin + pd.Timedelta(days=HORIZON)).date()}")
        fold_preds = {}
        for name in models:
            t0 = time.time()
            fold_preds[name] = MODELS[name](df, origin, HORIZON)
            print(f"  {name:15s} {time.time() - t0:6.1f} s")
        if {"prophet", "xgboost"} <= set(fold_preds):
            a, b = fold_preds["prophet"], fold_preds["xgboost"].set_index(KEYS + ["date"]).reindex(
                pd.MultiIndex.from_frame(fold_preds["prophet"][KEYS + ["date"]])).reset_index()
            fold_preds["ensemble"] = a[KEYS + ["date"]].assign(
                yhat=(a["yhat"].values + b["yhat"].values) / 2, lo=(a["lo"].values + b["lo"].values) / 2,
                hi=(a["hi"].values + b["hi"].values) / 2)
        for name, p in fold_preds.items():
            preds.append(p.assign(model=name, fold=folds - k + 1, origin=origin))
    pred = pd.concat(preds, ignore_index=True).merge(actual, on=KEYS + ["date"])

    by_fold = pred.groupby(["model", "fold"]).apply(metrics, include_groups=False).reset_index()
    overall = by_fold.groupby("model")[["wape", "bias", "rmse", "coverage"]].mean()
    overall["wape_std"] = by_fold.groupby("model")["wape"].std()
    if "seasonal_naive" in overall.index:
        base = overall.loc["seasonal_naive", "wape"]
        overall["wape_vs_baseline"] = overall["wape"] / base - 1
    order = [m for m in ["seasonal_naive", "prophet", "xgboost", "ensemble"] if m in overall.index]
    overall = overall.loc[order]
    by_dept = (pred.groupby(["model", "dept_id"]).apply(metrics, include_groups=False)["wape"]
               .unstack("model")[order])

    RESULTS.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(RESULTS / "backtest_predictions.parquet", index=False)
    by_fold.to_csv(RESULTS / "backtest_metrics.csv", index=False)
    text = (f"Rolling-origin backtest: {folds} folds of {HORIZON} days, {pred[KEYS].drop_duplicates().shape[0]} "
            f"series, last known day {last_known.date()}.\n\n**Overall (mean over folds)**\n\n"
            + overall.to_markdown(floatfmt=".3f") + "\n\n**WAPE by department (all folds)**\n\n"
            + by_dept.to_markdown(floatfmt=".3f") + "\n\n**WAPE by fold**\n\n"
            + by_fold.pivot(index="fold", columns="model", values="wape")[order].to_markdown(floatfmt=".3f"))
    (RESULTS / "backtest_summary.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--folds", type=int, default=BACKTEST_FOLDS)
    a = ap.parse_args()
    main(a.models, a.folds)
