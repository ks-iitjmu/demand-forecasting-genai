"""Load the M5 Walmart data into SQLite (data/demand.db).

Input  data/raw/: the M5 zip from Kaggle (m5-forecasting-accuracy.zip), or its CSV files:
       calendar.csv, sell_prices.csv, sales_train_evaluation.csv (or _validation.csv)
Output tables
  calendar      one row per day: weekday, month, year, Walmart week, events, SNAP days
  daily_sales   one row per store, department and day: units sold and revenue
  products      number of products per store and department (for context)

We forecast at store x department level: 4 California stores x 7 departments = 28 series.
Single products sell nothing on most days, which suits neither Prophet nor a business
planning view; department totals are smooth and are what a planner would use.
Revenue is computed per product as daily units x that week's shelf price, then summed.

Usage:  python ingest/load_m5.py
"""
import sqlite3
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB, RAW, STATE  # noqa: E402

NEEDED = ["calendar.csv", "sell_prices.csv"]
SALES = ["sales_train_evaluation.csv", "sales_train_validation.csv"]   # evaluation has 28 more days


def extract_if_zipped():
    if all((RAW / f).exists() for f in NEEDED) and any((RAW / f).exists() for f in SALES):
        return
    zips = sorted(RAW.glob("*.zip"))
    if not zips:
        sys.exit(f"Put the M5 zip (or its CSV files) in {RAW}")
    with zipfile.ZipFile(zips[0]) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for f in NEEDED + SALES:
            if f in names and not (RAW / f).exists():
                print(f"extracting {f}")
                with z.open(names[f]) as src, open(RAW / f, "wb") as dst:
                    dst.write(src.read())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    extract_if_zipped()
    sales_file = next(RAW / f for f in SALES if (RAW / f).exists())

    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    header = pd.read_csv(sales_file, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    sales = pd.read_csv(sales_file, dtype={c: "int16" for c in day_cols})
    sales = sales[sales["state_id"] == STATE].reset_index(drop=True)
    print(f"{sales_file.name}: {len(sales):,} product-store rows in {STATE}, {len(day_cols):,} days")

    # Weekly shelf prices -> one price per product-store-day, aligned with the sales columns
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"].str.startswith(STATE + "_")]
    week_of_day = cal.set_index("d").loc[day_cols, "wm_yr_wk"].to_numpy()
    price_wide = prices.pivot_table(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
    price_wide = price_wide.reindex(index=pd.MultiIndex.from_frame(sales[["store_id", "item_id"]]))
    price_daily = price_wide.reindex(columns=week_of_day).to_numpy(dtype="float32")

    units = sales[day_cols].to_numpy(dtype="float32")
    revenue = units * np.nan_to_num(price_daily)
    missing_price = int(((units > 0) & np.isnan(price_daily)).sum())
    print(f"revenue computed; product-days with sales but no listed price: {missing_price}")

    keys = sales[["store_id", "dept_id", "cat_id"]]
    u = pd.concat([keys, pd.DataFrame(units, columns=day_cols)], axis=1).groupby(["store_id", "dept_id", "cat_id"]).sum()
    r = pd.concat([keys, pd.DataFrame(revenue, columns=day_cols)], axis=1).groupby(["store_id", "dept_id", "cat_id"]).sum()
    long_u = u.stack().rename("units")
    long_r = r.stack().rename("revenue")
    daily = pd.concat([long_u, long_r], axis=1).reset_index().rename(columns={"level_3": "d"})
    daily = daily.merge(cal[["d", "date"]], on="d").drop(columns="d")
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily["units"] = daily["units"].round().astype(int)
    daily["revenue"] = daily["revenue"].round(2)
    daily = daily[["date", "store_id", "dept_id", "cat_id", "units", "revenue"]].sort_values(["store_id", "dept_id", "date"])

    calendar = cal.rename(columns={f"snap_{STATE}": "snap"})[
        ["date", "d", "wm_yr_wk", "weekday", "wday", "month", "year",
         "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap"]].copy()
    calendar["date"] = calendar["date"].dt.strftime("%Y-%m-%d")
    products = sales.groupby(["store_id", "dept_id"]).size().rename("products").reset_index()

    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    calendar.to_sql("calendar", con, index=False)
    daily.to_sql("daily_sales", con, index=False)
    products.to_sql("products", con, index=False)
    con.execute("CREATE UNIQUE INDEX ix_daily ON daily_sales (store_id, dept_id, date)")
    con.execute("CREATE UNIQUE INDEX ix_calendar ON calendar (date)")
    con.commit()

    summary = pd.read_sql("""
        SELECT store_id, COUNT(DISTINCT dept_id) AS departments, MIN(date) AS first_day, MAX(date) AS last_day,
               SUM(units) AS units, ROUND(SUM(revenue) / 1e6, 1) AS revenue_millions
        FROM daily_sales GROUP BY store_id ORDER BY store_id""", con)
    series = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT store_id, dept_id FROM daily_sales)").fetchone()[0]
    con.close()
    print(f"\nWrote {DB}: {len(daily):,} rows, {series} store-department series\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
