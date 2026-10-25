"""Build model features in SQL (data/demand.db, table `features`).

One row per store, department and date, from the first day of data to the end of the
calendar, which runs 28 days past the last day with sales: those future rows are what the
models forecast. Sales-based features use only information at least HORIZON (28) days old,
so the same features are available when forecasting 28 days ahead:

  lag_28, lag_35        units on the same weekday 4 and 5 weeks earlier
  mean_7_lag28          mean units over the 7 days ending 28 days earlier
  mean_28_lag28         mean units over the 28 days ending 28 days earlier
  lag_364               units on the same weekday one year earlier (52 weeks)
  calendar              weekday, month, year, day of month, weekend, event type, SNAP day
  price_index           mean of each product's shelf price relative to its own average
                        price (below 1.0 = discounts); Walmart prices are set in advance

Also written: table `dept_prices` (store, department, Walmart week, price index).

Usage:  python ingest/build_features.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DB, HORIZON, RAW, STATE  # noqa: E402

FEATURES_SQL = f"""
CREATE TABLE features AS
WITH series AS (SELECT DISTINCT store_id, dept_id, cat_id FROM daily_sales),
spine AS (                      -- every series x every calendar day, future days included
    SELECT s.store_id, s.dept_id, s.cat_id, c.date, c.wm_yr_wk, c.wday, c.month, c.year,
           c.event_name_1, c.event_type_1, c.event_name_2, c.snap
    FROM series s CROSS JOIN calendar c
    WHERE c.date >= (SELECT MIN(date) FROM daily_sales)
),
joined AS (
    SELECT sp.*, d.units, d.revenue
    FROM spine sp LEFT JOIN daily_sales d
      ON d.store_id = sp.store_id AND d.dept_id = sp.dept_id AND d.date = sp.date
)
SELECT
    j.store_id, j.dept_id, j.cat_id, j.date, j.units, j.revenue,
    CASE WHEN j.units IS NULL THEN 1 ELSE 0 END                         AS is_future,
    j.wday, j.month, j.year,
    CAST(strftime('%d', j.date) AS INTEGER)                             AS day_of_month,
    CASE WHEN j.wday IN (1, 2) THEN 1 ELSE 0 END                        AS is_weekend,   -- M5: 1 = Saturday
    j.snap,
    CASE WHEN j.event_name_1 IS NOT NULL THEN 1 ELSE 0 END              AS is_event,
    COALESCE(j.event_type_1, 'None')                                    AS event_type,
    COALESCE(j.event_name_1, '')                                        AS event_name,
    LAG(j.units, {HORIZON})     OVER w                                  AS lag_28,
    LAG(j.units, {HORIZON + 7}) OVER w                                  AS lag_35,
    LAG(j.units, 364)           OVER w                                  AS lag_364,
    AVG(j.units) OVER (PARTITION BY j.store_id, j.dept_id ORDER BY j.date
                       ROWS BETWEEN {HORIZON + 6} PRECEDING AND {HORIZON} PRECEDING)   AS mean_7_lag28,
    AVG(j.units) OVER (PARTITION BY j.store_id, j.dept_id ORDER BY j.date
                       ROWS BETWEEN {HORIZON + 27} PRECEDING AND {HORIZON} PRECEDING)  AS mean_28_lag28,
    p.price_index
FROM joined j
LEFT JOIN dept_prices p
  ON p.store_id = j.store_id AND p.dept_id = j.dept_id AND p.wm_yr_wk = j.wm_yr_wk
WINDOW w AS (PARTITION BY j.store_id, j.dept_id ORDER BY j.date)
"""


def dept_price_index() -> pd.DataFrame:
    """Weekly price index per store and department, from product shelf prices."""
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"].str.startswith(STATE + "_")].copy()
    prices["dept_id"] = prices["item_id"].str.rsplit("_", n=1).str[0]          # FOODS_1_001 -> FOODS_1
    prices["relative"] = prices["sell_price"] / prices.groupby(["store_id", "item_id"])["sell_price"].transform("mean")
    return (prices.groupby(["store_id", "dept_id", "wm_yr_wk"], as_index=False)["relative"].mean()
            .rename(columns={"relative": "price_index"}).round({"price_index": 4}))


def main():
    con = sqlite3.connect(DB)
    print(f"SQLite {sqlite3.sqlite_version} (window functions need 3.25+)")
    con.execute("DROP TABLE IF EXISTS dept_prices")
    con.execute("DROP TABLE IF EXISTS features")
    dept_price_index().to_sql("dept_prices", con, index=False)
    con.execute("CREATE UNIQUE INDEX ix_prices ON dept_prices (store_id, dept_id, wm_yr_wk)")
    con.execute(FEATURES_SQL)
    con.execute("CREATE UNIQUE INDEX ix_features ON features (store_id, dept_id, date)")
    con.commit()

    stats = pd.read_sql("""
        SELECT COUNT(*) AS rows, SUM(is_future) AS future_rows,
               MIN(CASE WHEN is_future = 1 THEN date END) AS first_future_day, MAX(date) AS last_day,
               ROUND(AVG(price_index), 3) AS mean_price_index,
               SUM(is_event) / COUNT(DISTINCT store_id || dept_id) AS event_days,
               SUM(CASE WHEN lag_364 IS NULL THEN 1 ELSE 0 END) AS rows_without_lag_364
        FROM features""", con)
    sample = pd.read_sql("""
        SELECT date, units, lag_28, lag_35, ROUND(mean_7_lag28, 1) AS mean_7_lag28, lag_364,
               is_event, event_type, snap, price_index
        FROM features WHERE store_id = 'CA_3' AND dept_id = 'FOODS_3'
        ORDER BY date DESC LIMIT 32""", con)
    con.close()
    print("\n" + stats.T.to_string(header=False))
    print("\nCA_3 / FOODS_3, most recent days (future days have no units yet):")
    print(sample.iloc[::-1].tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
