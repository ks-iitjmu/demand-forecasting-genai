# Demand Forecasting with GenAI Executive Summaries

Forecasts 28 days of daily unit sales and revenue for every store and department of
Walmart's California stores (M5 dataset), compares Prophet and XGBoost against a
seasonal-naive baseline in a rolling backtest, and uses an LLM to turn the forecasts into
plain-language executive summaries, served in a Streamlit dashboard.

## Data

M5 Forecasting (Kaggle): daily unit sales of about 3,000 products in 10 Walmart stores,
January 2011 to May 2016, with weekly prices, holidays, sporting and cultural events, and
SNAP (food-stamp) days. This project uses the 4 California stores and forecasts at
store x department level: 4 stores x 7 departments = 28 series. Single products sell
nothing on most days; department totals are smooth and are the level a planner works at.
Revenue is computed per product as daily units x that week's shelf price, then summed.

## Status

- [x] Step 1: load M5 into SQLite (store x department daily units and revenue, calendar)
- [x] Step 2: model features with SQL window functions
- [x] Step 3: seasonal-naive baseline, Prophet and XGBoost with a rolling-origin backtest
- [x] Step 4: 28-day forecasts with prediction intervals
- [x] Step 5: LLM executive summaries with a number check
- [x] Step 6: Streamlit dashboard
- [ ] Step 7: run guide

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your Gemini API key (aistudio.google.com), or LLM_PROVIDER=ollama
set -a && source .env && set +a
```

## Step 1: data into SQLite

Put the M5 zip from Kaggle (`m5-forecasting-accuracy.zip`) in `data/raw/`, then:

```bash
python ingest/load_m5.py        # data/demand.db
python scripts/check_llm.py     # checks the summary LLM (Gemini by default)
```

| Table | Contents |
|---|---|
| `calendar` | one row per day: weekday, month, year, Walmart week, events, SNAP days (California) |
| `daily_sales` | one row per store, department and day: units sold, revenue |
| `products` | number of products per store and department |

The summary LLM is reached through the OpenAI-compatible API, so the provider is a
setting: Gemini through Google AI Studio's free tier by default, or a local model through
Ollama (`LLM_PROVIDER=ollama`).

## Step 2: features in SQL

```bash
python ingest/build_features.py   # tables features and dept_prices
```

One row per store, department and date, including the 28 calendar days after the last
day with sales, which are the days to forecast. Built with SQL window functions over a grid
of every series and date. Sales-based features use only information at least 28 days old,
so they exist when forecasting 28 days ahead:

| Feature | Meaning |
|---|---|
| `lag_28`, `lag_35` | units on the same weekday 4 and 5 weeks earlier |
| `mean_7_lag28`, `mean_28_lag28` | mean units over the 7 and 28 days ending 28 days earlier |
| `lag_364` | units on the same weekday one year (52 weeks) earlier |
| `wday`, `month`, `year`, `day_of_month`, `is_weekend` | calendar |
| `is_event`, `event_type`, `event_name`, `snap` | holidays, sporting and cultural events, food-stamp days |
| `price_index` | mean of each product's shelf price relative to its own average price (below 1.0 = discounts); prices are set in advance, so this is known for future days |

The features were verified on synthetic data where each day's sales equal its day number,
so every lag and window average can be checked exactly, including that no feature crosses
from one series into another.

## Step 3: models and backtest

```bash
python forecast/backtest.py      # about 2-3 minutes; results in results/
```

| Model | How it forecasts |
|---|---|
| Seasonal naive | repeats the last observed week; the baseline any real model must beat |
| Prophet | one model per series: trend, weekly and yearly seasonality (multiplicative), calendar events as holidays (with the day before and after), SNAP days and the price index as regressors |
| XGBoost | one global model for all 28 series on the SQL features plus series identity; it predicts each day's sales relative to the series' recent 28-day level, so large and small departments are learned together; 10% and 90% quantile models give the interval |
| Ensemble | the mean of Prophet and XGBoost |

**Rolling-origin backtest.** Three 28-day periods are forecast as if each were the future:
the origin is the last known day minus 28, 56 and 84 days, and each model trains only on
data up to its origin. Scored with WAPE (total absolute error / total sales, the standard
retail measure), bias (over- or under-forecasting, which drives stock-outs or excess
stock), RMSE, and the coverage of the 80% prediction interval (ideal: about 0.80).

Results: `results/backtest_summary.md` (overall, by department, by fold),
`results/backtest_metrics.csv`, and every forecast in `results/backtest_predictions.parquet`.

## Step 4: the 28-day forecast

```bash
python forecast/predict.py      # needs results/ from forecast/backtest.py
```

Prophet and XGBoost are trained on all known days and averaged (the ensemble, the best model
in the backtest) to forecast the 28 days after the last day with sales, for all 28 series.

**Calibrated intervals.** The ensemble's raw 80% intervals cover less than 80% in the backtest,
so each side is widened by the smallest factor that reaches 80% on the backtest forecasts.
Fitting and checking that factor on the same data would flatter it, so the reported coverage
comes from leave-one-fold-out: the factor is fitted on two backtest periods and checked on the
third, for each period in turn (`results/interval_calibration.json`).

**Revenue** = forecast units x each series' price per unit over its last 28 days, adjusted by the
planned price change, since Walmart's prices (and so the price index) are known in advance.

Output: SQLite tables `forecasts` (units and revenue with intervals, the two model forecasts,
events and SNAP days) and `forecast_calibration`, plus `results/forecast.csv`.

XGBoost can train on an NVIDIA GPU with `XGB_DEVICE=cuda` in `.env`.

## Step 5: executive summaries (LLM)

```bash
python summaries/generate.py          # 5 summaries: all stores, then each store
python summaries/generate.py --force  # rewrite even if the facts have not changed
```

**The LLM never calculates.** Python computes every fact from SQL and formats each number:
forecast units and revenue with their range, the change against the previous 28 days and the
same weeks a year earlier, the departments growing and declining most, events and SNAP days in
the period, and the model's backtest accuracy. The LLM (Gemini by default, or a local model
through Ollama) only turns these facts into a three-paragraph summary for store managers.

Each store's summary uses that store's own backtest accuracy, and the forecast range is
described as conservative: it is the sum of the daily 80% ranges, while daily errors partly
cancel over 28 days, so the true range of the 28-day total is narrower.

**Checks.** Every number in a summary must match one of its facts (dates, years and small
counts excepted), and the summary must be complete: it must end with a full sentence and have a
reasonable length. If either check fails, the LLM gets one retry that names the problem; a
summary that still fails is stored with a visible warning rather than presented as reliable.
In testing, the checks caught invented figures, rounded figures, a store summary reusing the
all-store totals, and summaries cut off mid-sentence. The cut-offs came from the model's hidden
reasoning using up the output budget; the client now retries any reply that hits the output
limit with a larger budget.

Output: SQLite table `summaries`, with each summary's text, the facts it was written from,
the verification result, the provider and model, and a hash of the facts, so summaries are
only regenerated when the forecast changes.

## Step 6: dashboard (Streamlit)

```bash
streamlit run app/dashboard.py      # http://localhost:8501
```

| Part | Shows |
|---|---|
| Filters | store (all or one), departments, units or revenue, history length |
| KPIs | expected units or revenue for the next 28 days with the change vs the previous 28 days, change vs the same weeks last year, the conservative range, and backtest error vs the naive baseline |
| Forecast | recent actual sales, the forecast with its calibrated 80% daily range, events in the period; a department table sorted by change |
| Executive summary | the LLM summary for the selected store, its verification status, and the exact facts it was written from |
| Model accuracy | every model's WAPE, spread across periods, bias and interval coverage for the current selection, and backtest forecasts against actual sales |
| Data and method | forecast download (CSV) and a short description of the method |

The dashboard reads the database in read-only mode and caches queries.
