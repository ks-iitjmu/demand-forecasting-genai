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
- [ ] Step 3: seasonal-naive baseline, Prophet and XGBoost with a rolling-origin backtest
- [ ] Step 4: 28-day forecasts with prediction intervals
- [ ] Step 5: LLM executive summaries with a number check
- [ ] Step 6: Streamlit dashboard
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
