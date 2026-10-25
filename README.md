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
- [ ] Step 2: model features with SQL window functions
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
