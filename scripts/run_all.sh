#!/usr/bin/env bash
# Run the whole pipeline, in order (about 5-10 minutes; most of it is the backtest).
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

step() { echo; echo "==== $1"; }
step "1/6  Checking the setup";                  python scripts/check_setup.py
step "2/6  Loading M5 into SQLite";              python ingest/load_m5.py
step "3/6  Building features in SQL";            python ingest/build_features.py
step "4/6  Backtesting the models";              python forecast/backtest.py
step "5/6  Forecasting the next 28 days";        python forecast/predict.py
step "6/6  Writing the executive summaries";     python summaries/generate.py
echo; echo "Done. Open the dashboard with:  streamlit run app/dashboard.py"
