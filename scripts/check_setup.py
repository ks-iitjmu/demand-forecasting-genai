"""Check that everything the project needs is in place.   python scripts/check_setup.py"""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
OK, BAD, TODO = "[ OK ]", "[FAIL]", "[TODO]"
problems = 0


def line(status, msg):
    global problems
    problems += status == BAD
    print(f"{status} {msg}")


v = sys.version_info
line(OK if v >= (3, 10) else BAD, f"Python {v.major}.{v.minor}" + ("" if v >= (3, 10) else " (3.10 or newer needed)"))
line(OK if sys.prefix != sys.base_prefix else BAD, "virtual environment active" if sys.prefix != sys.base_prefix
     else "no virtual environment: run  source .venv/bin/activate")
missing = [m for m in ["pandas", "numpy", "pyarrow", "prophet", "xgboost", "sklearn", "openai", "streamlit",
                       "plotly", "tabulate"] if importlib.util.find_spec(m) is None]
line(BAD if missing else OK, f"packages missing: {', '.join(missing)}  ->  pip install -r requirements.txt"
     if missing else "all Python packages installed")

from config import LLM_MODEL, LLM_PROVIDER, RAW  # noqa: E402
if LLM_PROVIDER == "gemini":
    line(OK if os.environ.get("GEMINI_API_KEY") else BAD, f"Gemini key set (model {LLM_MODEL})" if os.environ.get("GEMINI_API_KEY")
         else "GEMINI_API_KEY not set: add it to .env, then  set -a && source .env && set +a")
else:
    line(OK, f"LLM provider {LLM_PROVIDER} (model {LLM_MODEL}); check it with  python scripts/check_llm.py")

has_m5 = any(RAW.glob("*.zip")) or (RAW / "sales_train_evaluation.csv").exists() or (RAW / "sales_train_validation.csv").exists()
line(OK if has_m5 else BAD, "M5 data found in data/raw/" if has_m5 else
     "M5 data missing: put m5-forecasting-accuracy.zip from Kaggle in data/raw/")

print("\nPipeline outputs so far:")
for path, step in [("data/demand.db", "python ingest/load_m5.py"),
                   ("results/backtest_predictions.parquet", "python forecast/backtest.py"),
                   ("results/forecast.csv", "python forecast/predict.py")]:
    exists = (ROOT / path).exists()
    print(f"{OK if exists else TODO} {path}" + ("" if exists else f"   <- {step}"))
print("\n" + ("Ready." if not problems else f"{problems} problem(s) to fix first.")
      + "  Run the whole pipeline with:  bash scripts/run_all.sh")
