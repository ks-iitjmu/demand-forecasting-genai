"""Project-wide settings, read from environment variables where they vary by machine."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                    # the M5 zip, or its extracted CSV files
DB = DATA / "demand.db"               # SQLite database
RESULTS = ROOT / "results"

STATE = "CA"                          # stores in California: CA_1 .. CA_4
HORIZON = 28                          # days ahead, as in the M5 competition
BACKTEST_FOLDS = 3                    # rolling-origin backtest: three 28-day periods
INTERVAL_TARGET = 0.80                # prediction intervals are calibrated to this coverage
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cpu")   # "cuda" trains XGBoost on an NVIDIA GPU

# LLM for executive summaries. Any OpenAI-compatible endpoint works:
#   gemini (default): Google AI Studio's OpenAI-compatible endpoint, free tier
#   ollama:           a local model, e.g. qwen3:14b, no key needed
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash" if LLM_PROVIDER == "gemini" else "qwen3:14b")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
