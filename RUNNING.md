# Running the project

Everything runs locally; the only online service is the LLM for the summaries (Gemini's free
tier, or a local model through Ollama). For what the project does and its results, see
[README.md](README.md).

## What you need

| | Requirement | Notes |
|---|---|---|
| OS | Linux, macOS or Windows (WSL2) | tested on Ubuntu |
| Python | 3.10 or newer | |
| Memory | about 4 GB free | loading the M5 sales table |
| Data | M5 Forecasting from Kaggle | free Kaggle account; accept the competition rules once |
| LLM | a Gemini API key (free tier), or Ollama with a local model | only for the summaries |
| GPU | optional | XGBoost can use an NVIDIA GPU (`XGB_DEVICE=cuda`) |
| Time | about 10 minutes for the full pipeline | most of it is the backtest |

## 1. Get the code and set up Python

```bash
git clone https://github.com/<your-username>/demand-forecasting-genai.git
cd demand-forecasting-genai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # OR
pip install -r requirements-lock.txt     # only if you want exact versions used for the reported results
```

## 2. Get the data

Download the dataset from kaggle.com/competitions/m5-forecasting-accuracy/data
(`m5-forecasting-accuracy.zip`) and put the zip in `data/raw/`:
```bash
mkdir -p data/raw
mv ~/Downloads/m5-forecasting-accuracy.zip data/raw/
```

## 3. Configure the LLM

```bash
cp .env.example .env
```
Edit `.env`:
- **Gemini** (default): get a free key at aistudio.google.com (Get API key) and set
  `GEMINI_API_KEY`. `LLM_MODEL` is pinned for reproducible summaries; if Google has retired that
  model, `python scripts/check_llm.py` lists the models your key can use.
- **Local model instead**: set `LLM_PROVIDER=ollama` and `LLM_MODEL=qwen3:14b` (or another
  model you have pulled with Ollama). No key needed.

Load the settings. **Repeat this in every new terminal:**
```bash
set -a && source .env && set +a
python scripts/check_llm.py        # should print: The LLM connection works.
python scripts/check_setup.py      # every line should say [ OK ]
```

## 4. Run the pipeline

All at once:
```bash
bash scripts/run_all.sh
```
Or step by step:

| Step | Command | Time | Output |
|---|---|---|---|
| Load California sales into SQLite | `python ingest/load_m5.py` | 1-2 min | `data/demand.db` |
| Build features with SQL | `python ingest/build_features.py` | seconds | tables `features`, `dept_prices` |
| Backtest the models | `python forecast/backtest.py` | 2-5 min | `results/backtest_*` |
| Forecast the next 28 days | `python forecast/predict.py` | 1-2 min | table `forecasts`, `results/forecast.csv` |
| Write the summaries | `python summaries/generate.py` | about 1 min | table `summaries` |

## 5. Open the dashboard

```bash
streamlit run app/dashboard.py     # http://localhost:8501
```

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `GEMINI_API_KEY is not set` | The terminal has not loaded `.env`: `set -a && source .env && set +a` |
| `404 ... model ... is no longer available` | Google retired that model: run `python scripts/check_llm.py` and set `LLM_MODEL` to one it lists |
| `server error 503` on every attempt | The model is overloaded: try again later, or use another model (`LLM_MODEL=...`) |
| `rate limited (429)` | The free tier allows a few requests per minute; the client waits and retries by itself |
| A summary is marked "contains unverified numbers" | The LLM used a number not in the facts even after a retry; run `python summaries/generate.py --force` |
| `sklearn needs to be installed` | `pip install scikit-learn` (it is in requirements.txt) |
| XGBoost is slow | Train on the GPU: add `XGB_DEVICE=cuda` to `.env` |
| `No ensemble forecasts in the backtest` | Run `python forecast/backtest.py` before `forecast/predict.py` |
| The dashboard says "No data yet" | Run the pipeline first (step 4) |

## Project layout

```
config.py            settings (paths, horizon, LLM provider and model), read from .env
ingest/              M5 -> SQLite; features with SQL window functions
forecast/            models (seasonal naive, Prophet, XGBoost), backtest, final forecast
summaries/           facts from SQL, LLM summaries, number and completeness checks
core/llm.py          LLM client for any OpenAI-compatible provider (Gemini, Ollama)
app/dashboard.py     Streamlit dashboard
scripts/             LLM check, setup check, one-command pipeline
results/             backtest and forecast results (committed)
```
