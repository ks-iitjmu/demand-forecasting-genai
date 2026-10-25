"""One small client for the summary LLM, through the OpenAI-compatible API.

Gemini (Google AI Studio) and Ollama both offer an OpenAI-compatible endpoint, so the same
code works for either; switching is a matter of LLM_PROVIDER in .env.
"""
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from openai import APIStatusError, OpenAI, RateLimitError

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import GEMINI_API_KEY, LLM_MODEL, LLM_PROVIDER, OLLAMA_URL  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is not set. Add it to .env, then run:  set -a && source .env && set +a")
        return OpenAI(base_url=GEMINI_URL, api_key=GEMINI_API_KEY, timeout=120)
    if LLM_PROVIDER == "ollama":
        return OpenAI(base_url=f"{OLLAMA_URL.rstrip('/')}/v1", api_key="ollama", timeout=600)
    raise SystemExit(f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}: use gemini or ollama")


def chat(system: str, user: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    """Return the model's text. Retries rate limits (the free tier allows a few requests
    per minute) and temporary server errors with increasing waits."""
    wait = 10
    for attempt in range(5):
        try:
            r = client().chat.completions.create(
                model=LLM_MODEL, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = r.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # local reasoning models
        except RateLimitError:
            if attempt == 4:
                raise
            print(f"  rate limited, waiting {wait} s ...")
        except APIStatusError as e:
            if e.status_code < 500 or attempt == 4:
                raise
            print(f"  server error {e.status_code}, retrying in {wait} s ...")
        time.sleep(wait)
        wait *= 2
    raise RuntimeError("unreachable")
