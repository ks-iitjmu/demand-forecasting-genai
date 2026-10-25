"""One small client for the summary LLM, through the OpenAI-compatible API.

Gemini (Google AI Studio) and Ollama both offer an OpenAI-compatible endpoint, so the same
code works for either; switching is a matter of LLM_PROVIDER in .env.

Retries are handled here, visibly, and the SDK's own silent retries are switched off, so a
failing provider is reported within about a minute instead of after many minutes of silence.
"""
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import GEMINI_API_KEY, LLM_MODEL, LLM_PROVIDER, OLLAMA_URL  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@lru_cache(maxsize=1)
def client() -> OpenAI:
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is not set. Add it to .env, then run:  set -a && source .env && set +a")
        return OpenAI(base_url=GEMINI_URL, api_key=GEMINI_API_KEY, timeout=45, max_retries=0)
    if LLM_PROVIDER == "ollama":
        return OpenAI(base_url=f"{OLLAMA_URL.rstrip('/')}/v1", api_key="ollama", timeout=600, max_retries=0)
    raise SystemExit(f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}: use gemini or ollama")


class LLMUnavailable(RuntimeError):
    """The provider kept failing; the message says what to try."""


def chat(system: str, user: str, max_tokens: int = 2048, temperature: float = 0.2, attempts: int = 4) -> str:
    """Return the model's text. Retries rate limits (429), overload (503), timeouts and empty
    replies (waits of 5, 10, 20 s, printing each retry); gives up with a clear message."""
    wait, last = 5, ""
    for attempt in range(1, attempts + 1):
        try:
            r = client().chat.completions.create(
                model=LLM_MODEL, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            choice = r.choices[0]
            text = re.sub(r"<think>.*?</think>", "", choice.message.content or "", flags=re.S).strip()
            if text:
                return text
            # Thinking models spend hidden reasoning tokens from the same output budget; if the
            # budget ran out before any visible text, retry with a larger one.
            if choice.finish_reason == "length" and max_tokens < 16000:
                max_tokens *= 4
                last = "the output limit was used up by hidden reasoning"
                print(f"  attempt {attempt}/{attempts}: {last}; retrying with max_tokens={max_tokens}", flush=True)
                continue
            last = f"empty reply (finish_reason={choice.finish_reason})"
        except RateLimitError:
            last = "rate limited (429): the free tier allows only a few requests per minute"
        except (APITimeoutError, APIConnectionError) as e:
            last = f"no response ({type(e).__name__})"
        except APIStatusError as e:
            if e.status_code < 500:
                raise                                    # wrong key, unknown model: retrying will not help
            last = f"server error {e.status_code}: the model is overloaded or unavailable"
        if attempt < attempts:
            print(f"  attempt {attempt}/{attempts}: {last}; retrying in {wait} s ...", flush=True)
            time.sleep(wait)
            wait *= 2
    raise LLMUnavailable(f"{LLM_MODEL} failed {attempts} times ({last}). Try again later, or another "
                         f"model: LLM_MODEL=gemini-3.5-flash, or LLM_PROVIDER=ollama for a local model.")
