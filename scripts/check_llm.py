"""Check that the summary LLM works.   python scripts/check_llm.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LLM_MODEL, LLM_PROVIDER  # noqa: E402
from core.llm import chat, client  # noqa: E402

print(f"Provider: {LLM_PROVIDER}   model: {LLM_MODEL}")
try:
    reply = chat("You are a concise assistant.", "Reply with exactly: connection works", max_tokens=20)
    print(f"Reply: {reply!r}\nThe LLM connection works.")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {str(e)[:300]}")
    try:
        names = sorted(m.id for m in client().models.list())
        print("Models available to this key (set LLM_MODEL in .env to one of them):")
        for n in names:
            if "gemini" in n or "gemma" in n or LLM_PROVIDER == "ollama":
                print("  ", n.replace("models/", ""))
    except Exception:
        print("Could not list models either: check the API key and the network connection.")
    sys.exit(1)
