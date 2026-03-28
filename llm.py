"""Ollama LLM communication helpers."""

import json
import urllib.error
import urllib.request

from config import OLLAMA_BASE_URL

_EMBED_KEYWORDS = ("embed", "embedding", "retrieval", "rerank")


def fetch_ollama_models() -> list[str]:
    """Return chat-capable model names from Ollama (excludes embedding models)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        return [
            m["name"] for m in data.get("models", [])
            if not any(kw in m["name"].lower() for kw in _EMBED_KEYWORDS)
        ]
    except Exception:
        return []


def extract_text(content) -> str:
    """Normalize Gradio message content (string or list) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def stream_ollama(messages: list[dict], model: str):
    """Stream from Ollama, yielding (token, done) pairs."""
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                done = chunk.get("done", False)
                yield token, done
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama: {e.reason}")


def build_messages(history: list[dict], system: str) -> list[dict]:
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.extend(
        {"role": m["role"], "content": extract_text(m["content"])}
        for m in history
    )
    return messages
