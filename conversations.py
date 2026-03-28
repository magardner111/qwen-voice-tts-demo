"""Conversation persistence — save/load chat history as JSON files."""

import json
import time
from pathlib import Path

from config import CONVERSATIONS_DIR


def save_conversation(history: list[dict], model: str) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = CONVERSATIONS_DIR / f"{ts}.json"
    path.write_text(json.dumps({"model": model, "saved": ts, "messages": history}, indent=2))
    print(f"[conv] saved → {path.name}")
    return path


def list_conversations() -> list[str]:
    return sorted((p.stem for p in CONVERSATIONS_DIR.glob("*.json")), reverse=True)


def load_conversation(name: str) -> list[dict]:
    path = CONVERSATIONS_DIR / f"{name}.json"
    try:
        return json.loads(path.read_text()).get("messages", [])
    except Exception:
        return []
