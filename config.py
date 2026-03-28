"""Shared constants and settings persistence."""

import json
from pathlib import Path

# ── directories ───────────────────────────────────────────────────────────────

VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

SETTINGS_FILE = Path("settings.json")

# ── TTS ───────────────────────────────────────────────────────────────────────

PRESET_SPEAKERS = [
    "Ryan", "Aiden",
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ono_Anna",
    "Sohee",
]

LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

TTS_MODEL_IDS = {
    "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base":   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

CHAT_TTS_MAX_CHARS = 600
CHAT_RESPONSE_MAX_CHARS = 600
CHAT_MAX_EXCHANGES = 25  # auto-save + reset after this many assistant turns

# ── Ollama / Chrome ───────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# ── personalities ─────────────────────────────────────────────────────────────

PERSONALITIES = {
    "Assistant": "You are a helpful, concise assistant. Keep all replies under 600 characters.",
    "Friend":    "You are a warm, casual friend having a conversation. Be brief and natural. Keep all replies under 600 characters.",
    "Teacher":   "You are a patient teacher. Explain things clearly and simply. Keep all replies under 600 characters.",
    "Coach":     "You are an encouraging life coach. Be motivating and direct. Keep all replies under 600 characters.",
    "Comedian":  "You are a witty comedian. Be funny and light-hearted. Keep all replies under 600 characters.",
    "Custom":    "",
}

# ── settings helpers ──────────────────────────────────────────────────────────


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_settings(**kwargs) -> None:
    current = load_settings()
    current.update(kwargs)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2))
