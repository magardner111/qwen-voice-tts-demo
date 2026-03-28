"""Voice profile storage — filesystem I/O only, no TTS model access."""

import json
import shutil
import time
from pathlib import Path

from config import VOICES_DIR


# ── private path helpers ──────────────────────────────────────────────────────

def _profile_dir(name: str) -> Path:
    return VOICES_DIR / name


def _profile_meta(name: str) -> Path:
    return _profile_dir(name) / "meta.json"


def _profile_audio(name: str) -> Path:
    d = _profile_dir(name)
    for ext in ("wav", "mp3", "flac", "ogg", "m4a"):
        p = d / f"ref.{ext}"
        if p.exists():
            return p
    return d / "ref.wav"


# ── public API ────────────────────────────────────────────────────────────────

def list_saved_voices() -> list[str]:
    if not VOICES_DIR.exists():
        return []
    return sorted(
        p.name for p in VOICES_DIR.iterdir()
        if p.is_dir() and (p / "meta.json").exists()
    )


def save_voice_profile(name: str, audio_src: str, ref_text: str, language: str) -> None:
    d = _profile_dir(name)
    d.mkdir(exist_ok=True)
    dest = d / ("ref" + Path(audio_src).suffix)
    shutil.copy2(audio_src, dest)
    meta = {"ref_text": ref_text, "language": language, "created": time.time()}
    _profile_meta(name).write_text(json.dumps(meta, indent=2))


def load_voice_profile(name: str) -> tuple[str, str, str]:
    """Returns (audio_path, ref_text, language)."""
    meta = json.loads(_profile_meta(name).read_text())
    return str(_profile_audio(name)), meta["ref_text"], meta.get("language", "Auto")


def delete_voice_profile(name: str) -> bool:
    d = _profile_dir(name)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False
