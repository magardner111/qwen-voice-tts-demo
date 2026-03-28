"""
Qwen3-TTS 1.7B — Voice Chat

Tabs:
  Chat          — converse with an LLM; responses are spoken aloud in the selected voice
  Conversations — browse and reload saved chat histories
  Generate      — one-shot TTS with preset speaker, voice clone, or voice design
  Saved Voices  — manage voice profiles (stored as audio + transcript)

Run:
  ./run.sh
  python gui.py --device mps
  python gui.py --device cpu --port 7861
"""

import argparse
import signal
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import gradio as gr

import tts as tts_module
from config import CHROME, OLLAMA_BASE_URL
from ui import AUTOPLAY_JS, CSS, build_ui


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS 1.7B GUI")
    p.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0, cpu, mps")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-chrome-app", action="store_true", help="Open in default browser instead of Chrome app")
    return p.parse_args()


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _start_ollama() -> subprocess.Popen | None:
    """Start `ollama serve` if not already running. Returns the process or None."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2):
            print("[ollama] already running")
            return None
    except Exception:
        pass

    print("[ollama] starting ollama serve …")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(20):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=1):
                print("[ollama] ready")
                return proc
        except Exception:
            pass

    print("[ollama] warning: ollama did not start in time")
    return proc


# ── Chrome app launcher ───────────────────────────────────────────────────────

def _open_chrome_app(url: str) -> subprocess.Popen:
    profile_dir = Path(tempfile.gettempdir()) / "voice-chat-chrome-profile"
    profile_dir.mkdir(exist_ok=True)
    return subprocess.Popen(
        [
            CHROME,
            f"--app={url}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    tts_module._device = args.device  # set before any model is loaded

    ollama_proc = _start_ollama()

    url = f"http://{args.host}:{args.port}"
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.no_chrome_app,
        prevent_thread_lock=True,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css=CSS,
        js=AUTOPLAY_JS,
    )

    print(f"[app] waiting for server at {url} …")
    for _ in range(40):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    _stop = threading.Event()
    _shutting_down = threading.Event()

    def _shutdown(sig=None, frame=None):
        if _shutting_down.is_set():
            return
        _shutting_down.set()
        print("\n[app] shutting down …")
        if ollama_proc is not None:
            print("[ollama] stopping …")
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()
        demo.close()
        print("[app] done")
        _stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not args.no_chrome_app:
        chrome_proc = _open_chrome_app(url)
        print(f"[app] opened Chrome app at {url}")

        def _watch_chrome():
            chrome_proc.wait()
            print("[app] Chrome window closed")
            _shutdown()

        threading.Thread(target=_watch_chrome, daemon=True).start()

    print("[app] running — press Ctrl+C to quit")
    _stop.wait()
