"""
Qwen3-TTS 1.7B — Gradio GUI

Tabs:
  Chat          — converse with an LLM; responses are spoken aloud in the selected voice
  Generate      — one-shot TTS with preset speaker, voice clone, or voice design
  Saved Voices  — manage voice profiles (stored as audio + transcript)

Run:
  python gui.py
  python gui.py --device cpu
  python gui.py --port 7861
"""

import argparse
import json
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

# ── constants ─────────────────────────────────────────────────────────────────

VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

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

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CHAT_TTS_MAX_CHARS = 600
CHAT_RESPONSE_MAX_CHARS = 600
CHAT_MAX_EXCHANGES = 25       # save + reset after this many assistant turns
SETTINGS_FILE = Path("settings.json")
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

PERSONALITIES = {
    "Assistant":  "You are a helpful, concise assistant. Keep all replies under 600 characters.",
    "Friend":     "You are a warm, casual friend having a conversation. Be brief and natural. Keep all replies under 600 characters.",
    "Teacher":    "You are a patient teacher. Explain things clearly and simply. Keep all replies under 600 characters.",
    "Coach":      "You are an encouraging life coach. Be motivating and direct. Keep all replies under 600 characters.",
    "Comedian":   "You are a witty comedian. Be funny and light-hearted. Keep all replies under 600 characters.",
    "Custom":     "",
}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_settings(**kwargs) -> None:
    current = load_settings()
    current.update(kwargs)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2))

# ── TTS model cache ───────────────────────────────────────────────────────────

_tts_cache: dict[str, object] = {}
_voice_prompt_cache: dict[str, object] = {}  # voice name → VoiceClonePromptItem
_device: str = "cuda:0"


def _get_tts(model_type: str):
    if model_type in _tts_cache:
        return _tts_cache[model_type]

    from qwen_tts import Qwen3TTSModel

    kwargs: dict = {"device_map": _device, "dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    print(f"[gui] loading {TTS_MODEL_IDS[model_type]} …")
    model = Qwen3TTSModel.from_pretrained(TTS_MODEL_IDS[model_type], **kwargs)
    _tts_cache[model_type] = model
    return model


# ── voice profile helpers ─────────────────────────────────────────────────────

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


# ── TTS synthesis ─────────────────────────────────────────────────────────────

def _write_wav(wavs, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wavs[0], sr)
    return tmp.name


def _synthesize(
    text: str,
    voice_mode: str,
    language: str,
    preset_speaker: str,
    preset_instruct: str,
    clone_audio,
    clone_ref_text: str,
    saved_voice: str,
    design_instruct: str,
) -> tuple[str | None, str]:
    """Core TTS call. Returns (wav_path, status)."""
    text = text.strip()
    if not text:
        return None, "No text to synthesize."

    try:
        if voice_mode == "Preset Speaker":
            model = _get_tts("custom")
            wavs, sr = model.generate_custom_voice(
                text=text, language=language,
                speaker=preset_speaker, instruct=preset_instruct.strip(),
            )

        elif voice_mode == "Clone Voice":
            if clone_audio is None:
                return None, "Please upload a reference audio file."
            if not clone_ref_text.strip():
                return None, "Please enter the transcript of the reference audio."
            model = _get_tts("base")
            wavs, sr = model.generate_voice_clone(
                text=text, language=language,
                ref_audio=clone_audio, ref_text=clone_ref_text.strip(),
            )

        elif voice_mode == "Saved Voice":
            if not saved_voice:
                return None, "No saved voice selected."
            audio_path, ref_text, _ = load_voice_profile(saved_voice)
            model = _get_tts("base")
            wavs, sr = model.generate_voice_clone(
                text=text, language=language,
                ref_audio=audio_path, ref_text=ref_text,
            )

        elif voice_mode == "Voice Design":
            if not design_instruct.strip():
                return None, "Please describe the voice."
            model = _get_tts("design")
            wavs, sr = model.generate_voice_design(
                text=text, language=language, instruct=design_instruct.strip(),
            )

        else:
            return None, f"Unknown voice mode: {voice_mode}"

    except Exception as e:
        return None, f"TTS error: {e}"

    return _write_wav(wavs, sr), "Done."


# ── Ollama helpers ────────────────────────────────────────────────────────────

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


# ── LLM chat ──────────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Normalize Gradio message content (string or list) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _stream_ollama(messages: list[dict], model: str) -> str:
    """Stream from Ollama, yielding text chunks, returning full reply."""
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


def _build_messages(history: list[dict], system: str) -> list[dict]:
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.extend(
        {"role": m["role"], "content": _extract_text(m["content"])}
        for m in history
    )
    return messages


def _add_pauses(text: str) -> str:
    """Insert pause markers so TTS breathes naturally at punctuation."""
    import re
    # Sentence pause: ". ", "! ", "? " → add ellipsis after
    text = re.sub(r'([.!?])\s+', r'\1... ', text)
    # Comma pause: ", " → add short ellipsis
    text = re.sub(r',\s+', r',.. ', text)
    return text


def _synthesize_voice(reply: str, saved_voice: str) -> str | None:
    if not saved_voice:
        return None
    try:
        audio_path, ref_text, language = load_voice_profile(saved_voice)
        model = _get_tts("base")

        if saved_voice not in _voice_prompt_cache:
            print(f"[tts] encoding voice prompt for '{saved_voice}' …")
            _voice_prompt_cache[saved_voice] = model.create_voice_clone_prompt(
                ref_audio=audio_path, ref_text=ref_text,
            )
        prompt = _voice_prompt_cache[saved_voice]

        tts_text = reply[:CHAT_TTS_MAX_CHARS].rsplit(" ", 1)[0] if len(reply) > CHAT_TTS_MAX_CHARS else reply
        tts_text = _add_pauses(tts_text)
        print(f"[tts] generating {len(tts_text)} chars …")
        wavs, sr = model.generate_voice_clone(
            text=tts_text, language=language,
            voice_clone_prompt=prompt,
        )
        return _write_wav(wavs, sr)
    except Exception as e:
        print(f"[tts] error: {e}")
        return None


# ── conversation persistence ──────────────────────────────────────────────────

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


# ── chat handler ──────────────────────────────────────────────────────────────

def chat_and_speak(
    user_msg: str,
    history: list[dict],
    chat_model: str,
    saved_voice: str,
    system_prompt: str,
):
    """Generator: holds text until TTS is ready, then shows both at once.
    Also handles conversation chunking at CHAT_MAX_EXCHANGES exchanges."""
    user_msg = user_msg.strip()
    if not user_msg:
        return

    # Check if we need to chunk before adding new message
    exchanges = sum(1 for m in history if m["role"] == "assistant")
    chunk_notice = None
    if exchanges >= CHAT_MAX_EXCHANGES:
        save_conversation(history, chat_model)
        chunk_notice = (
            f"💾 Conversation saved ({exchanges} exchanges). Starting fresh. "
            "Find it in the Conversations tab."
        )
        history = []

    history = list(history) + [{"role": "user", "content": user_msg}]

    # Show user message + thinking indicator immediately
    thinking = history + [{"role": "assistant", "content": "⏳ Thinking…"}]
    yield thinking, None

    # Collect full LLM reply (stream internally, don't surface tokens yet)
    system = system_prompt.strip()
    length_hint = f"Keep your reply under {CHAT_RESPONSE_MAX_CHARS} characters."
    system = f"{system}\n{length_hint}".strip() if system else length_hint
    messages = _build_messages(history, system)
    print(f"[llm] model={chat_model!r}, {len(messages)} messages")

    reply = ""
    try:
        for token, _ in _stream_ollama(messages, chat_model):
            reply += token
    except Exception as e:
        print(f"[llm] error: {e}")
        reply = f"[LLM error: {e}]"

    # Run TTS before showing text
    wav_path = _synthesize_voice(reply, saved_voice if saved_voice != "(none)" else "")

    # Prepend chunk notice as a system message if needed
    final_history = list(history) + [{"role": "assistant", "content": reply}]
    if chunk_notice:
        final_history = [{"role": "assistant", "content": chunk_notice}] + final_history

    yield final_history, wav_path


# ── standalone generate ────────────────────────────────────────────────────────

def generate_speech(
    voice_mode, text, language,
    preset_speaker, preset_instruct,
    clone_audio, clone_ref_text, clone_save_name,
    saved_voice, design_instruct,
):
    save_name = clone_save_name.strip() if clone_save_name else ""
    if voice_mode == "Clone Voice" and save_name:
        if any(c in save_name for c in r'\/:*?"<>|'):
            return None, "Voice name contains invalid characters."
        if clone_audio:
            save_voice_profile(save_name, clone_audio, clone_ref_text.strip(), language)

    return _synthesize(
        text, voice_mode, language,
        preset_speaker, preset_instruct,
        clone_audio, clone_ref_text,
        saved_voice, design_instruct,
    )


# ── saved voices table ────────────────────────────────────────────────────────

def refresh_voice_table():
    rows = []
    for name in list_saved_voices():
        try:
            meta = json.loads(_profile_meta(name).read_text())
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("created", 0)))
            rows.append([name, meta.get("language", ""), created])
        except Exception:
            rows.append([name, "", ""])
    return rows


def delete_voice_action(name: str):
    name = name.strip()
    if not name:
        return "Enter a voice name to delete.", refresh_voice_table()
    if delete_voice_profile(name):
        return f"Deleted '{name}'.", refresh_voice_table()
    return f"Voice '{name}' not found.", refresh_voice_table()


# ── shared voice settings block ───────────────────────────────────────────────

def _voice_settings_block(saved_voices_initial: list[str]):
    """Returns a dict of Gradio components for voice selection."""
    voice_mode = gr.Radio(
        label="Voice",
        choices=["Preset Speaker", "Clone Voice", "Saved Voice", "Voice Design"],
        value="Preset Speaker",
    )
    language = gr.Dropdown(label="Language", choices=LANGUAGES, value="English")

    with gr.Group(visible=True) as preset_group:
        preset_speaker = gr.Dropdown(label="Speaker", choices=PRESET_SPEAKERS, value="Ryan")
        preset_instruct = gr.Textbox(label="Style instruction (optional)", placeholder="e.g. Speak warmly.")

    with gr.Group(visible=False) as clone_group:
        clone_audio = gr.Audio(label="Reference audio (≥ 3 s)", type="filepath", sources=["upload", "microphone"])
        clone_ref_text = gr.Textbox(label="Transcript", lines=2, placeholder="Exact words spoken in the clip…")

    with gr.Group(visible=False) as saved_group:
        saved_voice = gr.Dropdown(label="Voice profile", choices=saved_voices_initial, value=None)
        gr.Button("Refresh", size="sm").click(
            fn=lambda: gr.update(choices=list_saved_voices()), outputs=saved_voice
        )

    with gr.Group(visible=False) as design_group:
        design_instruct = gr.Textbox(label="Describe the voice", lines=2, placeholder="e.g. Deep, calm British male.")

    def _update(mode):
        return (
            gr.update(visible=(mode == "Preset Speaker")),
            gr.update(visible=(mode == "Clone Voice")),
            gr.update(visible=(mode == "Saved Voice")),
            gr.update(visible=(mode == "Voice Design")),
        )

    voice_mode.change(
        fn=_update,
        inputs=voice_mode,
        outputs=[preset_group, clone_group, saved_group, design_group],
    )

    return dict(
        voice_mode=voice_mode, language=language,
        preset_speaker=preset_speaker, preset_instruct=preset_instruct,
        clone_audio=clone_audio, clone_ref_text=clone_ref_text,
        saved_voice=saved_voice, design_instruct=design_instruct,
    )


# ── styling ───────────────────────────────────────────────────────────────────

AUTOPLAY_JS = """
() => {
    // Poll for new audio srcs and play them immediately
    const played = new Set();
    setInterval(() => {
        document.querySelectorAll('audio').forEach(a => {
            if (a.src && !played.has(a.src)) {
                played.add(a.src);
                a.play().catch(() => {});
            }
        });
    }, 300);
}
"""

CSS = """
/* ── global ── */
body, .gradio-container { background: #0f1117 !important; }
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; }

/* ── header ── */
#app-header {
    text-align: center;
    padding: 28px 0 12px;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 8px;
}
#app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
    margin: 0;
}
#app-header p {
    color: #64748b;
    font-size: 0.85rem;
    margin: 4px 0 0;
}

/* ── tabs ── */
.tab-nav button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 20px !important;
}
.tab-nav button.selected {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* ── toolbar row (dropdowns + refresh) ── */
#chat-toolbar {
    background: #161b27;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    align-items: flex-end !important;
}
#chat-toolbar label { color: #94a3b8 !important; font-size: 0.75rem !important; }
#chat-toolbar select, #chat-toolbar input {
    background: #0f1117 !important;
    border: 1px solid #2d3148 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}

/* ── chatbot ── */
#chatbox {
    border: 1px solid #1e2130 !important;
    border-radius: 10px !important;
    background: #161b27 !important;
}
#chatbox .message.user { background: #1e2540 !important; border-radius: 10px !important; }
#chatbox .message.bot  { background: #161b27 !important; border-radius: 10px !important; }
#chatbox .message { color: #cbd5e1 !important; font-size: 0.9rem !important; }

/* ── input row ── */
#chat-input-row {
    background: #161b27;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 8px;
    align-items: center !important;
    gap: 8px !important;
}
#chat-input-row textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    resize: none !important;
}
#chat-input-row textarea:focus { outline: none !important; }

/* ── buttons ── */
button.primary {
    background: #6366f1 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #fff !important;
}
button.primary:hover { background: #4f52d1 !important; }
#clear-btn {
    background: transparent !important;
    border: 1px solid #2d3148 !important;
    color: #64748b !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
    padding: 4px 12px !important;
}
#clear-btn:hover { border-color: #6366f1 !important; color: #a5b4fc !important; }

/* ── audio player (half size) ── */
#chat-audio {
    transform: scale(0.5);
    transform-origin: left top;
    height: 36px !important;
    overflow: hidden;
    margin-bottom: -18px;
}

/* ── generate tab ── */
#gen-panel {
    background: #161b27;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 16px;
}

/* ── labels / inputs globally ── */
label span { color: #94a3b8 !important; font-size: 0.78rem !important; font-weight: 500 !important; }
input, textarea, select {
    background: #0f1117 !important;
    border-color: #2d3148 !important;
    color: #e2e8f0 !important;
}
.form { border-color: #1e2130 !important; background: #161b27 !important; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    saved = list_saved_voices()

    with gr.Blocks(title="Voice Chat") as demo:
        gr.HTML("""
            <div id="app-header">
                <h1>Voice Chat</h1>
                <p>Local LLM &nbsp;·&nbsp; Qwen3-TTS &nbsp;·&nbsp; Ollama</p>
            </div>
        """)

        with gr.Tabs():

            # ── Chat ───────────────────────────────────────────────────────
            with gr.Tab("Chat"):
                _initial_models = fetch_ollama_models()
                with gr.Row(elem_id="chat-toolbar"):
                    chat_model = gr.Dropdown(
                        label="Chat Model",
                        choices=_initial_models,
                        value=_initial_models[0] if _initial_models else None,
                        allow_custom_value=True,
                        scale=3,
                    )
                    chat_voice = gr.Dropdown(
                        label="Voice",
                        choices=["(none)"] + list_saved_voices(),
                        value="(none)",
                        scale=3,
                    )
                    refresh_chat_btn = gr.Button("⟳ Refresh", size="sm", scale=1)

                def _refresh_chat_dropdowns():
                    models = fetch_ollama_models()
                    voices = ["(none)"] + list_saved_voices()
                    return gr.update(choices=models), gr.update(choices=voices)

                refresh_chat_btn.click(
                    fn=_refresh_chat_dropdowns,
                    outputs=[chat_model, chat_voice],
                )

                with gr.Accordion("⚙ Settings", open=False):
                    _s = load_settings()
                    _saved_personality = _s.get("personality", "Assistant")
                    _saved_prompt = _s.get("system_prompt", PERSONALITIES["Assistant"])

                    personality = gr.Dropdown(
                        label="Personality",
                        choices=list(PERSONALITIES.keys()),
                        value=_saved_personality,
                    )
                    system_prompt = gr.Textbox(
                        label="System prompt",
                        lines=3,
                        value=_saved_prompt,
                        interactive=(_saved_personality == "Custom"),
                        placeholder="Describe how the AI should behave…",
                    )

                    def _on_personality_change(p):
                        prompt = PERSONALITIES[p] if p != "Custom" else load_settings().get("system_prompt", "")
                        save_settings(personality=p)
                        return gr.update(value=prompt, interactive=(p == "Custom"))

                    def _on_prompt_change(prompt):
                        save_settings(system_prompt=prompt)

                    personality.change(fn=_on_personality_change, inputs=personality, outputs=system_prompt)
                    system_prompt.change(fn=_on_prompt_change, inputs=system_prompt)

                chatbot = gr.Chatbot(height=440, elem_id="chatbox", show_label=False)
                chat_audio = gr.Audio(
                    type="filepath", autoplay=True, show_label=False,
                    elem_id="chat-audio", waveform_options={"waveform_progress_color": "#6366f1"},
                )

                with gr.Row(elem_id="chat-input-row"):
                    chat_input = gr.Textbox(
                        placeholder="Message…",
                        show_label=False,
                        lines=1,
                        max_lines=6,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear", size="sm", elem_id="clear-btn")
                    save_btn = gr.Button("Save conversation", size="sm")

                def _send(user_msg, history, model, voice, sys_prompt):
                    yield from chat_and_speak(user_msg, history, model, voice, sys_prompt)

                def _save(history, model):
                    if history:
                        save_conversation(history, model)

                send_inputs = [chat_input, chatbot, chat_model, chat_voice, system_prompt]

                send_btn.click(fn=_send, inputs=send_inputs, outputs=[chatbot, chat_audio])
                chat_input.submit(fn=_send, inputs=send_inputs, outputs=[chatbot, chat_audio])
                clear_btn.click(fn=lambda: ([], None), outputs=[chatbot, chat_audio])
                save_btn.click(fn=_save, inputs=[chatbot, chat_model])

            # ── Conversations ──────────────────────────────────────────────
            with gr.Tab("Conversations"):
                gr.Markdown("Saved conversations. Click one to load it back into chat.")
                conv_list = gr.Dropdown(
                    label="Saved conversations",
                    choices=list_conversations(),
                    value=None,
                    allow_custom_value=False,
                )
                with gr.Row():
                    load_conv_btn = gr.Button("Load into Chat", variant="primary")
                    refresh_conv_btn = gr.Button("Refresh")
                conv_viewer = gr.Chatbot(label="Preview", height=380)

                def _preview_conv(name):
                    if not name:
                        return []
                    return load_conversation(name)

                def _load_conv(name):
                    msgs = load_conversation(name)
                    return msgs, None  # load into chatbot, clear audio

                conv_list.change(fn=_preview_conv, inputs=conv_list, outputs=conv_viewer)
                refresh_conv_btn.click(
                    fn=lambda: gr.update(choices=list_conversations()), outputs=conv_list
                )
                load_conv_btn.click(
                    fn=_load_conv, inputs=conv_list, outputs=[chatbot, chat_audio]
                )

            # ── Generate ───────────────────────────────────────────────────
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gv = _voice_settings_block(saved)
                        gen_text = gr.Textbox(label="Text to synthesize", lines=4, placeholder="Enter text…")
                        clone_save_name = gr.Textbox(
                            label="Save clone as voice profile (optional)",
                            placeholder="Leave blank to skip",
                            visible=False,
                        )
                        gv["voice_mode"].change(
                            fn=lambda m: gr.update(visible=(m == "Clone Voice")),
                            inputs=gv["voice_mode"],
                            outputs=clone_save_name,
                        )
                        gen_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        gen_audio = gr.Audio(label="Output", type="filepath")
                        gen_status = gr.Textbox(label="Status", interactive=False)

                gen_btn.click(
                    fn=generate_speech,
                    inputs=[
                        gv["voice_mode"], gen_text, gv["language"],
                        gv["preset_speaker"], gv["preset_instruct"],
                        gv["clone_audio"], gv["clone_ref_text"], clone_save_name,
                        gv["saved_voice"], gv["design_instruct"],
                    ],
                    outputs=[gen_audio, gen_status],
                )

            # ── Saved Voices ───────────────────────────────────────────────
            with gr.Tab("Saved Voices"):
                gr.Markdown("Voice profiles store raw audio + transcript. Encoding happens at inference time.")
                voices_table = gr.Dataframe(
                    headers=["Name", "Language", "Created"],
                    datatype=["str", "str", "str"],
                    value=refresh_voice_table(),
                    interactive=False,
                )
                with gr.Row():
                    delete_name = gr.Textbox(label="Voice name to delete", scale=3)
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)
                refresh_btn = gr.Button("Refresh")
                manage_status = gr.Textbox(label="Status", interactive=False)

                delete_btn.click(fn=delete_voice_action, inputs=delete_name, outputs=[manage_status, voices_table])
                refresh_btn.click(fn=refresh_voice_table, outputs=voices_table)

    return demo


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS 1.7B GUI")
    p.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0, cpu, mps")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-chrome-app", action="store_true", help="Open in default browser instead of Chrome app")
    return p.parse_args()


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

    # Wait until it's reachable (up to 10 s)
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


def _open_chrome_app(url: str) -> None:
    profile_dir = Path(tempfile.gettempdir()) / "voice-chat-chrome-profile"
    profile_dir.mkdir(exist_ok=True)
    subprocess.Popen(
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


if __name__ == "__main__":
    args = parse_args()
    _device = args.device

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

    # Wait until Gradio is actually serving
    print(f"[app] waiting for server at {url} …")
    for _ in range(40):
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    if not args.no_chrome_app:
        _open_chrome_app(url)
        print(f"[app] opened Chrome app at {url}")

    def _shutdown(sig=None, frame=None):
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
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[app] running — press Ctrl+C to quit")
    threading.Event().wait()  # block forever until signal
