"""
Qwen3-TTS 1.7B — Gradio GUI
Tabs:
  Train Voice   — upload audio + transcript → save a named voice profile (.pt)
  Generate      — pick saved voice or preset speaker, type text, synthesize
  Manage Voices — list and delete saved voices

Run:
  python gui.py
  python gui.py --device cpu          # for machines without CUDA
  python gui.py --port 7861
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import soundfile as sf
import torch

# ── constants ─────────────────────────────────────────────────────────────────

VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

PRESET_SPEAKERS = [
    "Ryan", "Aiden",                                           # English
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",          # Chinese
    "Ono_Anna",                                                # Japanese
    "Sohee",                                                   # Korean
]

LANGUAGES = [
    "Auto", "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

MODEL_IDS = {
    "custom":  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "base":    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design":  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# ── model cache ───────────────────────────────────────────────────────────────

_model_cache: dict[str, object] = {}
_device: str = "cuda:0"


def _get_model(model_type: str):
    if model_type in _model_cache:
        return _model_cache[model_type]

    from qwen_tts import Qwen3TTSModel

    kwargs: dict = {"device_map": _device, "dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    print(f"[gui] loading {MODEL_IDS[model_type]} …")
    model = Qwen3TTSModel.from_pretrained(MODEL_IDS[model_type], **kwargs)
    _model_cache[model_type] = model
    return model


# ── voice file helpers ─────────────────────────────────────────────────────────

def _voice_pt(name: str) -> Path:
    return VOICES_DIR / f"{name}.pt"


def _voice_meta(name: str) -> Path:
    return VOICES_DIR / f"{name}.json"


def list_saved_voices() -> list[str]:
    return sorted(p.stem for p in VOICES_DIR.glob("*.pt"))


def save_voice(name: str, prompt_items, ref_text: str, language: str) -> None:
    torch.save(prompt_items, _voice_pt(name))
    meta = {"ref_text": ref_text, "language": language, "created": time.time()}
    _voice_meta(name).write_text(json.dumps(meta, indent=2))


def load_voice(name: str):
    pt_path = _voice_pt(name)
    if not pt_path.exists():
        raise FileNotFoundError(f"Voice '{name}' not found in {VOICES_DIR}/")
    return torch.load(pt_path, weights_only=False)


def delete_voice(name: str) -> bool:
    deleted = False
    for path in (_voice_pt(name), _voice_meta(name)):
        if path.exists():
            path.unlink()
            deleted = True
    return deleted


# ── synthesis helpers ──────────────────────────────────────────────────────────

def _write_wav(wavs, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wavs[0], sr)
    return tmp.name


# ── tab: Train Voice ──────────────────────────────────────────────────────────

def train_voice(audio_path: str, ref_text: str, voice_name: str, language: str):
    voice_name = voice_name.strip()
    if not audio_path:
        return "Please upload a reference audio file."
    if not ref_text.strip():
        return "Please enter the transcript of the reference audio."
    if not voice_name:
        return "Please enter a name for this voice."
    if any(c in voice_name for c in r'\/:*?"<>|'):
        return "Voice name contains invalid characters."

    yield f"Loading model…"
    try:
        model = _get_model("base")
    except Exception as e:
        yield f"Error loading model: {e}"
        return

    yield "Encoding reference audio…"
    try:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=ref_text.strip(),
            x_vector_only_mode=False,
        )
        save_voice(voice_name, prompt_items, ref_text.strip(), language)
    except Exception as e:
        yield f"Error: {e}"
        return

    yield f"Voice '{voice_name}' saved successfully to voices/{voice_name}.pt"


# ── tab: Generate Speech ──────────────────────────────────────────────────────

def generate_speech(
    mode: str,
    text: str,
    language: str,
    # preset
    preset_speaker: str,
    preset_instruct: str,
    # saved voice
    saved_voice: str,
    # voice design
    design_instruct: str,
):
    text = text.strip()
    if not text:
        return None, "Please enter some text."

    try:
        if mode == "Preset Speaker":
            model = _get_model("custom")
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=preset_speaker,
                instruct=preset_instruct.strip(),
            )

        elif mode == "Saved Voice":
            if not saved_voice:
                return None, "No saved voice selected. Train one in the Train tab first."
            model = _get_model("base")
            prompt_items = load_voice(saved_voice)
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt_items,
            )

        elif mode == "Voice Design":
            if not design_instruct.strip():
                return None, "Please describe the voice you want."
            model = _get_model("design")
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=design_instruct.strip(),
            )

        else:
            return None, f"Unknown mode: {mode}"

    except Exception as e:
        return None, f"Error: {e}"

    wav_path = _write_wav(wavs, sr)
    return wav_path, "Done."


# ── tab: Manage Voices ────────────────────────────────────────────────────────

def refresh_voice_table():
    rows = []
    for name in list_saved_voices():
        meta_path = _voice_meta(name)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("created", 0)))
            rows.append([name, meta.get("language", ""), created])
        else:
            rows.append([name, "", ""])
    return rows


def delete_voice_action(name: str):
    name = name.strip()
    if not name:
        return "Enter a voice name to delete.", refresh_voice_table()
    if delete_voice(name):
        return f"Deleted voice '{name}'.", refresh_voice_table()
    return f"Voice '{name}' not found.", refresh_voice_table()


# ── build UI ───────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3-TTS 1.7B") as demo:
        gr.Markdown("# Qwen3-TTS 1.7B")
        gr.Markdown(
            "Train a custom voice from an audio clip, then use it (or a preset speaker) "
            "to generate speech."
        )

        with gr.Tabs():

            # ── Train Voice ────────────────────────────────────────────────
            with gr.Tab("Train Voice"):
                gr.Markdown(
                    "Upload a clear audio recording (at least 3 seconds) and provide its "
                    "exact transcript. The voice will be saved and available in the Generate tab."
                )
                with gr.Row():
                    with gr.Column():
                        train_audio = gr.Audio(
                            label="Reference audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        train_ref_text = gr.Textbox(
                            label="Transcript (what is said in the audio)",
                            lines=3,
                            placeholder="Type the exact words spoken in the audio…",
                        )
                        train_name = gr.Textbox(
                            label="Voice name",
                            placeholder="e.g. my_voice, alice, narrator",
                        )
                        train_lang = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="English",
                        )
                        train_btn = gr.Button("Save Voice", variant="primary")
                    with gr.Column():
                        train_status = gr.Textbox(label="Status", lines=4, interactive=False)

                train_btn.click(
                    fn=train_voice,
                    inputs=[train_audio, train_ref_text, train_name, train_lang],
                    outputs=train_status,
                )

            # ── Generate Speech ────────────────────────────────────────────
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_mode = gr.Radio(
                            label="Voice source",
                            choices=["Preset Speaker", "Saved Voice", "Voice Design"],
                            value="Preset Speaker",
                        )
                        gen_text = gr.Textbox(
                            label="Text",
                            lines=5,
                            placeholder="Enter the text you want to synthesize…",
                        )
                        gen_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="English",
                        )

                        # Preset speaker controls
                        with gr.Group(visible=True) as preset_group:
                            gen_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=PRESET_SPEAKERS,
                                value="Ryan",
                            )
                            gen_instruct = gr.Textbox(
                                label="Style instruction (optional)",
                                placeholder="e.g. Speak with excitement.",
                            )

                        # Saved voice controls
                        with gr.Group(visible=False) as saved_group:
                            gen_saved = gr.Dropdown(
                                label="Saved voice",
                                choices=list_saved_voices(),
                                value=None,
                            )
                            refresh_saved_btn = gr.Button("Refresh list", size="sm")

                        # Voice design controls
                        with gr.Group(visible=False) as design_group:
                            gen_design_instruct = gr.Textbox(
                                label="Voice description",
                                lines=2,
                                placeholder="e.g. A deep, calm British male voice.",
                            )

                        gen_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column(scale=1):
                        gen_audio = gr.Audio(label="Output", type="filepath")
                        gen_status = gr.Textbox(label="Status", interactive=False)

                # Show/hide control groups based on mode
                def update_mode_visibility(mode):
                    return (
                        gr.update(visible=(mode == "Preset Speaker")),
                        gr.update(visible=(mode == "Saved Voice")),
                        gr.update(visible=(mode == "Voice Design")),
                    )

                gen_mode.change(
                    fn=update_mode_visibility,
                    inputs=gen_mode,
                    outputs=[preset_group, saved_group, design_group],
                )

                refresh_saved_btn.click(
                    fn=lambda: gr.update(choices=list_saved_voices()),
                    outputs=gen_saved,
                )

                gen_btn.click(
                    fn=generate_speech,
                    inputs=[
                        gen_mode,
                        gen_text,
                        gen_language,
                        gen_speaker,
                        gen_instruct,
                        gen_saved,
                        gen_design_instruct,
                    ],
                    outputs=[gen_audio, gen_status],
                )

            # ── Manage Voices ──────────────────────────────────────────────
            with gr.Tab("Manage Voices"):
                gr.Markdown("View and delete saved voice profiles.")
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

                delete_btn.click(
                    fn=delete_voice_action,
                    inputs=delete_name,
                    outputs=[manage_status, voices_table],
                )
                refresh_btn.click(fn=refresh_voice_table, outputs=voices_table)

    return demo


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS 1.7B GUI")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device, e.g. cuda:0, cpu, mps")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _device = args.device

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
