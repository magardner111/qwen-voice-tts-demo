"""
Qwen3-TTS 1.7B — Text-to-Speech demo
Supports three modes:
  custom    — preset speakers with optional style/emotion instructions
  clone     — voice cloning from a 3-second reference audio file
  design    — describe the voice you want in natural language

Install:
  pip install -U qwen-tts
  pip install -U flash-attn --no-build-isolation  # optional but recommended
"""

import argparse
import sys

import soundfile as sf
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(model_id: str, device: str) -> "Qwen3TTSModel":
    from qwen_tts import Qwen3TTSModel

    kwargs = {
        "device_map": device,
        "dtype": torch.bfloat16,
    }
    try:
        import flash_attn  # noqa: F401
        kwargs["attn_implementation"] = "flash_attention_2"
        print(f"[info] flash-attn detected — using flash_attention_2")
    except ImportError:
        pass

    print(f"[info] loading {model_id} …")
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)


def _save(wavs, sr: int, output: str) -> None:
    sf.write(output, wavs[0], sr)
    print(f"[info] saved → {output}")


# ── mode handlers ─────────────────────────────────────────────────────────────

def run_custom(args) -> None:
    model = _load_model("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", args.device)

    if args.list_speakers:
        print("Supported speakers:", model.get_supported_speakers())
        print("Supported languages:", model.get_supported_languages())
        return

    wavs, sr = model.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct or "",
    )
    _save(wavs, sr, args.output)


def run_clone(args) -> None:
    model = _load_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base", args.device)

    if not args.ref_audio:
        sys.exit("[error] --ref-audio is required for clone mode")
    if not args.ref_text:
        sys.exit("[error] --ref-text is required for clone mode")

    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
    _save(wavs, sr, args.output)


def run_design(args) -> None:
    model = _load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", args.device)

    if not args.instruct:
        sys.exit("[error] --instruct is required for design mode (describe the voice)")

    wavs, sr = model.generate_voice_design(
        text=args.text,
        language=args.language,
        instruct=args.instruct,
    )
    _save(wavs, sr, args.output)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qwen3-TTS 1.7B text-to-speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # List available preset speakers
  python qwen3_tts.py custom --list-speakers

  # Preset speaker (English)
  python qwen3_tts.py custom \\
      --text "Hello, welcome to the future of speech synthesis." \\
      --speaker Ryan --language English --output hello.wav

  # Preset speaker with emotion instruction
  python qwen3_tts.py custom \\
      --text "I can't believe you did that!" \\
      --speaker Aiden --language English \\
      --instruct "Speak with shocked disbelief." --output shocked.wav

  # Voice cloning from a reference file
  python qwen3_tts.py clone \\
      --text "The quick brown fox jumps over the lazy dog." \\
      --ref-audio reference.wav \\
      --ref-text "This is what I said in the reference recording." \\
      --language English --output cloned.wav

  # Voice design via natural language description
  python qwen3_tts.py design \\
      --text "Good evening, ladies and gentlemen." \\
      --instruct "A warm, deep British male voice with a calm authority." \\
      --language English --output designed.wav
""",
    )

    # Shared options
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--text", default="Hello, this is Qwen3-TTS speaking.")
    shared.add_argument(
        "--language",
        default="English",
        choices=["Auto", "Chinese", "English", "Japanese", "Korean",
                 "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
    )
    shared.add_argument("--output", default="output.wav", help="Output WAV file path")
    shared.add_argument("--device", default="cuda:0",
                        help="Torch device (cuda:0, cpu, mps, …)")

    sub = p.add_subparsers(dest="mode", required=True)

    # custom
    c = sub.add_parser("custom", parents=[shared], help="Preset speakers")
    c.add_argument("--speaker", default="Ryan",
                   help="Speaker name (run with --list-speakers to see options)")
    c.add_argument("--instruct", default="",
                   help="Optional style/emotion instruction, e.g. 'Speak very cheerfully.'")
    c.add_argument("--list-speakers", action="store_true",
                   help="Print supported speakers/languages and exit")

    # clone
    cl = sub.add_parser("clone", parents=[shared], help="Voice cloning")
    cl.add_argument("--ref-audio", dest="ref_audio",
                    help="Path, URL, or base64 string for reference audio (≥3 s)")
    cl.add_argument("--ref-text", dest="ref_text",
                    help="Transcript of the reference audio")

    # design
    d = sub.add_parser("design", parents=[shared], help="Natural language voice design")
    d.add_argument("--instruct",
                   help="Natural language description of the desired voice")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {"custom": run_custom, "clone": run_clone, "design": run_design}
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
