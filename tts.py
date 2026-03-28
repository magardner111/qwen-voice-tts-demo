"""TTS model cache, synthesis, and voice-prompt cache for chat."""

import re
import tempfile

import soundfile as sf
import torch

from config import CHAT_TTS_MAX_CHARS, TTS_MODEL_IDS
from voices import load_voice_profile

# ── module-level state ────────────────────────────────────────────────────────

_tts_cache: dict[str, object] = {}
_voice_prompt_cache: dict[str, object] = {}  # voice name → VoiceClonePromptItem
_device: str = "cuda:0"  # set by gui.py before any synthesis call


# ── model loader ──────────────────────────────────────────────────────────────

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_wav(wavs, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wavs[0], sr)
    return tmp.name


def _add_pauses(text: str) -> str:
    """Insert pause markers so TTS breathes naturally at punctuation."""
    text = re.sub(r'([.!?])\s+', r'\1... ', text)   # sentence pause
    text = re.sub(r',\s+', r',.. ', text)             # comma pause
    return text


# ── core synthesis dispatcher ─────────────────────────────────────────────────

def synthesize(
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


# ── chat-time synthesis (uses voice-prompt cache) ─────────────────────────────

def synthesize_voice(reply: str, saved_voice: str) -> str | None:
    """Synthesize a chat reply using a saved voice profile. Caches the voice prompt."""
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
