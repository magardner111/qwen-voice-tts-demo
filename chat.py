"""Chat orchestration: wires LLM streaming + TTS synthesis together."""

from config import CHAT_MAX_EXCHANGES, CHAT_RESPONSE_MAX_CHARS
from conversations import save_conversation
from llm import build_messages, stream_ollama
from tts import synthesize, synthesize_voice
from voices import save_voice_profile


def chat_and_speak(
    user_msg: str,
    history: list[dict],
    chat_model: str,
    saved_voice: str,
    system_prompt: str,
):
    """Generator: shows a thinking indicator immediately, collects the full LLM
    reply silently, runs TTS, then yields text + audio together.
    Auto-saves and resets the conversation at CHAT_MAX_EXCHANGES exchanges."""
    user_msg = user_msg.strip()
    if not user_msg:
        return

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

    # Show thinking indicator immediately
    yield history + [{"role": "assistant", "content": "⏳ Thinking…"}], None

    # Collect full LLM reply before surfacing any text
    system = system_prompt.strip()
    length_hint = f"Keep your reply under {CHAT_RESPONSE_MAX_CHARS} characters."
    system = f"{system}\n{length_hint}".strip() if system else length_hint
    messages = build_messages(history, system)
    print(f"[llm] model={chat_model!r}, {len(messages)} messages")

    reply = ""
    try:
        for token, _ in stream_ollama(messages, chat_model):
            reply += token
    except Exception as e:
        print(f"[llm] error: {e}")
        reply = f"[LLM error: {e}]"

    # Run TTS before revealing text
    wav_path = synthesize_voice(reply, saved_voice if saved_voice != "(none)" else "")

    final_history = list(history) + [{"role": "assistant", "content": reply}]
    if chunk_notice:
        final_history = [{"role": "assistant", "content": chunk_notice}] + final_history

    yield final_history, wav_path


def generate_speech(
    voice_mode, text, language,
    preset_speaker, preset_instruct,
    clone_audio, clone_ref_text, clone_save_name,
    saved_voice, design_instruct,
):
    """One-shot TTS for the Generate tab. Optionally saves the clone as a profile."""
    save_name = clone_save_name.strip() if clone_save_name else ""
    if voice_mode == "Clone Voice" and save_name:
        if any(c in save_name for c in r'\/:*?"<>|'):
            return None, "Voice name contains invalid characters."
        if clone_audio:
            save_voice_profile(save_name, clone_audio, clone_ref_text.strip(), language)

    return synthesize(
        text, voice_mode, language,
        preset_speaker, preset_instruct,
        clone_audio, clone_ref_text,
        saved_voice, design_instruct,
    )
