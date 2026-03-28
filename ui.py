"""Gradio UI construction — all tabs and event wiring."""

import json
import time

import gradio as gr

from chat import chat_and_speak, generate_speech
from config import LANGUAGES, PERSONALITIES, PRESET_SPEAKERS, load_settings, save_settings
from conversations import list_conversations, load_conversation, save_conversation
from llm import fetch_ollama_models
from voices import _profile_meta, delete_voice_profile, list_saved_voices

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


# ── voice settings block (shared across tabs) ─────────────────────────────────

def _voice_settings_block(saved_voices_initial: list[str]) -> dict:
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


# ── saved voices table helpers ────────────────────────────────────────────────

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


# ── main UI builder ───────────────────────────────────────────────────────────

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
                    return load_conversation(name), None

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
