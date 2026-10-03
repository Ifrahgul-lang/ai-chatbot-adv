"""Streamlit UI. AI logic: llm.py | settings: config.py | prompt: prompts.py | voice helpers: voice.py"""
import hashlib
import logging

import streamlit as st

from config import Settings, load_settings
from llm import ChatClient, ChatError
from prompts import APP_NAME, SYSTEM_PROMPT
from voice import VOICE_LANGUAGES, clean_for_speech, speak_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title=APP_NAME, page_icon="💬")
settings = load_settings()


def render_html(html: str, height: int) -> None:
    """st.iframe on new Streamlit; components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components

        components.html(html, height=height)


@st.cache_resource(ttl=600)  # one client shared by all reruns; refreshed every 10 minutes
def get_chat_client(s: Settings) -> ChatClient:
    return ChatClient(
        s.api_key, SYSTEM_PROMPT, model_override=s.model_override, max_history=s.max_history,
        max_history_chars=s.max_history_chars, max_tokens=s.max_tokens,
        timeout=s.timeout, max_retries=s.max_retries,
    )


st.title(f"💬 {APP_NAME}")
st.caption("Your friendly AI tutor for programming and AI.")

if not settings.api_key:
    st.error(
        "GROQ_API_KEY is missing. Locally: add it to the .env file. "
        "On Streamlit Cloud: add it under App settings → Secrets."
    )
    st.stop()

chat_client = get_chat_client(settings)

for key, default in (("messages", []), ("tokens", 0), ("last_audio", None), ("mic_n", 0)):
    st.session_state.setdefault(key, default)

# ---------------- sidebar ----------------
with st.sidebar:
    st.subheader("Chat")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages, st.session_state.tokens = [], 0
        st.rerun()

    voice_on = st.toggle("🎙️ Voice mode", help="Speak your question and listen to the answer.")
    voice_lang = "English"
    if voice_on:
        voice_lang = st.selectbox("Voice language", list(VOICE_LANGUAGES))

    if chat_client.active_model:
        st.caption(f"Model in use: {chat_client.active_model}")
    st.caption(f"Tokens used in this chat: {st.session_state.tokens}")

stt_lang, tts_lang = VOICE_LANGUAGES[voice_lang]

# ---------------- voice input: speech -> text ----------------
MAX_AUDIO_MB = 20  # Groq's free tier accepts files up to ~25 MB
audio = None
if voice_on:
    n = st.session_state.mic_n  # the widget keys change after each question, which resets the recorder
    mic = st.audio_input("🎤 Speak your question", key=f"mic_{n}")
    with st.expander("Or upload an audio file instead"):
        upload = st.file_uploader(
            "Audio file", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
            key=f"upload_{n}", label_visibility="collapsed",
        )
    audio = mic if mic is not None else upload

voice_text = None
if audio is not None:
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        st.warning(f"Audio file is too large. Please keep it under {MAX_AUDIO_MB} MB.")
    elif digest != st.session_state.last_audio:  # transcribe each recording only once
        st.session_state.last_audio = digest
        try:
            with st.spinner("Listening…"):
                voice_text = chat_client.transcribe(data, stt_lang, filename=getattr(audio, "name", None) or "voice.wav")
        except ChatError as e:
            st.error(e.user_message)

# ---------------- conversation ----------------
messages = st.session_state.messages
for i, m in enumerate(messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if voice_on and m["role"] == "assistant" and i == len(messages) - 1:
            render_html(speak_html(clean_for_speech(m["content"]), tts_lang), height=48)

if not messages:
    st.info("Ask me anything about Python, APIs, or AI. For example: *What is an API?*")

prompt = (st.chat_input("Ask anything…") or voice_text or "").strip()

# input validation: protects against very long (expensive) messages
if len(prompt) > settings.max_input_chars:
    st.warning(f"Your message has {len(prompt)} characters. Please keep it under {settings.max_input_chars}.")
    prompt = ""

if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):  # loading state
                result = chat_client.reply(messages)
            messages.append({"role": "assistant", "content": result.text})
            st.session_state.tokens += result.tokens
            st.session_state.mic_n += 1  # fresh recorder for the next question
            st.rerun()  # redraw: shows the answer, the voice button and updated sidebar
        except ChatError as e:  # known failures: friendly message, user can retry
            st.error(e.user_message)
            messages.pop()
        except Exception:  # unknown failures: log details, show a generic message
            logging.getLogger("chatbot.app").exception("Unexpected error")
            st.error("Something went wrong on our side. Please try again.")
            messages.pop()
