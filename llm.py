"""LLM access layer: Groq client, model fallback, token control, voice transcription,
error mapping, logging. No Streamlit imports here, so this module is unit-testable."""
import logging
import time
from typing import NamedTuple

from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError

logger = logging.getLogger("chatbot.llm")

# Tried in this order; the app uses the first one the API key can access.
PREFERRED_MODELS = [
    "llama-3.3-70b-versatile",  # retired by Groq in Aug 2026; kept first, fallbacks follow
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]
TRANSCRIBE_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"]
NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "embed")  # audio / safety models


class ChatError(Exception):
    """An error whose message is safe to show to the user."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class ModelUnavailable(Exception):
    """Groq says this model does not exist / was retired."""


class Reply(NamedTuple):
    text: str
    model: str
    tokens: int  # total tokens used by this request (0 if unknown)


def trim_history(history: list, max_messages: int, max_chars: int) -> list:
    """Keep the newest messages within BOTH a message limit and a character budget."""
    kept, used = [], 0
    for m in reversed(history[-max_messages:]):
        used += len(m["content"])
        if kept and used > max_chars:
            break
        kept.append(m)
    return list(reversed(kept))


class ChatClient:
    def __init__(self, api_key, system_prompt, model_override="", max_history=10,
                 max_history_chars=4000, max_tokens=800, timeout=30.0, max_retries=2, client=None):
        # The SDK retries 429 / 5xx / connection errors with backoff (max_retries)
        self._client = client or Groq(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.system_prompt = system_prompt
        self.model_override = model_override
        self.max_history = max_history
        self.max_history_chars = max_history_chars
        self.max_tokens = max_tokens
        self.active_model = None
        self._models = None

    # ---------- helpers ----------
    def _available_models(self) -> list:
        if self._models is None:
            try:
                data = self._client.models.list().data
                self._models = sorted(m.id for m in data if not any(x in m.id.lower() for x in NOT_CHAT))
            except Exception as e:  # listing is optional; fall back to the preference list
                logger.warning("Could not list models: %s", e)
                self._models = []
        return self._models

    def _candidates(self) -> list:
        available = self._available_models()
        ordered = ([self.model_override] if self.model_override else []) + PREFERRED_MODELS
        if available:
            ordered = [m for m in ordered if m in available] + [m for m in available if m not in ordered]
        if self.active_model:  # the model that worked last time goes first
            ordered = [self.active_model] + [m for m in ordered if m != self.active_model]
        return ordered[:6]

    @staticmethod
    def _call(fn, **kwargs):
        """Run one API call and translate SDK errors into friendly ones."""
        try:
            return fn(**kwargs)
        except AuthenticationError as e:
            logger.error("Authentication failed")
            raise ChatError("Invalid Groq API key. Check your .env file or Streamlit secrets.") from e
        except RateLimitError as e:
            logger.warning("Rate limited")
            raise ChatError("Groq is getting too many requests. Please wait a moment and try again.") from e
        except APIConnectionError as e:  # includes timeouts
            logger.warning("Connection problem: %s", e)
            raise ChatError("Could not reach Groq (network problem or timeout). Please try again.") from e
        except APIStatusError as e:
            if e.status_code in (400, 404) and "model" in str(e.message).lower():
                raise ModelUnavailable(str(e.message)) from e
            logger.error("Groq API error %s: %s", e.status_code, e.message)
            raise ChatError(f"Groq returned an error ({e.status_code}). Please try again.") from e

    # ---------- chat ----------
    def reply(self, history: list) -> Reply:
        """history: [{"role": "user"|"assistant", "content": str}, ...]"""
        trimmed = trim_history(history, self.max_history, self.max_history_chars)
        messages = [{"role": "system", "content": self.system_prompt}] + trimmed
        for model in self._candidates():
            kwargs = {"model": model, "messages": messages, "max_tokens": self.max_tokens}
            if model.startswith("openai/gpt-oss"):
                kwargs["extra_body"] = {"reasoning_effort": "low"}  # fewer hidden "thinking" tokens
            start = time.perf_counter()
            try:
                res = self._call(self._client.chat.completions.create, **kwargs)
            except ModelUnavailable as e:
                logger.warning("Model %s unavailable, trying next: %s", model, e)
                continue

            choice = res.choices[0]
            text = (choice.message.content or "").strip()
            tokens = getattr(getattr(res, "usage", None), "total_tokens", 0) or 0
            logger.info("model=%s tokens=%s latency_ms=%d", model, tokens, (time.perf_counter() - start) * 1000)
            if not text:
                raise ChatError("The model returned an empty answer. Please try again.")
            if getattr(choice, "finish_reason", None) == "length":
                text += "\n\n*(Answer shortened to save tokens. Ask me to continue for more.)*"
            self.active_model = model
            return Reply(text, model, tokens)

        raise ChatError(
            "No usable AI model is available for this API key. "
            "Run `python check_models.py` to see which models your key can access."
        )

    # ---------- voice input ----------
    def transcribe(self, audio_bytes: bytes, language: str = "en", filename: str = "voice.wav") -> str:
        """Speech -> text with Groq Whisper."""
        for model in TRANSCRIBE_MODELS:
            try:
                res = self._call(
                    self._client.audio.transcriptions.create,
                    file=(filename, audio_bytes), model=model, language=language, temperature=0.0,
                )
            except ModelUnavailable:
                continue
            text = (getattr(res, "text", "") or "").strip()
            if not text:
                raise ChatError("I could not hear anything clearly. Please try recording again.")
            return text
        raise ChatError("Speech recognition is not available for this API key.")
