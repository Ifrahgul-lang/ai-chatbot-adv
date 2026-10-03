"""Unit tests for llm.py using a fake Groq client (no network, no API key)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import groq
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm import ChatClient, ChatError, trim_history  # noqa: E402

REQ = httpx.Request("POST", "http://test")


def status_error(cls, code, msg="error"):
    return cls(msg, response=httpx.Response(code, request=REQ), body=None)


class FakeGroq:
    """Mimics the parts of the Groq client we use."""

    def __init__(self, available=("openai/gpt-oss-120b",), behaviour=None, finish="stop", tokens=42):
        self.available, self.behaviour, self.finish, self.tokens = available, behaviour or {}, finish, tokens
        self.calls, self.audio_calls = [], []
        outer = self

        def list_models():
            return NS(data=[NS(id=i) for i in outer.available])

        def create(**kw):
            outer.calls.append(kw)
            action = outer.behaviour.get(kw["model"], "ok")
            if isinstance(action, Exception):
                raise action
            text = "hello" if action == "ok" else action
            return NS(choices=[NS(message=NS(content=text), finish_reason=outer.finish)],
                      usage=NS(total_tokens=outer.tokens))

        def transcribe(**kw):
            outer.audio_calls.append(kw)
            action = outer.behaviour.get(kw["model"], "ok")
            if isinstance(action, Exception):
                raise action
            return NS(text="what is an api" if action == "ok" else action)

        self.models = NS(list=list_models)
        self.chat = NS(completions=NS(create=create))
        self.audio = NS(transcriptions=NS(create=transcribe))


def make(fake, **kw):
    return ChatClient("k", "SYSTEM", client=fake, **kw)


H = [{"role": "user", "content": "hi"}]


# ---------- chat ----------
def test_success_returns_text_model_tokens_and_system_prompt_first():
    fake = FakeGroq()
    r = make(fake).reply(H)
    assert (r.text, r.model, r.tokens) == ("hello", "openai/gpt-oss-120b", 42)
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": "SYSTEM"}


def test_max_tokens_is_sent():
    fake = FakeGroq()
    make(fake, max_tokens=123).reply(H)
    assert fake.calls[0]["max_tokens"] == 123


def test_reasoning_effort_low_only_for_gpt_oss():
    fake = FakeGroq(available=("openai/gpt-oss-120b", "llama-3.1-8b-instant"))
    make(fake).reply(H)
    assert fake.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    fake2 = FakeGroq(available=("llama-3.1-8b-instant",))
    make(fake2).reply(H)
    assert "extra_body" not in fake2.calls[0]


def test_truncated_answer_gets_a_note():
    r = make(FakeGroq(finish="length")).reply(H)
    assert "shortened" in r.text


def test_trim_history_by_messages_and_chars():
    hist = [{"role": "user", "content": "x" * 100} for _ in range(10)]
    assert len(trim_history(hist, max_messages=3, max_chars=10_000)) == 3
    assert len(trim_history(hist, max_messages=10, max_chars=250)) == 2
    assert len(trim_history(hist, max_messages=10, max_chars=1)) == 1  # latest message is always kept


def test_history_is_trimmed_before_sending():
    fake = FakeGroq()
    make(fake, max_history=3).reply([{"role": "user", "content": str(i)} for i in range(10)])
    assert len(fake.calls[0]["messages"]) == 4  # system + last 3


def test_falls_back_when_model_not_found():
    err = status_error(groq.NotFoundError, 404, "The model `x` does not exist")
    fake = FakeGroq(available=("a-model", "b-model"), behaviour={"a-model": err})
    c = make(fake)
    assert c.reply(H).text == "hello"
    assert [k["model"] for k in fake.calls] == ["a-model", "b-model"]
    assert c.active_model == "b-model"


def test_no_usable_model():
    err = status_error(groq.NotFoundError, 404, "The model `x` does not exist")
    with pytest.raises(ChatError, match="No usable"):
        make(FakeGroq(available=("a",), behaviour={"a": err})).reply(H)


@pytest.mark.parametrize("exc,text", [
    (status_error(groq.AuthenticationError, 401), "Invalid Groq API key"),
    (status_error(groq.RateLimitError, 429), "too many requests"),
    (groq.APIConnectionError(request=REQ), "Could not reach"),
    (status_error(groq.InternalServerError, 500), r"error \(500\)"),
])
def test_errors_become_friendly_messages(exc, text):
    with pytest.raises(ChatError, match=text):
        make(FakeGroq(behaviour={"openai/gpt-oss-120b": exc})).reply(H)


def test_empty_answer():
    with pytest.raises(ChatError, match="empty"):
        make(FakeGroq(behaviour={"openai/gpt-oss-120b": "   "})).reply(H)


# ---------- voice input ----------
def test_transcribe_success():
    fake = FakeGroq()
    assert make(fake).transcribe(b"audio", "ur") == "what is an api"
    assert fake.audio_calls[0]["language"] == "ur"


def test_transcribe_falls_back_to_second_model():
    err = status_error(groq.NotFoundError, 404, "The model `x` does not exist")
    fake = FakeGroq(behaviour={"whisper-large-v3-turbo": err})
    assert make(fake).transcribe(b"audio") == "what is an api"
    assert [k["model"] for k in fake.audio_calls] == ["whisper-large-v3-turbo", "whisper-large-v3"]


def test_transcribe_silence():
    fake = FakeGroq(behaviour={"whisper-large-v3-turbo": "  "})
    with pytest.raises(ChatError, match="could not hear"):
        make(fake).transcribe(b"audio")
