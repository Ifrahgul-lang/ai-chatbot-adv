"""Central configuration. Secrets come from Streamlit secrets (deployed) or .env (local)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str = "") -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_override: str = ""        # optional: force a model
    # --- cost / token control ---
    max_history: int = 10           # max messages sent to the model
    max_history_chars: int = 4000   # ...and max characters (~1000 tokens) of history
    max_tokens: int = 800           # hard cap on the length of one answer
    max_input_chars: int = 1000     # longest message a user may send
    # --- reliability ---
    timeout: float = 30.0           # seconds before a request is abandoned
    max_retries: int = 2            # automatic retries on 429 / 5xx / network errors


def load_settings() -> Settings:
    return Settings(
        api_key=get_setting("GROQ_API_KEY").strip().strip('"').strip("'"),
        model_override=get_setting("GROQ_MODEL").strip(),
    )
