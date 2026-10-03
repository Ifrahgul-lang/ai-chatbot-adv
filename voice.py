"""Voice helpers. Pure functions (no Streamlit), so they are easy to test.
Input : microphone -> Groq Whisper (see llm.ChatClient.transcribe)
Output: the browser's built-in speech synthesis (free, no server cost)."""
import json
import re

# label shown in the UI -> (Whisper language code, browser voice language)
VOICE_LANGUAGES = {
    "English": ("en", "en-US"),
    "Urdu": ("ur", "ur-PK"),
}


def clean_for_speech(text: str, max_chars: int = 1500) -> str:
    """Remove Markdown so the voice does not read symbols aloud."""
    text = re.sub(r"```.*?```", " (code example) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*(?:[-+*]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"[#*>|~]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def speak_html(text: str, lang: str = "en-US") -> str:
    """Small HTML widget with Listen / Stop buttons (browser speechSynthesis).
    Speech starts on a click because browsers block autoplay audio."""
    # json.dumps escapes quotes/newlines/non-ASCII; we also escape "<" so the text can never
    # close or open an HTML tag (the iframe has same-origin access, so this matters)
    payload = json.dumps(text).replace("<", "\\u003c")
    return f"""
<div style="font-family:system-ui,sans-serif">
  <button id="play" style="padding:6px 12px;cursor:pointer">🔊 Listen</button>
  <button id="stop" style="padding:6px 12px;cursor:pointer">⏹ Stop</button>
</div>
<script>
  const text = {payload};
  document.getElementById("play").onclick = () => {{
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = {json.dumps(lang)};
    speechSynthesis.speak(u);
  }};
  document.getElementById("stop").onclick = () => speechSynthesis.cancel();
</script>"""
