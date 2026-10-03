"""Unit tests for voice.py helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice import clean_for_speech, speak_html  # noqa: E402


def test_markdown_is_removed():
    md = "## Title\n**Bold** and `code`.\n- item one\n1. step\n[link](http://x.com)\n```py\nprint(1)\n```"
    out = clean_for_speech(md)
    assert "#" not in out and "*" not in out and "`" not in out and "http" not in out
    assert "(code example)" in out and "print" not in out
    assert "Bold and code." in out and "item one" in out and "link" in out


def test_length_is_limited():
    assert len(clean_for_speech("word " * 1000, max_chars=100)) <= 100


def test_html_is_safe_for_script_tag():
    html = speak_html('hello </script><script>alert(1)</script> "quote"', "ur-PK")
    assert "</script><script>alert" not in html
    assert "<script>alert" not in html and "u003c" in html
    assert "ur-PK" in html and "speechSynthesis" in html
