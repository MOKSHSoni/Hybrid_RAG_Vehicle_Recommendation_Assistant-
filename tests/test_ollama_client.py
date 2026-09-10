import json
from unittest.mock import patch

from src.query.ollama_client import OllamaError, _looks_like_rambling, chat_text


def test_looks_like_rambling_detects_marker_phrases():
    assert _looks_like_rambling("Okay, the user wants a hypothetical vehicle...") is True
    assert _looks_like_rambling("Let me think about what to include here") is True


def test_looks_like_rambling_detects_excessive_length():
    assert _looks_like_rambling("x" * 501) is True


def test_looks_like_rambling_accepts_clean_short_text():
    assert _looks_like_rambling("The Everest GX is a rugged 7-seat SUV priced from Rs 12 Lakh.") is False


def test_chat_text_returns_clean_result_without_retry():
    payload = json.dumps({"result": "affordable SUV"})
    with patch("src.query.ollama_client.chat", return_value=payload) as mock_chat:
        result = chat_text(messages=[{"role": "user", "content": "rewrite this"}])
    assert result == "affordable SUV"
    assert mock_chat.call_count == 1  # no retry needed


def test_chat_text_retries_once_when_rambling_detected():
    rambling_payload = json.dumps({"result": "Okay, the user wants an affordable SUV, let me think about this..."})
    clean_payload = json.dumps({"result": "affordable SUV"})
    with patch("src.query.ollama_client.chat", side_effect=[rambling_payload, clean_payload]) as mock_chat:
        result = chat_text(messages=[{"role": "user", "content": "rewrite this"}])
    assert result == "affordable SUV"
    assert mock_chat.call_count == 2


def test_chat_text_returns_last_attempt_if_still_rambling_after_retry():
    rambling_payload = json.dumps({"result": "Okay, the user wants an affordable SUV, let me think about this..."})
    with patch("src.query.ollama_client.chat", return_value=rambling_payload) as mock_chat:
        result = chat_text(messages=[{"role": "user", "content": "rewrite this"}])
    assert "let me think" in result.lower()  # still rambling, but returned rather than crashing/blocking forever
    assert mock_chat.call_count == 2


def test_chat_text_falls_back_to_raw_text_when_model_ignores_schema():
    with patch("src.query.ollama_client.chat", return_value="not json at all"):
        result = chat_text(messages=[{"role": "user", "content": "rewrite this"}])
    assert result == "not json at all"


def test_chat_text_propagates_ollama_error():
    import pytest

    with patch("src.query.ollama_client.chat", side_effect=OllamaError("simulated")):
        with pytest.raises(OllamaError):
            chat_text(messages=[{"role": "user", "content": "rewrite this"}])
