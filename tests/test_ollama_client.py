import json

import pytest
from unittest.mock import MagicMock, patch

import config
from src.query.ollama_client import (
    OllamaError,
    _partial_json_string,
    chat,
    chat_long_form,
    chat_text,
    chat_text_stream,
    is_reachable,
    looks_like_rambling,
)


def test_is_reachable_true_when_list_succeeds():
    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.list.return_value = {"models": []}
        assert is_reachable() is True


def test_is_reachable_false_on_any_exception():
    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.list.side_effect = ConnectionError("refused")
        assert is_reachable() is False


def test_is_reachable_does_not_invoke_chat(monkeypatch):
    # A reachability check should never trigger real generation -- confirm
    # it only ever calls .list(), never .chat().
    mock_client = MagicMock()
    with patch("src.query.ollama_client.ollama.Client", return_value=mock_client):
        is_reachable()
    mock_client.chat.assert_not_called()
    mock_client.list.assert_called_once()


def test_looks_like_rambling_detects_marker_phrases():
    assert looks_like_rambling("Okay, the user wants a hypothetical vehicle...") is True
    assert looks_like_rambling("Let me think about what to include here") is True


def test_looks_like_rambling_detects_excessive_length():
    assert looks_like_rambling("x" * 501) is True


def test_looks_like_rambling_accepts_clean_short_text():
    assert looks_like_rambling("The Everest GX is a rugged 7-seat SUV priced from Rs 12 Lakh.") is False


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


def test_chat_long_form_strips_stray_think_tags():
    raw = "<think>internal reasoning that should never appear</think>The actual clean answer."
    with patch("src.query.ollama_client.chat", return_value=raw):
        result = chat_long_form(messages=[{"role": "user", "content": "explain this"}])
    assert result == "The actual clean answer."
    assert "internal reasoning" not in result


def test_chat_long_form_omits_think_param_from_underlying_call():
    # think=None must mean the `think` kwarg is left out of the client.chat()
    # call entirely, not sent as False -- see ollama_client.py's module
    # docstring for why that distinction is what actually keeps content clean.
    with patch("src.query.ollama_client.chat") as mock_chat:
        mock_chat.return_value = "clean answer"
        chat_long_form(messages=[{"role": "user", "content": "explain this"}])
    assert mock_chat.call_args.kwargs["think"] is None


def test_chat_long_form_passes_no_format_schema():
    with patch("src.query.ollama_client.chat", return_value="clean answer") as mock_chat:
        chat_long_form(messages=[{"role": "user", "content": "explain this"}])
    assert mock_chat.call_args.kwargs["format"] is None


# ---------------------------------------------------------------------------
# Streaming: incremental decode of the {"result": "..."} schema wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "buffer,expected",
    [
        ('{"resu', None),  # key not yet complete
        ('{"result"', None),  # opening quote of the value not yet seen
        ('{"result": "', ""),  # value started, nothing in it yet
        ('{"result": "Hello', "Hello"),  # mid-stream partial
        ('{"result": "Hello world"}', "Hello world"),  # complete
        ('{"result": "line1\\nline2"}', "line1\nline2"),  # escaped newline decoded
        ('{"result": "say \\"hi\\""}', 'say "hi"'),  # escaped quotes don't end the string
        ('{"result": "trailing esc\\', "trailing esc"),  # half-streamed escape stops cleanly
    ],
)
def test_partial_json_string(buffer, expected):
    assert _partial_json_string(buffer, "result") == expected


def test_chat_text_stream_yields_only_prose_deltas():
    # The wire format is JSON, but callers must only ever see the prose.
    chunks = ['{"resu', 'lt": "Aff', "ordable SUV", 's like the Nexon."}']
    fake_stream = [{"message": {"content": c}} for c in chunks]

    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = fake_stream
        out = list(chat_text_stream(messages=[{"role": "user", "content": "suvs?"}]))

    assert "".join(out) == "Affordable SUVs like the Nexon."
    assert not any("{" in piece or "result" in piece for piece in out)


def test_chat_text_stream_requests_streaming_and_schema():
    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = []
        list(chat_text_stream(messages=[{"role": "user", "content": "hi"}]))

    kwargs = mock_client_cls.return_value.chat.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["format"]["properties"]["result"]["type"] == "string"


def test_chat_text_stream_wraps_errors_as_ollama_error():
    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.side_effect = ConnectionError("refused")
        with pytest.raises(OllamaError):
            list(chat_text_stream(messages=[{"role": "user", "content": "hi"}]))


def test_chat_passes_keep_alive():
    with patch("src.query.ollama_client.ollama.Client") as mock_client_cls:
        mock_client_cls.return_value.chat.return_value = {"message": {"content": "x"}}
        chat(messages=[{"role": "user", "content": "hi"}])
    assert mock_client_cls.return_value.chat.call_args.kwargs["keep_alive"] == config.OLLAMA_KEEP_ALIVE
