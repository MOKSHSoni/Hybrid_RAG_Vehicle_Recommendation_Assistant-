import json
from unittest.mock import patch

from conftest import requires_ollama

from src.query.expansion import expand_query
from src.query.ollama_client import OllamaError
from src.query.transformation import transform_query


# ---------------------------------------------------------------------------
# transform_query
# ---------------------------------------------------------------------------


def test_transform_query_falls_back_on_ollama_error():
    with patch("src.query.transformation.chat_text", side_effect=OllamaError("simulated")):
        result = transform_query("I really want an affordable SUV please")
    assert result == "I really want an affordable SUV please"


def test_transform_query_strips_quotes_and_whitespace():
    with patch("src.query.transformation.chat_text", return_value='  "affordable SUV"  '):
        result = transform_query("I really want an affordable SUV please")
    assert result == "affordable SUV"


def test_transform_query_falls_back_on_empty_response():
    with patch("src.query.transformation.chat_text", return_value="   "):
        result = transform_query("affordable SUV")
    assert result == "affordable SUV"


@requires_ollama
def test_transform_query_real_call_removes_filler():
    result = transform_query("I would really really love it if you could show me an affordable SUV please")
    assert "suv" in result.lower()
    assert len(result) <= 80  # should be shorter/more direct than the verbose input


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------


def test_expand_query_returns_three_queries_on_valid_response():
    payload = json.dumps({"queries": ["spec-focused query", "use-case query", "budget query"]})
    with patch("src.query.expansion.chat", return_value=payload):
        result = expand_query("affordable family SUV with good mileage")
    assert result == ["spec-focused query", "use-case query", "budget query"]


def test_expand_query_retries_once_on_invalid_json_then_succeeds():
    payload = json.dumps({"queries": ["a", "b", "c"]})
    with patch("src.query.expansion.chat", side_effect=["not json", payload]) as mock_chat:
        result = expand_query("affordable SUV")
    assert result == ["a", "b", "c"]
    assert mock_chat.call_count == 2


def test_expand_query_returns_empty_list_after_exhausting_retries():
    with patch("src.query.expansion.chat", return_value="not json at all"):
        result = expand_query("affordable SUV")
    assert result == []


def test_expand_query_returns_empty_list_immediately_on_ollama_error_no_retry():
    with patch("src.query.expansion.chat", side_effect=OllamaError("simulated")) as mock_chat:
        result = expand_query("affordable SUV")
    assert result == []
    assert mock_chat.call_count == 1  # dead endpoint: no retry


@requires_ollama
def test_expand_query_real_call_produces_distinct_angles():
    result = expand_query("affordable family SUV with good mileage")
    assert len(result) == 3
    # Genuinely different angles, not near-duplicates of each other.
    assert len({q.lower() for q in result}) == 3
