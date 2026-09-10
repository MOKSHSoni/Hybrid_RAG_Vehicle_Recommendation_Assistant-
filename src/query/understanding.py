"""Phase 3 orchestrator: context resolution -> standalone query ->
constraint extraction (LLM, validated, with regex fallback).
"""

from typing import List

from src.query.context import resolve_context
from src.query.llm_extraction import extract_constraints_via_llm
from src.query.models import ConversationTurn, QueryUnderstandingResult
from src.query.ollama_client import OllamaError
from src.query.regex_extraction import extract_regex_constraints


def understand_query(
    query: str,
    history: List[ConversationTurn],
    known_brands: List[str],
) -> QueryUnderstandingResult:
    standalone_query = resolve_context(query, history)

    try:
        constraints = extract_constraints_via_llm(standalone_query)
        return QueryUnderstandingResult(
            original_query=query,
            standalone_query=standalone_query,
            constraints=constraints,
            extraction_method="llm",
        )
    except (OllamaError, ValueError) as e:
        # OllamaError: the server call itself failed/timed out -- fall back
        # immediately, do not retry against a dead endpoint.
        # ValueError: invalid JSON/schema even after the LLM-side retry.
        constraints = extract_regex_constraints(standalone_query, known_brands)
        return QueryUnderstandingResult(
            original_query=query,
            standalone_query=standalone_query,
            constraints=constraints,
            extraction_method="regex_fallback",
            extraction_error=str(e),
        )
