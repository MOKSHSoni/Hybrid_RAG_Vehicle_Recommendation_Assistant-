"""Phase 3 orchestrator: context resolution -> standalone query ->
constraint extraction (LLM, validated, with regex fallback).
"""

from dataclasses import replace
from typing import List

from src.query.context import resolve_context
from src.query.llm_extraction import extract_constraints_via_llm
from src.query.models import Constraints, ConversationTurn, QueryUnderstandingResult
from src.query.ollama_client import OllamaError
from src.query.regex_extraction import extract_regex_constraints
from src.query.scope import match_known_brand


def understand_query(
    query: str,
    history: List[ConversationTurn],
    known_brands: List[str],
) -> QueryUnderstandingResult:
    standalone_query = resolve_context(query, history)

    try:
        constraints = _canonicalise_brand(extract_constraints_via_llm(standalone_query), known_brands)
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


def _canonicalise_brand(constraints: Constraints, known_brands: List[str]) -> Constraints:
    """Rewrite an extracted brand to the catalogue's own spelling.

    metadata_filter compares brand by exact string and documents that
    "extraction already canonicalizes to a known brand". The regex path
    does; the LLM path never did. It returns "Rolls Royce" and
    "Mercedes-Benz" where the dataset stores "Rolls" and "Mercedes", so the
    filter matched nothing and relaxation dropped the brand -- a request for
    Rolls-Royce came back as three Mercedes.

    A brand with no catalogue match is left untouched rather than cleared:
    the scope check needs the name to tell the user that brand isn't
    carried, instead of silently showing them some other make.
    """
    canonical = match_known_brand(constraints.brand, known_brands)
    if canonical and canonical != constraints.brand:
        return replace(constraints, brand=canonical)
    return constraints
