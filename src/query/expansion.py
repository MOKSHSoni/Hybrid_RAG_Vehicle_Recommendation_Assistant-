"""Query expansion: generate 3 genuinely different angles (Q1/Q2/Q3) on
the same underlying search intent -- not superficial paraphrases of each
other or of the original query (Q0). Combined with Q0, later phases feed
all 4 into retrieval and merge/dedup the results (Phase 8).
"""

import json
from typing import List

import config
from src.query.ollama_client import OllamaError, chat

_EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

_SYSTEM_PROMPT = """Given a vehicle search query, generate exactly 3 alternative search
queries that approach the SAME underlying intent from three genuinely different
angles -- not paraphrases of each other or of the original query. For example, for
"affordable family SUV with good mileage" you might produce one query focused on
technical specs (fuel efficiency, engine size), one focused on use-case (family
trips, spacious, practical), and one focused on value/price positioning
(budget-friendly, low running costs).
Respond with ONLY a JSON object: {"queries": ["...", "...", "..."]} containing
exactly 3 strings, no explanation."""


def expand_query(query: str) -> List[str]:
    """Returns up to 3 expanded queries. Returns an empty list if the LLM
    call fails or never produces a valid 3-item array after retries --
    callers should treat "no expansion" as a graceful degradation to
    just the original query, not an error."""
    for _ in range(config.EXTRACTION_RETRY_COUNT + 1):
        try:
            raw = chat(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                format=_EXPANSION_SCHEMA,
            )
        except OllamaError:
            return []  # dead endpoint -- do not retry, degrade immediately

        try:
            data = json.loads(raw)
            queries = data["queries"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        if not isinstance(queries, list) or not all(isinstance(q, str) and q.strip() for q in queries):
            continue

        return queries[:3] if len(queries) >= 3 else queries

    return []
