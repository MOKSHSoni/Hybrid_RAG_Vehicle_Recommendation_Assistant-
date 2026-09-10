"""Query transformation: rewrite the standalone query into a form
optimized for retrieval (strip conversational filler, phrase as a direct
descriptive query). Distinct from context.py's job (resolving
conversational references) -- this runs after that, purely to improve
retrieval quality of the query text itself.
"""

from src.query.ollama_client import OllamaError, chat_text

_SYSTEM_PROMPT = """You optimize vehicle search queries for a retrieval system.
Rewrite the query into a concise, retrieval-optimized form: keep all substantive
search intent (vehicle type, features, price, use-case), remove conversational
filler ("I really want", "please", "could you show me", "I'm looking for"), and
phrase it as a direct descriptive query rather than a question or request.
If the query is already concise and direct, return it unchanged.
Respond with ONLY the rewritten query text, no quotes, no explanation."""


def transform_query(standalone_query: str) -> str:
    """Returns an optimized query. Falls back to the input unchanged if
    the LLM call fails."""
    try:
        rewritten = chat_text(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": standalone_query},
            ]
        )
    except OllamaError:
        return standalone_query

    rewritten = rewritten.strip().strip('"').strip()
    return rewritten if rewritten else standalone_query
