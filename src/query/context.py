"""Multi-turn conversation context resolution.

Resolves follow-up queries ("what about diesel options", "any cheaper
ones?") into standalone queries using recent conversation history. Pure
text rewriting -- never touches structured constraints directly, so it
doesn't run afoul of the "LLM never controls filtering" rule.
"""

from typing import List

import config
from src.query.models import ConversationTurn
from src.query.ollama_client import OllamaError, chat_text


def resolve_context(query: str, history: List[ConversationTurn]) -> str:
    """Returns a standalone version of `query`. Falls back to `query`
    unchanged if there's no history, or if the LLM call fails."""
    if not history:
        return query

    recent = history[-config.CONVERSATION_HISTORY_TURNS :]
    history_text = "\n".join(f"{turn.role}: {turn.content}" for turn in recent)

    prompt = f"""Conversation so far:
{history_text}

Latest user message: "{query}"

Rewrite the latest user message as a standalone vehicle search query that does
not depend on the conversation above -- resolve references like "that one",
"what about diesel", "cheaper options", "the second one", etc. using the
context. If the latest message is already standalone, return it unchanged.
Respond with ONLY the rewritten query text, no quotes, no explanation."""

    try:
        rewritten = chat_text(messages=[{"role": "user", "content": prompt}])
    except OllamaError:
        return query

    rewritten = rewritten.strip().strip('"').strip()
    return rewritten if rewritten else query
