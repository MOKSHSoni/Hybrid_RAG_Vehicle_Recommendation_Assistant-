"""HyDE (Hypothetical Document Embeddings) -- experimental retrieval path.

Asks Qwen3 to write a short, plausible, listing-style description of a
HYPOTHETICAL vehicle matching the query, then embeds THAT text (instead
of the bare query) for dense retrieval -- doc-shaped text often lands
closer in embedding space to real matching chunks than a short query
does.

CRITICAL: the hypothetical description is synthetic and must never be
returned to a caller as if it were a real vehicle, and must never be
shown to an end user. It exists only to produce a search vector; only
the real Chunks retrieved using that vector carry facts. Treat as
experimental (Phase 12 A/B-evaluates it against plain dense retrieval)
-- not wired into the default retrieval path by default.
"""

from typing import Optional

from src.query.ollama_client import OllamaError, chat_text

_SYSTEM_PROMPT = """You write a short, plausible product-listing-style description of a
HYPOTHETICAL vehicle that would perfectly satisfy the given search query. Write it in
the style of a real car listing: mention body type, approximate price range, seating,
fuel type, transmission, and 1-2 key features, as if describing an actual vehicle for
sale, in a confident direct tone (e.g. "The Everest GX is a rugged 7-seat SUV priced
from Rs 12 Lakh, delivering 18 kmpl..."). Use "Rs" for currency, never the symbol.
This description is used ONLY internally to improve search retrieval -- it will never
be shown to a user as a real vehicle, so prioritize matching the query's intent and the
natural phrasing of a real listing over factual precision about any specific real car.

IMPORTANT: output ONLY the finished description, 2-4 sentences, nothing else. You
already know the answer -- state it directly. Do not reason, plan, or think out loud;
do not write phrases like "let me think" or "the user wants"."""


def generate_hypothetical_description(query: str) -> Optional[str]:
    """Returns a synthetic vehicle description for embedding, or None if
    the LLM is unavailable (caller should fall back to embedding the raw
    query instead -- see HydeRetriever)."""
    try:
        text = chat_text(messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": query}])
    except OllamaError:
        return None

    text = text.strip()
    return text if text else None
