"""Phase 11: final RAG generation.

Top-K Actual Vehicle Chunks -> Context Builder -> Grounded Qwen3 Prompt ->
Qwen3/Ollama -> Final Answer.

Routed through chat_text (schema-wrapped) first, same as every other
free-form generation call in this project, with three defenses layered on
top after real testing surfaced three distinct failure modes on this
specific, longer, multi-vehicle task (all confirmed to be genuine
run-to-run sampling variance on this CPU-only setup -- e.g. the exact same
5-vehicle prompt produced a complete, well-formed answer on one run and a
one-sentence truncated non-answer on another, both at temperature=0):

1. Even the fast schema-based path occasionally exceeded the default 60s
   Ollama timeout on a genuine (non-simulated) call explaining 5 vehicles
   -- so generation calls use config.GENERATION_FALLBACK_TIMEOUT_SECONDS
   throughout, not the shorter default used elsewhere.
2. The model sometimes produces output that is technically clean (no
   rambling markers, under the length cap) but is really just an echo of
   the structured context block (e.g. "RETRIEVED VEHICLES (5): [Vehicle 1]
   BMW X7, [Vehicle 2] ...") rather than actual explanatory prose --
   caught by _looks_like_context_echo().
3. The model sometimes stops after a short preamble sentence ("Here's the
   recommendation:") without ever actually naming a vehicle -- caught by
   _looks_incomplete() checking whether the answer mentions ANY retrieved
   vehicle by name.

Any of the three routes to the same chat_long_form() fallback (slower,
but reliably complete; see ollama_client.py's module docstring)."""

import config
from src.query.ollama_client import OllamaError, chat_long_form, chat_text, looks_like_rambling
from src.rag.context_builder import build_context_block
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.modes import RetrievalOutcome

_SYSTEM_PROMPT = """You are a vehicle recommendation assistant. You are given a user's request
and RETRIEVED VEHICLE DATA -- the ONLY source of truth. Never invent specifications, prices,
or features not present in that data; if something isn't in the data, say it's not available
rather than guessing.

Write a genuine, natural-language recommendation -- flowing sentences, not a data dump. Do NOT
copy the raw "[Vehicle N]" labels or repeat the retrieved text verbatim; refer to vehicles by
name in your own sentences instead. For example, write something like: "Since you're after an
affordable SUV under 15 Lakh, the Tata Safari (Rs 14.9 Lakh, 7 seats) is a strong match because
..." -- not "RETRIEVED VEHICLES: [Vehicle 1] Tata Safari ...".

For your response:
- State the retrieval mode (Exact / Relaxed / Fallback / Superlative) explicitly near the start.
- For each recommended vehicle, briefly explain WHY it matches the request, citing specific
  retrieved facts (price, seats, fuel, features, etc.) in your own words.
- Compare vehicles against each other where relevant (price, features, fit for the request) --
  if a COMPARISON TABLE is included below, reference its real values rather than re-deriving them.
- Mention limitations or missing data honestly rather than guessing.

CRITICAL, depending on the retrieval mode given to you:
- EXACT: every listed vehicle satisfies every requested constraint.
- RELAXED: some constraints were dropped or widened to find these results (the specifics are
  listed below) -- you MUST tell the user which of their original requirements these results
  do NOT fully satisfy. Never claim a relaxed result meets a constraint that was relaxed away.
- FALLBACK: no constraints could be verified at all -- these are general semantic matches only.
  You MUST tell the user this plainly and not claim any specific requirement (price, seats,
  brand, fuel, etc.) is guaranteed to be met by these results.
- SUPERLATIVE: results are ranked by a direct, exact sort on one real metadata field (named in
  the context below), NOT by search relevance -- say so explicitly, e.g. "ranked by top speed,
  highest first," rather than implying these were chosen for general relevance to the request.

IMPORTANT: output your answer directly and immediately. Do not reason, plan, or think out loud
first -- you already have everything you need in the data below."""

_ECHO_MARKERS = ("RETRIEVED VEHICLES", "RETRIEVAL MODE:", "[Vehicle 1]", "[Vehicle 2]")


def generate_answer(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore) -> str:
    """Returns the final natural-language answer, or a graceful,
    non-crashing error message if the Ollama call fails/times out."""
    if not outcome.results:
        return (
            "I couldn't find any vehicles matching your request, even after relaxing several "
            "constraints. Could you try a broader or different query?"
        )

    context = build_context_block(outcome, chunk_store)
    prompt = f"User request: {query}\n\n{context}\n\nWrite the recommendation response now."
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

    try:
        answer = chat_text(messages=messages, timeout=config.GENERATION_FALLBACK_TIMEOUT_SECONDS)
        if looks_like_rambling(answer) or _looks_like_context_echo(answer) or _looks_incomplete(answer, outcome):
            # The fast schema path rambled, echoed the raw context back, or
            # trailed off incomplete -- fall back to letting the model think
            # freely (slower, ~tens of seconds, but reliably a real written
            # answer; see ollama_client.py's module docstring).
            answer = chat_long_form(messages=messages)
        return answer
    except OllamaError as e:
        return _fallback_message(outcome, e)


def _looks_like_context_echo(answer: str) -> bool:
    return any(marker in answer for marker in _ECHO_MARKERS)


def _looks_incomplete(answer: str, outcome: RetrievalOutcome) -> bool:
    """Catches a real observed failure mode: a short preamble sentence
    ("Here's the recommendation:") with the model then stopping before
    ever actually naming a vehicle."""
    if len(answer) < 100:
        return True
    names = (m.best_chunk.metadata.get("name") for m in outcome.results)
    return not any(name and name in answer for name in names)


def _fallback_message(outcome: RetrievalOutcome, error: OllamaError) -> str:
    names = ", ".join(m.best_chunk.metadata.get("name", "?") for m in outcome.results[:5])
    return (
        "Sorry, I couldn't generate a written summary right now -- the local AI model backend "
        f"appears to be unavailable ({error}). Here are the top matching vehicles found "
        f"(mode: {outcome.mode}) without commentary: {names}. Please try again shortly."
    )
