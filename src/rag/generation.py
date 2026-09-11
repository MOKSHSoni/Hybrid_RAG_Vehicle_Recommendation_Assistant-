"""Phase 11: final RAG generation.

Top-K Actual Vehicle Chunks -> Context Builder -> Grounded Qwen3 Prompt ->
Qwen3/Ollama -> Final Answer.

Routed through chat_text (schema-wrapped) first, same as every other
free-form generation call in this project. Real testing surfaced several
distinct failure modes on this specific, longer, multi-vehicle task (all
genuine run-to-run sampling variance on this CPU-only setup -- the exact
same prompt produced a clean answer on one run and a truncated non-answer
on another, both at temperature=0):

1. Even the fast schema-based path occasionally exceeded the default 60s
   Ollama timeout -- so generation uses config.GENERATION_FALLBACK_TIMEOUT_SECONDS.
2. Echoing the structured context block back ("RETRIEVED VEHICLES (5):
   [Vehicle 1] BMW X7 ...") instead of writing prose -- _looks_like_context_echo().
3. Stopping after a preamble ("Here's the recommendation:") without ever
   naming a vehicle -- _looks_incomplete().
4. Narrating the task in third person ("The user is looking for...", "The
   retrieved vehicles are...") and dumping every spec field instead of
   recommending -- _looks_like_meta_narration(), plus a much firmer
   _SYSTEM_PROMPT with explicit good/bad examples.

needs_regeneration() gates all of these and routes to chat_long_form()
(slower but more reliable; see ollama_client.py's module docstring).

Note what needs_regeneration() deliberately does NOT use:
ollama_client.looks_like_rambling(). Its 500-char ceiling is calibrated
for one-line outputs, and reusing it here flagged every genuine
multi-vehicle answer -- forcing the slow path on essentially every query,
which then often timed out and replaced a perfectly readable answer with
an error. Regeneration is now the exception, not the norm, and a failed
regeneration returns None so the caller keeps what it already showed."""

from typing import Iterator, Optional

import config
from src.query.ollama_client import (
    OllamaError,
    chat_long_form,
    chat_text,
    chat_text_stream,
)
from src.rag.context_builder import build_context_block
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.modes import RetrievalOutcome

_SYSTEM_PROMPT = """You are a car salesperson replying directly to a customer.

HOW TO WRITE (most important):
1. Talk TO them, using "you". NEVER write about "the user" in third person, and never mention
   "retrieved vehicles", "the data", "the retrieval mode", or anything about how you searched.
   Write "Based on what you're after, the Tata Safari..." -- NOT "The user is looking for...".
2. Do NOT list every specification. Pick only the 2-3 facts per car that actually matter for
   what they asked, and say why they matter. A wall of numbers is a failure, not thoroughness.
3. Never repeat the same point twice.
4. Keep it to roughly 3-6 sentences total. Recommend a clear favourite and say why.

GROUNDING: the vehicle data below is your ONLY source of truth. Never invent a specification,
price, or feature that isn't there. If something is missing, say so plainly or leave it out --
never guess. If a COMPARISON TABLE is provided, use its exact values.

GROUPING: when one sentence covers two or more cars ("the X and Y both..."), check the figure for
EACH of them first. Only state what is true of every car you named. If they differ, split them up
or leave the point out -- e.g. do not write "the A and B both seat fewer" when only A does.
If a figure is N/A for one of them, that car is UNKNOWN on that point, not low -- leave it out of
the comparison entirely rather than writing "A and B have smaller boots (N/A and 326L)".

HONESTY -- this depends on the mode given below, and you must get it right:
- EXACT: everything listed genuinely meets what they asked. Just recommend naturally; there is
  no need to announce the mode.
- RELAXED: some of their requirements were loosened to find anything at all. You MUST say plainly
  which of their requirements these cars do NOT meet. Never imply a relaxed-away requirement was met.
- FALLBACK: nothing could be verified against their requirements -- these are loose matches only.
  Say so plainly and promise nothing about price, seats, brand, or fuel.
- SUPERLATIVE: these are ranked by sorting one real spec (named below), not by general fit. Say
  which spec they're ranked by.

GOOD EXAMPLE: "For an affordable 7-seater, the Mahindra XUV500 is your best bet at Rs 13.8 Lakh --
it seats 7, returns about 16 kmpl, and costs a fraction of the alternatives. The Volvo XC90
(Rs 80.99 Lakh) and BMW X7 (Rs 93 Lakh) also seat 7 and are far more refined, but they're in a
completely different price bracket."

BAD EXAMPLE (never do this): "The user is looking for a 7-seater. The retrieved vehicles are
priced from Rs 13.8 Lakh to Rs 93 Lakh. The top speeds are 227 km/h, 180 km/h and 144.57 km/h.
The boot space is 326 L, 530 L and N/A..."

Output the reply immediately. Do not think out loud first."""

_ECHO_MARKERS = ("RETRIEVED VEHICLES", "RETRIEVAL MODE:", "[Vehicle 1]", "[Vehicle 2]")

# Third-person narration about the request, or about the retrieval machinery.
# Both mean the model is describing the task instead of answering it.
_META_NARRATION_MARKERS = (
    "the user is looking for",
    "the user wants",
    "the user asked",
    "the user's request",
    "the retrieved vehicles",
    "the retrieval mode",
    "let me think",
    "i need to",
    "i should",
)

# A genuinely detailed 3-vehicle recommendation lands around 600-1500 chars.
# This ceiling only catches real runaway, not normal thoroughness.
_MAX_ANSWER_CHARS = 4000


NO_RESULTS_MESSAGE = (
    "I couldn't find any vehicles matching your request, even after relaxing several "
    "constraints. Could you try a broader or different query?"
)


def _build_messages(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore):
    context = build_context_block(outcome, chunk_store)
    prompt = f"User request: {query}\n\n{context}\n\nWrite the recommendation response now."
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def generate_answer(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore) -> str:
    """Returns the final natural-language answer, or a graceful,
    non-crashing error message if the Ollama call fails/times out."""
    if not outcome.results:
        return NO_RESULTS_MESSAGE

    messages = _build_messages(query, outcome, chunk_store)

    try:
        answer = chat_text(messages=messages, timeout=config.GENERATION_FALLBACK_TIMEOUT_SECONDS)
        if needs_regeneration(answer, outcome):
            # The fast schema path rambled, echoed the raw context back, or
            # trailed off incomplete -- fall back to letting the model think
            # freely (slower, ~tens of seconds, but reliably a real written
            # answer; see ollama_client.py's module docstring).
            answer = chat_long_form(messages=messages)
        return answer
    except OllamaError as e:
        return _fallback_message(outcome, e)


def generate_answer_stream(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore) -> Iterator[str]:
    """Streaming counterpart to generate_answer(), for the UI.

    Generation is by far the slowest stage (measured at ~30-210s on the
    target CPU-only hardware, scaling with how many vehicles it has to
    write about), so streaming is purely a perceived-latency win: the user
    sees prose within a few seconds instead of staring at a spinner.

    The quality checks generate_answer() relies on can only run once the
    full text exists, so they are NOT applied here -- the caller streams
    this, then asks needs_regeneration() about the accumulated text and
    calls regenerate_long_form() to replace it if required.
    """
    if not outcome.results:
        yield NO_RESULTS_MESSAGE
        return

    messages = _build_messages(query, outcome, chunk_store)
    try:
        for delta in chat_text_stream(messages=messages, timeout=config.GENERATION_FALLBACK_TIMEOUT_SECONDS):
            yield delta
    except OllamaError as e:
        yield _fallback_message(outcome, e)


def needs_regeneration(answer: str, outcome: RetrievalOutcome) -> bool:
    """Whether an answer failed a quality check and should be replaced via
    the slower, more reliable long-form path.

    Deliberately does NOT use ollama_client.looks_like_rambling(): that
    helper's 500-character ceiling is calibrated for one-line outputs
    (query rewrites, HyDE snippets). A legitimate recommendation covering
    three vehicles runs well past 500 chars, so reusing it here fired on
    essentially every substantive answer -- forcing the slow regeneration
    path every time, which then frequently timed out and left the user
    with an error instead of the answer they'd already been shown.
    """
    return (
        _looks_like_meta_narration(answer)
        or _looks_like_context_echo(answer)
        or _looks_incomplete(answer, outcome)
        or len(answer) > _MAX_ANSWER_CHARS
    )


def _looks_like_meta_narration(answer: str) -> bool:
    """Catches answers written ABOUT the request rather than TO the person
    -- "The user is looking for...", "The retrieved vehicles are..." --
    which read as internal analysis leaking into the response."""
    lowered = answer.lower()
    return any(marker in lowered for marker in _META_NARRATION_MARKERS)


def regenerate_long_form(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore) -> Optional[str]:
    """Returns a replacement answer, or None if regeneration failed.

    None (rather than an error message) is deliberate: the caller has
    already shown the user a real, readable answer. Swapping that for
    "Sorry, I couldn't generate..." because the *second, optional* attempt
    timed out would be strictly worse than leaving the flawed original in
    place.
    """
    try:
        return chat_long_form(messages=_build_messages(query, outcome, chunk_store))
    except OllamaError:
        return None


def _looks_like_context_echo(answer: str) -> bool:
    return any(marker in answer for marker in _ECHO_MARKERS)


_MIN_ANSWER_CHARS = 60


def _looks_incomplete(answer: str, outcome: RetrievalOutcome) -> bool:
    """Catches a real observed failure mode: a short preamble sentence
    ("Here's the recommendation:") with the model then stopping before
    ever actually naming a vehicle.

    Naming a retrieved vehicle is the primary signal -- a truncated answer
    essentially never gets that far. The length floor is only a secondary
    guard against a degenerate reply that IS just a bare vehicle name. An
    earlier version applied the length floor unconditionally, which
    misfired on terse-but-valid answers and sent them through a needless
    (and very slow) regeneration.
    """
    names = (m.best_chunk.metadata.get("name") for m in outcome.results)
    if not any(name and name in answer for name in names):
        return True
    return len(answer) < _MIN_ANSWER_CHARS


def _fallback_message(outcome: RetrievalOutcome, error: OllamaError) -> str:
    names = ", ".join(m.best_chunk.metadata.get("name", "?") for m in outcome.results[:5])
    return (
        "Sorry, I couldn't generate a written summary right now -- the local AI model backend "
        f"appears to be unavailable ({error}). Here are the top matching vehicles found "
        f"(mode: {outcome.mode}) without commentary: {names}. Please try again shortly."
    )
