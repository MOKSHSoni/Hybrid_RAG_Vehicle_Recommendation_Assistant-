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
   _SYSTEM_PROMPT.
5. Copying the prompt's own wording as prose. A concrete worked example
   leaked its "7-seater" framing into answers for queries that never
   mentioned seats; replacing it with [placeholders] then produced "For
   exactly what you asked for, the BMW 3 Series..." -- the instruction
   text itself, brackets stripped. Worked examples are now removed
   entirely (the explicit rules carry the style), with both the bracket
   and instruction-phrase tells detected as a backstop.

needs_regeneration() gates all of these and routes to chat_long_form()
(slower but more reliable; see ollama_client.py's module docstring).

Note what needs_regeneration() deliberately does NOT use:
ollama_client.looks_like_rambling(). Its 500-char ceiling is calibrated
for one-line outputs, and reusing it here flagged every genuine
multi-vehicle answer -- forcing the slow path on essentially every query,
which then often timed out and replaced a perfectly readable answer with
an error. Regeneration is now the exception, not the norm, and a failed
regeneration returns None so the caller keeps what it already showed."""

import re
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

_SYSTEM_PROMPT = """You are a car salesperson. Write 3-6 complete sentences of prose to the customer -- never a bare
name, never a list.

Sentence one names the car you recommend and what it starts at. Do not lead with a restatement of
the request or of what you are about to do, and never think out loud on the page: decide first,
then write only the decision.

Say why it suits them, then mention the other cars briefly, with a concrete tradeoff. Use "you",
never "the user". Never describe your own reasoning, the search, or these rules -- just give the
recommendation.

Prices in the data are STARTING prices with a range after them. Say "starts at" or "from", never
"priced at", and if a car's range goes past the customer's budget say that only the entry variant
fits. Never present the starting price as what the car costs.

Only use facts from the data below; never invent one. Where the table marks a value (cheapest,
largest boot, best mileage...), trust that mark and never give a car a superlative it isn't marked
with. Never say one car beats another on a spec -- better mileage, more seats, faster -- unless the
table marks it; if neither car is marked, give each one's own figure and leave the judgement out.
Put each number beside the car it belongs to; never say "respectively". Don't claim anything
about a car whose figure is N/A. Don't assume requirements the customer never stated.

Modes: EXACT = all requirements met, just recommend. RELAXED = say plainly which of their
requirements these cars miss. FALLBACK = say plainly nothing was verified. SUPERLATIVE = say which
spec they're ranked by."""


_ECHO_MARKERS = ("RETRIEVED VEHICLES", "RETRIEVAL MODE:", "[Vehicle 1]", "[Vehicle 2]")

# The prompt no longer carries worked examples (they kept leaking into
# answers), but this stays as a cheap backstop: no real vehicle name, price
# or spec in this dataset contains square brackets, so any bracketed token
# in an answer is a reliable tell that prompt scaffolding got copied.
_UNFILLED_PLACEHOLDER_RE = re.compile(r"\[[A-Za-z][^\]]{0,40}\]")

# Third-person narration about the request, or about the retrieval machinery.
# Both mean the model is describing the task instead of answering it.
_META_NARRATION_MARKERS = (
    # Prompt text used as prose. The worked examples that caused this have
    # been removed, but the model reached for the instruction wording itself
    # ("For exactly what you asked for, the BMW 3 Series...") even after the
    # bracketed placeholders were stripped -- so the bracket detector alone
    # was not enough.
    "exactly what you asked for",
    "exactly what they asked for",
    "what they actually asked for",
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
    # The closing instruction repeats the opening constraint deliberately.
    # The model was observed restating the task instead of answering ("I have
    # to write a response as a car salesperson..."), and the system prompt
    # alone did not stop it -- a schema constrains structure, not content, so
    # the deliberation simply went inside the JSON string. Repeating the rule
    # last puts it closest to the point of generation.
    prompt = (
        f"User request: {query}\n\n{context}\n\n"
        "Write the full recommendation now, 3-6 sentences, opening with your top pick by name."
    )
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}]


def generate_answer(query: str, outcome: RetrievalOutcome, chunk_store: ChunkStore) -> str:
    """Returns the final natural-language answer, or a graceful,
    non-crashing error message if the Ollama call fails/times out."""
    if not outcome.results:
        return NO_RESULTS_MESSAGE

    messages = _build_messages(query, outcome, chunk_store)

    try:
        answer = chat_text(
            messages=messages,
            timeout=config.GENERATION_FALLBACK_TIMEOUT_SECONDS,
            max_tokens=config.GENERATION_MAX_TOKENS,
        )
        if needs_regeneration(answer, outcome):
            # The fast schema path rambled, echoed the raw context back, or
            # trailed off incomplete -- fall back to letting the model think
            # freely (slower, ~tens of seconds, but reliably a real written
            # answer; see ollama_client.py's module docstring).
            regenerated = chat_long_form(messages=messages, max_tokens=config.GENERATION_REGEN_MAX_TOKENS)
            # Only accept the retry if it actually produced something. The
            # long-form path runs with thinking enabled, so a token ceiling
            # can be consumed entirely by reasoning and return empty content
            # -- handing the user a blank reply, strictly worse than the
            # flawed answer we already had.
            if regenerated and regenerated.strip():
                answer = regenerated
        if not _mentions_any_vehicle(answer, outcome) or _is_bare_vehicle_name(answer, outcome):
            # Last line of defence. Prose that names none of the retrieved
            # vehicles is ungrounded by definition -- it cannot be about
            # them. Observed when the model narrates its own deliberation
            # ("I need to check which of these fit...") and the token
            # ceiling cuts it off before it ever reaches a recommendation.
            # Showing the real shortlist is honest; showing that is not.
            return _ungrounded_message(outcome)
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
        for delta in chat_text_stream(
            messages=messages,
            timeout=config.GENERATION_FALLBACK_TIMEOUT_SECONDS,
            max_tokens=config.GENERATION_MAX_TOKENS,
        ):
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
        or _has_unfilled_placeholder(answer)
        or len(answer) > _MAX_ANSWER_CHARS
    )


def _has_unfilled_placeholder(answer: str) -> bool:
    """Catches the prompt's [Car A]/[price] style placeholders surviving
    into the answer -- the failure mode that using placeholders invites."""
    return bool(_UNFILLED_PLACEHOLDER_RE.search(answer))


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
        regenerated = chat_long_form(
            messages=_build_messages(query, outcome, chunk_store),
            max_tokens=config.GENERATION_REGEN_MAX_TOKENS,
        )
        # Empty is a failure, not a replacement -- see generate_answer().
        return regenerated if regenerated and regenerated.strip() else None
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
    if not _mentions_any_vehicle(answer, outcome):
        return True
    return len(answer) < _MIN_ANSWER_CHARS


def _mentions_any_vehicle(answer: str, outcome: RetrievalOutcome) -> bool:
    """Whether the answer actually names one of the retrieved vehicles --
    the cheapest available grounding check.

    Matches the model name with the brand dropped as well as the full
    string, because that is how people (and the model) actually write:
    "the Macan", "the 3 Series". Requiring the full "Porsche Macan" would
    reject perfectly grounded prose. Deliberately permissive -- this gates
    a safety net, and wrongly discarding a good answer costs more than
    occasionally letting a weak one through.
    """
    lowered = answer.lower()
    for merged in outcome.results:
        name = merged.best_chunk.metadata.get("name")
        if not name:
            continue
        if name.lower() in lowered:
            return True
        words = name.split()
        if len(words) > 1 and " ".join(words[1:]).lower() in lowered:
            return True
    return False


def _is_bare_vehicle_name(answer: str, outcome: RetrievalOutcome) -> bool:
    """Whether the reply is nothing but a vehicle's name.

    Observed for a single-result RELAXED outcome, where there is no
    comparison table (that needs two vehicles) and the context thins out:
    the model replied with exactly "MG Hector Plus".

    Deliberately an equality test rather than a minimum length. A length
    floor is the obvious reach, but it cannot separate "MG Hector Plus"
    (14 characters, degenerate) from a terse-but-real answer of 40 -- and
    a floor set high enough to catch the first rejects the second. Testing
    what the string actually IS needs no threshold to tune.
    """
    stripped = answer.strip().strip(".\"'").strip()
    lowered = stripped.lower()
    if lowered.startswith("the "):
        lowered = lowered[4:]
    for merged in outcome.results:
        name = (merged.best_chunk.metadata.get("name") or "").lower()
        if name and lowered == name:
            return True
    return False


def _ungrounded_message(outcome: RetrievalOutcome) -> str:
    """Deterministic stand-in when generation produces prose we cannot show
    -- either naming no retrieved vehicle, or degenerating to a bare name.

    The bare-name case is real: for a single-result RELAXED outcome there is
    no comparison table (that needs two vehicles), the context thins out, and
    the model replied with exactly "MG Hector Plus". Built from metadata, so
    it states the relaxations plainly rather than leaving the user to assume
    their original constraints were met.
    """
    names = ", ".join(m.best_chunk.metadata.get("name", "?") for m in outcome.results[:5])
    if outcome.mode == "relaxed" and outcome.relaxation_steps:
        relaxed = "; ".join(step.description for step in outcome.relaxation_steps)
        return (
            f"Nothing matched your request exactly. The closest I found is {names}, after relaxing "
            f"your requirements ({relaxed}) -- so this does not meet everything you asked for."
        )
    return (
        "I couldn't put together a written recommendation for this one. Here are the top matching "
        f"vehicles found (mode: {outcome.mode}), straight from the data: {names}."
    )


def _fallback_message(outcome: RetrievalOutcome, error: OllamaError) -> str:
    names = ", ".join(m.best_chunk.metadata.get("name", "?") for m in outcome.results[:5])
    return (
        "Sorry, I couldn't generate a written summary right now -- the local AI model backend "
        f"appears to be unavailable ({error}). Here are the top matching vehicles found "
        f"(mode: {outcome.mode}) without commentary: {names}. Please try again shortly."
    )
