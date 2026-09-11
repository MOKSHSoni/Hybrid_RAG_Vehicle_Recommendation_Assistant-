from unittest.mock import patch

from conftest import requires_ollama

from src.query.models import Constraints
from src.query.ollama_client import OllamaError
from src.rag.generation import (
    _looks_incomplete,
    _looks_like_context_echo,
    generate_answer,
    generate_answer_stream,
    needs_regeneration,
    regenerate_long_form,
)
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.retrieval.merge import MergedResult
from src.retrieval.modes import RetrievalOutcome, retrieve_with_relaxation


def _merged_for_macan(chunk_store) -> MergedResult:
    return MergedResult(
        vehicle_id="cars_cleaned_0",
        best_chunk=chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview"),
        aggregate_score=1.0,
        supporting_chunks=[],
        match_count=1,
        methods={"hybrid"},
    )


def test_generate_answer_empty_results_no_llm_call(knowledge_base):
    outcome = RetrievalOutcome(mode="fallback", results=[], original_constraints=None)
    with patch("src.rag.generation.chat_text") as mock_chat_text:
        answer = generate_answer("anything", outcome, knowledge_base.chunk_store)
    assert "couldn't find any vehicles" in answer
    mock_chat_text.assert_not_called()


def test_generate_answer_graceful_fallback_on_ollama_error(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    with patch("src.rag.generation.chat_text", side_effect=OllamaError("simulated: connection refused")):
        answer = generate_answer("affordable SUV", outcome, knowledge_base.chunk_store)

    assert "couldn't generate a written summary" in answer
    assert "Porsche Macan" in answer  # still surfaces the vehicle name even without LLM commentary
    assert "exact" in answer


def test_generate_answer_calls_chat_text_with_grounded_context(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    # Realistic-length mock naming the vehicle -- must pass all three quality
    # checks (rambling/echo/incomplete) so this test exercises the fast path
    # in isolation, not an unintended cascade into the real chat_long_form.
    mock_answer = "The Porsche Macan is a strong match for an affordable SUV, offering solid performance and practicality."
    with patch("src.rag.generation.chat_text", return_value=mock_answer) as mock_chat_text:
        answer = generate_answer("affordable SUV", outcome, knowledge_base.chunk_store)

    assert answer == mock_answer
    call_messages = mock_chat_text.call_args.kwargs["messages"]
    user_message = call_messages[-1]["content"]
    assert "affordable SUV" in user_message  # the actual user query
    assert "Porsche Macan" in user_message  # grounded context, not just the query


def test_looks_like_context_echo_detects_raw_dump():
    echo = "RETRIEVED VEHICLES (5), most relevant first: [Vehicle 1] BMW X7, [Vehicle 2] Audi Q8"
    assert _looks_like_context_echo(echo) is True


def test_looks_like_context_echo_accepts_real_prose():
    prose = "Since you're after a spacious SUV, the BMW X7 (Rs 93 Lakh, 7 seats) is a strong match."
    assert _looks_like_context_echo(prose) is False


def test_looks_incomplete_detects_short_preamble_only(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)
    truncated = "For the user's request, the retrieval mode is EXACT. Here's the recommendation:"
    assert _looks_incomplete(truncated, outcome) is True


def test_looks_incomplete_rejects_bare_vehicle_name(knowledge_base):
    # Names a vehicle, but is obviously not a recommendation -- the length
    # floor exists for exactly this degenerate case.
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)
    assert _looks_incomplete("Porsche Macan", outcome) is True


def test_looks_incomplete_accepts_terse_but_valid_answer(knowledge_base):
    # Regression: an unconditional <100-char floor used to flag this valid
    # answer and trigger a needless (very slow) regeneration.
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)
    terse = "The Porsche Macan is a strong match here thanks to its balance of performance and practicality."
    assert len(terse) < 100  # the case the old threshold got wrong
    assert _looks_incomplete(terse, outcome) is False


def test_looks_incomplete_accepts_answer_naming_a_vehicle(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)
    real_answer = (
        "Since you're after a versatile SUV, the Porsche Macan is a strong match thanks to its "
        "balance of performance and everyday usability."
    )
    assert _looks_incomplete(real_answer, outcome) is False


def test_generate_answer_falls_back_to_long_form_on_incomplete_answer(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    truncated = "For the user's request, the retrieval mode is EXACT. Here's the recommendation:"
    with patch("src.rag.generation.chat_text", return_value=truncated), patch(
        "src.rag.generation.chat_long_form", return_value="A complete answer naming the Porsche Macan."
    ) as mock_long_form:
        answer = generate_answer("affordable SUV", outcome, knowledge_base.chunk_store)

    assert answer == "A complete answer naming the Porsche Macan."
    mock_long_form.assert_called_once()


def test_generate_answer_falls_back_to_long_form_on_context_echo(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    echo_answer = "RETRIEVED VEHICLES (1): [Vehicle 1] Porsche Macan"
    with patch("src.rag.generation.chat_text", return_value=echo_answer), patch(
        "src.rag.generation.chat_long_form", return_value="A proper written answer about the Macan."
    ) as mock_long_form:
        answer = generate_answer("affordable SUV", outcome, knowledge_base.chunk_store)

    assert answer == "A proper written answer about the Macan."
    mock_long_form.assert_called_once()


def test_generate_answer_stream_yields_deltas(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    with patch("src.rag.generation.chat_text_stream", return_value=iter(["The Porsche ", "Macan is a ", "solid pick."])):
        out = list(generate_answer_stream("affordable SUV", outcome, knowledge_base.chunk_store))

    assert "".join(out) == "The Porsche Macan is a solid pick."


def test_generate_answer_stream_empty_results_yields_no_results_message(knowledge_base):
    outcome = RetrievalOutcome(mode="fallback", results=[], original_constraints=None)
    with patch("src.rag.generation.chat_text_stream") as mock_stream:
        out = list(generate_answer_stream("anything", outcome, knowledge_base.chunk_store))
    assert "couldn't find any vehicles" in "".join(out)
    mock_stream.assert_not_called()


def test_generate_answer_stream_degrades_gracefully_on_ollama_error(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    with patch("src.rag.generation.chat_text_stream", side_effect=OllamaError("simulated")):
        out = "".join(generate_answer_stream("affordable SUV", outcome, knowledge_base.chunk_store))

    assert "couldn't generate a written summary" in out
    assert "Porsche Macan" in out  # vehicle still surfaced despite the failure


def test_needs_regeneration_accepts_a_long_legitimate_answer(knowledge_base):
    # Regression: needs_regeneration used to call looks_like_rambling(),
    # whose 500-char ceiling is meant for one-line rewrites. Any genuine
    # multi-vehicle recommendation exceeds it, so EVERY substantive answer
    # was sent down the slow regeneration path (which then often timed out).
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    long_but_good = (
        "For a versatile SUV, the Porsche Macan is the standout here at Rs 69.98 Lakh. "
        "It seats five, runs on petrol, and pairs genuine sports-car handling with enough "
        "boot space for a weekend away, which is a rare combination at this size. "
        "The cabin is also considerably better finished than most rivals in the segment, "
        "so it works just as well as an everyday car as it does on a back road. "
        "If your budget stretches, the larger models offer more space, but you give up "
        "the agility that makes the Macan worth having in the first place."
    )
    assert len(long_but_good) > 500  # the exact case the old check got wrong
    assert needs_regeneration(long_but_good, outcome) is False


def test_needs_regeneration_flags_third_person_narration(knowledge_base):
    # The real answer that prompted this fix: written ABOUT the request
    # rather than TO the person, and dumping every spec field.
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    narrated = (
        "The user is looking for an affordable 7-seater vehicle. The retrieved vehicles are all "
        "priced at approximately Rs 13.8 Lakh to Rs 93 Lakh. The Porsche Macan top speed is 254 km/h."
    )
    assert needs_regeneration(narrated, outcome) is True


def test_needs_regeneration_flags_runaway_length(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)
    runaway = "The Porsche Macan is great. " * 200  # ~5600 chars
    assert needs_regeneration(runaway, outcome) is True


def test_regenerate_long_form_returns_none_on_failure(knowledge_base):
    # None, not an error string -- the caller has already shown the user a
    # readable answer and must not replace it with "Sorry, I couldn't...".
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    with patch("src.rag.generation.chat_long_form", side_effect=OllamaError("timed out")):
        assert regenerate_long_form("affordable SUV", outcome, knowledge_base.chunk_store) is None


def test_needs_regeneration_flags_each_failure_mode(knowledge_base):
    merged = _merged_for_macan(knowledge_base.chunk_store)
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    assert needs_regeneration("Okay, let me think about this for the user...", outcome) is True  # rambling
    assert needs_regeneration("RETRIEVED VEHICLES (1): [Vehicle 1] Porsche Macan", outcome) is True  # echo
    assert needs_regeneration("Here's the recommendation:", outcome) is True  # incomplete
    good = "The Porsche Macan is a strong match here thanks to its balance of performance and practicality."
    assert needs_regeneration(good, outcome) is False


@requires_ollama
def test_generate_answer_real_end_to_end(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)
    reranker = CrossEncoderReranker(knowledge_base.chunk_store)

    query = "affordable SUV under 15 lakh"
    constraints = Constraints(body_type="SUV", price_max_lakhs=15.0)
    outcome = retrieve_with_relaxation(query, constraints, knowledge_base.chunk_store, hybrid, reranker)

    answer = generate_answer(query, outcome, knowledge_base.chunk_store)

    assert len(answer) > 20
    # At least one retrieved vehicle's real name should be mentioned -- grounding, not invention.
    assert any(m.best_chunk.metadata["name"] in answer for m in outcome.results)
    # No mode announcement is expected for EXACT: the UI shows a mode badge,
    # and requiring the model to state it verbatim is what produced robotic
    # "The retrieval mode is EXACT" openers. What must hold instead is that
    # the answer reads as a reply to the customer, not analysis about them.
    assert not needs_regeneration(answer, outcome)
