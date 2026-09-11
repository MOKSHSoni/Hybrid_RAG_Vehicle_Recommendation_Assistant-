from src.rag.context_builder import build_context_block
from src.retrieval.merge import MergedResult
from src.retrieval.modes import RelaxationStep, RetrievalOutcome


def _merged_for_macan() -> MergedResult:
    return MergedResult(
        vehicle_id="cars_cleaned_0",
        best_chunk=None,  # not used by build_context_block beyond .metadata
        aggregate_score=1.0,
        supporting_chunks=[],
        match_count=1,
        methods={"hybrid"},
    )


def test_build_context_block_exact_mode_header(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "RETRIEVAL MODE: EXACT" in block
    assert "Porsche Macan" in block


def test_build_context_block_includes_both_chunk_types(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "SUV" in block  # from product_overview
    assert "Peak power" in block  # from features -- present even though best_chunk was product_overview


def test_build_context_block_relaxed_mode_lists_steps(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    steps = [RelaxationStep(field="brand", description="brand removed")]
    outcome = RetrievalOutcome(mode="relaxed", results=[merged], original_constraints=None, relaxation_steps=steps)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "RETRIEVAL MODE: RELAXED" in block
    assert "do NOT" in block  # explicit non-satisfaction warning
    assert "brand removed" in block


def test_build_context_block_fallback_mode_warns_no_constraints_verified(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    outcome = RetrievalOutcome(mode="fallback", results=[merged], original_constraints=None)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "RETRIEVAL MODE: FALLBACK" in block
    assert "general semantic matches only" in block
