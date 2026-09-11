from src.query.models import Constraints
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


def test_build_context_block_superlative_mode_states_direct_sort(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    constraints = Constraints(superlative_field="top_speed_kmph", superlative_direction="desc")
    outcome = RetrievalOutcome(mode="superlative", results=[merged], original_constraints=constraints)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "RETRIEVAL MODE: SUPERLATIVE" in block
    assert "DIRECT SORT" in block
    assert "top_speed_kmph" in block
    assert "not by search relevance" in block


def test_build_context_block_includes_comparison_table_for_multiple_results(knowledge_base):
    macan = _merged_for_macan()
    macan.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    cayenne = MergedResult(
        vehicle_id="cars_cleaned_2",
        best_chunk=knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_2::product_overview"),
        aggregate_score=1.0,
    )
    outcome = RetrievalOutcome(mode="exact", results=[macan, cayenne], original_constraints=None)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "COMPARISON TABLE" in block
    assert "Porsche Macan" in block
    assert "Porsche Cayenne" in block


def test_build_context_block_no_comparison_table_for_single_result(knowledge_base):
    merged = _merged_for_macan()
    merged.best_chunk = knowledge_base.chunk_store.get_by_chunk_id("cars_cleaned_0::product_overview")
    outcome = RetrievalOutcome(mode="exact", results=[merged], original_constraints=None)

    block = build_context_block(outcome, knowledge_base.chunk_store)

    assert "COMPARISON TABLE" not in block
