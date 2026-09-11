from conftest import requires_ollama

from src.app.orchestrator import Pipeline, TurnResult, answer_query


@requires_ollama
def test_answer_query_returns_populated_result(knowledge_base):
    pipeline = Pipeline(
        kb=knowledge_base,
        dense=None,
        bm25=None,
        hybrid=None,
        hyde=None,
        reranker=None,
        known_brands=[],
    )
    # Build the real retrievers against the shared knowledge_base fixture
    # rather than Pipeline.build(), which would reload models unnecessarily.
    from src.query.regex_extraction import known_brands_from_chunk_store
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.retrieval.bm25.retriever import BM25Retriever
    from src.retrieval.dense.hyde_retriever import HydeRetriever
    from src.retrieval.dense.retriever import DenseRetriever
    from src.retrieval.hybrid.hybrid_retriever import HybridRetriever

    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)
    pipeline.dense, pipeline.bm25, pipeline.hybrid = dense, bm25, hybrid
    pipeline.hyde = HydeRetriever(dense)
    pipeline.reranker = CrossEncoderReranker(knowledge_base.chunk_store)
    pipeline.known_brands = known_brands_from_chunk_store(knowledge_base.chunk_store)

    result = answer_query(pipeline, "affordable SUV under 15 lakh", history=[])

    assert isinstance(result, TurnResult)
    assert len(result.answer) > 0
    assert result.log.mode in ("exact", "relaxed", "fallback")
    assert result.log.total_ms() > 0
    assert result.expansion_queries is None  # debug-only extras off by default
    assert result.hyde_description is None


@requires_ollama
def test_answer_query_with_debug_extras(knowledge_base):
    from src.query.regex_extraction import known_brands_from_chunk_store
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.retrieval.bm25.retriever import BM25Retriever
    from src.retrieval.dense.hyde_retriever import HydeRetriever
    from src.retrieval.dense.retriever import DenseRetriever
    from src.retrieval.hybrid.hybrid_retriever import HybridRetriever

    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    bm25 = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    hybrid = HybridRetriever(dense, bm25, knowledge_base.chunk_store)
    pipeline = Pipeline(
        kb=knowledge_base,
        dense=dense,
        bm25=bm25,
        hybrid=hybrid,
        hyde=HydeRetriever(dense),
        reranker=CrossEncoderReranker(knowledge_base.chunk_store),
        known_brands=known_brands_from_chunk_store(knowledge_base.chunk_store),
    )

    result = answer_query(pipeline, "affordable SUV under 15 lakh", history=[], include_expansion_hyde_debug=True)

    assert result.expansion_queries is not None
    assert result.hyde_description is not None
