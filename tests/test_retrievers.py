import config
from src.retrieval.bm25.retriever import BM25Retriever
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.models import RetrievalResult


def test_dense_retriever_returns_ranked_results(knowledge_base):
    retriever = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    results = retriever.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert len(results) == 5
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.method == "dense" for r in results)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    # Scores should be non-increasing (best match first).
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # The exact-entity query should surface the Cayenne itself at rank 1.
    assert "Cayenne" in results[0].chunk.metadata["name"]


def test_bm25_retriever_returns_ranked_results(knowledge_base):
    retriever = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    results = retriever.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert len(results) == 5
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.method == "bm25" for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert "Cayenne" in results[0].chunk.metadata["name"]


def test_dense_retriever_top_k_larger_than_corpus_does_not_crash(knowledge_base):
    retriever = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    results = retriever.retrieve("SUV", top_k=10_000)
    assert len(results) == len(knowledge_base.chunk_store)  # capped at corpus size, not padded with -1 entries


def test_default_top_k_used_when_unspecified(knowledge_base):
    retriever = BM25Retriever(knowledge_base.bm25_index, knowledge_base.chunk_store)
    results = retriever.retrieve("sedan")
    assert len(results) == config.DEFAULT_TOP_K
