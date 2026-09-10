from pathlib import Path

import numpy as np
import pytest

from src.chunking.models import Chunk
from src.retrieval.bm25.bm25_index import BM25Index
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.dense.faiss_index import build_faiss_index, search


def test_faiss_self_match_and_ntotal():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    index = build_faiss_index(vectors)
    assert index.ntotal == 3

    results = search(index, vectors[0], top_k=1)
    top_idx, top_score = results[0]
    assert top_idx == 0
    assert top_score == pytest.approx(1.0, abs=1e-4)


def test_bm25_unique_term_ranks_first():
    texts = [
        "affordable hatchback with good mileage",
        "luxury sedan with premium leather seats",
        "rugged SUV with a powerful turbocharged engine",
    ]
    chunk_ids = ["a", "b", "c"]
    index = BM25Index.build(texts, chunk_ids)

    results = index.search("turbocharged engine", top_k=3)
    assert results[0][0] == "c"


def test_chunk_store_save_load_round_trip(tmp_path: Path):
    chunks = [
        Chunk(
            chunk_id="v1::product_overview",
            doc_id="v1",
            chunk_type="product_overview",
            text="A test vehicle overview.",
            metadata={"name": "Test Vehicle", "price_lakhs": 10.5},
            vector_index=0,
        ),
        Chunk(
            chunk_id="v1::features",
            doc_id="v1",
            chunk_type="features",
            text="Test vehicle features.",
            metadata={"name": "Test Vehicle"},
            vector_index=1,
        ),
    ]
    store = ChunkStore(chunks)
    path = tmp_path / "chunk_store.json"
    store.save(path)

    loaded = ChunkStore.load(path)
    assert len(loaded) == 2
    assert loaded.get_by_chunk_id("v1::product_overview").text == "A test vehicle overview."
    assert loaded.get_by_vector_index(1).chunk_id == "v1::features"
    assert loaded.get_by_chunk_id("v1::product_overview").metadata["price_lakhs"] == 10.5


def test_real_knowledge_base_index_consistency(knowledge_base):
    chunk_store = knowledge_base.chunk_store
    faiss_index = knowledge_base.faiss_index

    assert len(chunk_store) == 300
    assert faiss_index.ntotal == 300

    vector_indices = sorted(c.vector_index for c in chunk_store.all_chunks())
    assert vector_indices == list(range(300))
