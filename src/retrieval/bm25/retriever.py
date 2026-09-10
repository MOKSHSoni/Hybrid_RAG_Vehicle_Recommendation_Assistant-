"""BM25 (sparse) retrieval, wrapped behind the common RetrievalResult
interface used by every retrieval backend."""

from typing import List

import config
from src.retrieval.bm25.bm25_index import BM25Index
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.models import RetrievalResult


class BM25Retriever:
    def __init__(self, bm25_index: BM25Index, chunk_store: ChunkStore):
        self._bm25_index = bm25_index
        self._chunk_store = chunk_store

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[RetrievalResult]:
        hits = self._bm25_index.search(query, top_k)

        results = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = self._chunk_store.get_by_chunk_id(chunk_id)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank, method="bm25"))
        return results
