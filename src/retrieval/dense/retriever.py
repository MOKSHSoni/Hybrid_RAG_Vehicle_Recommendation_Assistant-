"""Dense (FAISS) retrieval, wrapped behind the common RetrievalResult
interface used by every retrieval backend."""

from typing import List

import config
from src.embeddings.embedder import Embedder, normalize_embeddings
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.dense.faiss_index import search
from src.retrieval.models import RetrievalResult


class DenseRetriever:
    def __init__(self, faiss_index, chunk_store: ChunkStore, embedder: Embedder):
        self._faiss_index = faiss_index
        self._chunk_store = chunk_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[RetrievalResult]:
        query_vector = normalize_embeddings(self._embedder.embed_texts([query]))[0]
        hits = search(self._faiss_index, query_vector, top_k)

        results = []
        for rank, (vector_index, score) in enumerate(hits, start=1):
            if vector_index < 0:  # FAISS pads with -1 when fewer than top_k results exist
                continue
            chunk = self._chunk_store.get_by_vector_index(vector_index)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank, method="dense"))
        return results
