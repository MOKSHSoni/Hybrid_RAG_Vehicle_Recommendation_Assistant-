"""Shared chunk persistence/join layer used by both the BM25 and FAISS
retrieval backends.

Because BM25's corpus and FAISS's vectors are built from the exact same
ordered chunk list, Chunk.vector_index is simultaneously the FAISS row
position AND the BM25 corpus position -- one join key instead of two
separate mapping tables. This is the concrete implementation of the
"vehicle_id -> chunk_id -> metadata -> vector/index position" mapping.

JSON (not SQLite) is deliberate at this scale: 150 vehicles / 300 chunks
fits trivially in memory, loads instantly, and stays human-readable for
debugging. Reconsider only if the corpus grows into the tens of
thousands.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import List

from src.chunking.models import Chunk


class ChunkStore:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._by_chunk_id = {c.chunk_id: c for c in chunks}
        self._by_vector_index = {c.vector_index: c for c in chunks if c.vector_index is not None}
        self._by_doc_id = defaultdict(list)
        for c in chunks:
            self._by_doc_id[c.doc_id].append(c)

    def __len__(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def all_texts(self) -> List[str]:
        return [c.text for c in self._chunks]

    def all_chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self._chunks]

    def get_by_chunk_id(self, chunk_id: str) -> Chunk:
        return self._by_chunk_id[chunk_id]

    def get_by_vector_index(self, index: int) -> Chunk:
        return self._by_vector_index[index]

    def get_by_doc_id(self, doc_id: str) -> List[Chunk]:
        """All chunks belonging to one vehicle (e.g. both product_overview
        and features), regardless of which of them a given retrieval pass
        happened to surface."""
        return list(self._by_doc_id[doc_id])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "chunk_type": c.chunk_type,
                "text": c.text,
                "vector_index": c.vector_index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "ChunkStore":
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        chunks = [
            Chunk(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                chunk_type=r["chunk_type"],
                text=r["text"],
                metadata=r["metadata"],
                vector_index=r["vector_index"],
            )
            for r in records
        ]
        return cls(chunks)
