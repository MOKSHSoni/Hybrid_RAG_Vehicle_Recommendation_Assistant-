"""Orchestrates ingest -> enrich -> chunk -> embed -> index.

Shared by demo_phase1.py and the test suite so the wiring exists in
exactly one place.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import faiss

import config
from src.chunking.models import Chunk
from src.chunking.vehicle_chunks import build_vehicle_chunks
from src.embeddings.embedder import Embedder, normalize_embeddings
from src.enrichment.models import EnrichedDocument
from src.enrichment.vehicle import VehicleEnrichment
from src.ingestion.csv_loader import load_documents
from src.retrieval.bm25.bm25_index import BM25Index
from src.retrieval.chunk_store import ChunkStore
from src.retrieval.dense.faiss_index import build_faiss_index, load_index, save_index


@dataclass
class KnowledgeBase:
    enriched_documents: List[EnrichedDocument]
    chunk_store: ChunkStore
    faiss_index: faiss.Index
    bm25_index: BM25Index
    embedder: Embedder


def build_knowledge_base(csv_path: Optional[Path] = None, save: bool = True) -> KnowledgeBase:
    csv_path = csv_path or config.RAW_CSV_PATH

    documents = load_documents(csv_path, source_name="cars_cleaned")
    enriched_documents = VehicleEnrichment().enrich_batch(documents)

    chunks: List[Chunk] = []
    for enriched in enriched_documents:
        chunks.extend(build_vehicle_chunks(enriched))
    for i, chunk in enumerate(chunks):
        chunk.vector_index = i

    embedder = Embedder()
    vectors = embedder.embed_texts([c.text for c in chunks])
    vectors = normalize_embeddings(vectors)

    faiss_idx = build_faiss_index(vectors)
    bm25_idx = BM25Index.build([c.text for c in chunks], [c.chunk_id for c in chunks])
    chunk_store = ChunkStore(chunks)

    if save:
        chunk_store.save(config.CHUNK_STORE_PATH)
        save_index(faiss_idx, config.FAISS_INDEX_PATH)
        bm25_idx.save(config.BM25_INDEX_PATH)
        _save_enriched_documents(enriched_documents, config.ENRICHED_VEHICLES_PATH)

    return KnowledgeBase(
        enriched_documents=enriched_documents,
        chunk_store=chunk_store,
        faiss_index=faiss_idx,
        bm25_index=bm25_idx,
        embedder=embedder,
    )


def load_knowledge_base() -> KnowledgeBase:
    """Load a previously-saved knowledge base from data/index/ without rebuilding it."""
    chunk_store = ChunkStore.load(config.CHUNK_STORE_PATH)
    faiss_idx = load_index(config.FAISS_INDEX_PATH)
    bm25_idx = BM25Index.load(config.BM25_INDEX_PATH)
    return KnowledgeBase(
        enriched_documents=[],
        chunk_store=chunk_store,
        faiss_index=faiss_idx,
        bm25_index=bm25_idx,
        embedder=Embedder(),
    )


def _save_enriched_documents(documents: List[EnrichedDocument], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in documents], f, indent=2, ensure_ascii=False)
