# Car Sales Assistant — Hybrid RAG Vehicle Recommendation System

A conversational, hybrid retrieval-augmented generation (RAG) system that
recommends vehicles from a ~150-row car dataset, combining dense + sparse
retrieval, query understanding, reranking, and constraint-based filtering
with graceful fallback. Built incrementally, one phase at a time.

**Status: Phase 1 (Data Ingestion & Knowledge Base) complete.**

## Setup

Requires Python 3.11 and internet access on first run (downloads the
`all-MiniLM-L6-v2` embedding model, ~80MB, from the Hugging Face Hub).

### Windows (PowerShell)

```powershell
python -m venv .venv
# If activation is blocked by execution policy, run this first:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact, fully-resolved dependency closure
(direct + transitive) as verified by installing into a clean venv — not
just the handful of top-level packages. Notably it pulls in `torch`
(CPU build) as a dependency of `sentence-transformers`, which is a
sizeable download.

## Running

```bash
python demo_phase1.py
```

Builds the knowledge base from `data/raw/cars_cleaned.csv` (ingest ->
enrich -> chunk -> embed -> index), saves the FAISS/BM25 indices and
chunk store to `data/index/`, and runs three sample queries against both
retrieval backends to prove the pipeline works end to end.

## Testing

```bash
pytest
```

## Project layout

```
config.py              # all tunable parameters (paths, model names, chunk size, ...)
data/
  raw/                  # cars_cleaned.csv (committed; original, never modified by any code)
  processed/            # generated, human-inspectable enriched vehicle data (gitignored)
  index/                # generated FAISS/BM25 indices + chunk store (gitignored)
src/
  ingestion/            # GENERIC CSV -> NormalizedDocument loader (no domain knowledge)
  enrichment/            # domain-specific enrichment: BaseEnrichment interface + VehicleEnrichment
  chunking/              # generic recursive text splitter + vehicle chunk templates
  embeddings/            # sentence-transformers embedding wrapper
  retrieval/
    bm25/                # sparse retrieval (rank-bm25)
    dense/               # dense retrieval (FAISS IndexFlatIP)
    chunk_store.py       # shared chunk persistence/join layer
  pipeline.py            # orchestrates the full ingest->index pipeline
tests/                   # pytest suite (unit + integration, run locally)
demo_phase1.py           # Phase 1 deliverable: end-to-end knowledge-base demo
```

The ingestion layer is dataset-agnostic by design: it maps any structured
CSV source into a common document representation with no knowledge of
"vehicles." Everything vehicle-specific (e.g. deriving a body type, since
the source data has no such column) lives in the enrichment layer, which
inspects the CSV's actual columns at runtime rather than assuming a fixed
schema. This is what will let a future domain (e.g. real estate,
products) reuse the ingestion and chunking infrastructure by writing a
new `BaseEnrichment` subclass, without touching the generic layers.

## Roadmap

This is Phase 1 of a 14-phase build (see the project's implementation
plan for the full roadmap): Basic RAG -> BM25 -> Dense -> Hybrid -> Query
Understanding -> Context -> Expansion -> HyDE -> Deduplication ->
Cross-Encoder -> Relaxation -> Final RAG -> Evaluation -> UI -> optional
Langfuse tracing. Each phase is built and validated before the next
begins.
