# Car Sales Assistant — Hybrid RAG Vehicle Recommendation System

A conversational, hybrid retrieval-augmented generation (RAG) system that
recommends vehicles from a ~150-row car dataset, combining dense + sparse
retrieval, query understanding, query expansion, HyDE, cross-encoder
reranking, and constraint-based filtering with graceful fallback — all
grounded in retrieved data, with a Qwen3/Ollama-powered Streamlit chat UI.

**Status: Phases 1-13 complete** (Phase 14, optional Langfuse tracing, not
yet implemented — the spec explicitly treats it as an add-on only after
everything else works end to end).

## Setup

Requires Python 3.11, internet access on first run (downloads the
`all-MiniLM-L6-v2` embedding model and the `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranker, ~100MB combined, from the Hugging Face Hub), and a local
[Ollama](https://ollama.com) installation with the `qwen3:4b` model pulled.

### 1. Python environment

Windows (PowerShell):
```powershell
python -m venv .venv
# If activation is blocked by execution policy, run this first:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact, fully-resolved dependency closure
(direct + transitive) as verified by installing into a clean venv — not
just the top-level packages. It pulls in `torch` (CPU build) as a
dependency of `sentence-transformers`, a sizeable download.

### 2. Ollama + Qwen3

Install Ollama (`winget install Ollama.Ollama` on Windows, or see
[ollama.com](https://ollama.com)), then pull the model:
```bash
ollama pull qwen3:4b
```
The app degrades gracefully without it (regex-only constraint extraction,
plain vehicle listings instead of written recommendations) but query
understanding, transformation, expansion, HyDE, and generation all need
it for full functionality.

**Performance note (CPU-only hardware):** Qwen3 is a "thinking" model
that can add significant latency if not handled carefully — see
`src/query/ollama_client.py`'s module docstring for the full writeup on
the thinking-suppression techniques this project uses (schema-constrained
decoding + explicit anti-rambling prompting), and `src/rag/generation.py`
for how longer, multi-vehicle generation specifically defends against the
occasional truncated/incomplete response (confirmed to be genuine
run-to-run sampling variance on this hardware, not a deterministic bug).
Under sustained heavy load (e.g. right after running the full test
suite), generation can occasionally exceed even a 150s timeout; this
degrades gracefully to a plain-text vehicle listing rather than crashing.

## Running

### The chat app
```bash
streamlit run streamlit_app.py
```
Enter a natural-language request (e.g. *"affordable 7 seater SUV with
good mileage"*), continue with follow-ups (*"what about diesel
options?"*), and see recommendations labeled Exact / Relaxed / Fallback
with an optional debug panel (query understanding, retrieval internals,
per-stage timings, and — if enabled — informational query
expansion/HyDE output).

### Phase-by-phase demos
Each phase has a standalone, runnable demo proving it works in isolation:
```bash
python demo_phase1.py   # ingestion -> enrichment -> chunking -> embedding -> FAISS/BM25
python demo_phase2.py   # Dense vs BM25 baseline, independently
python demo_phase3.py   # query understanding & multi-turn context resolution
python demo_phase4.py   # query transformation & expansion
python demo_phase5.py   # HyDE (experimental) vs plain dense retrieval
python demo_phase6.py   # metadata filtering
python demo_phase7.py   # hybrid retrieval (score normalization + weighted fusion)
python demo_phase8.py   # multi-query merging & deduplication
python demo_phase9.py   # cross-encoder reranking
python demo_phase10.py  # constraint relaxation & fallback (Exact/Relaxed/Fallback modes)
python demo_phase11.py  # final RAG generation
python demo_phase12.py  # evaluation across retrieval configurations
```

### Testing
```bash
pytest
```
Most tests are hermetic (mocked LLM calls); a subset makes real Ollama
calls and are automatically skipped (not failed) if the server/model
isn't available, via the `requires_ollama` marker in `tests/conftest.py`.

## Project layout

```
config.py                # every tunable parameter (paths, model names, weights, thresholds, ...)
data/
  raw/                   # cars_cleaned.csv (committed; original, never modified by any code)
  processed/             # generated, human-inspectable enriched vehicle data (gitignored)
  index/                 # generated FAISS/BM25 indices + chunk store (gitignored)
evaluation/
  queries_v1.json        # DRAFT eval query set (27 queries, 13 categories) -- needs human review
src/
  ingestion/             # GENERIC CSV -> NormalizedDocument loader (no domain knowledge)
  enrichment/            # domain-specific: BaseEnrichment interface + VehicleEnrichment (body-type heuristic, etc.)
  chunking/              # generic recursive text splitter + vehicle chunk templates
  embeddings/            # sentence-transformers embedding wrapper
  query/                 # context resolution, constraint extraction (regex + Qwen3), transformation, expansion, HyDE
  retrieval/
    bm25/, dense/        # sparse & dense retrievers (+ HyDE-augmented dense)
    hybrid/               # score-normalized weighted fusion
    chunk_store.py        # shared chunk persistence/join layer
    merge.py              # multi-query merge & chunk/vehicle-level dedup
    modes.py              # Exact / Relaxed / Fallback retrieval orchestration
  reranking/             # cross-encoder reranking
  rag/                   # context builder + grounded generation
  evaluation/            # metrics, eval dataset loader, query-understanding & retrieval evaluators
  app/                   # structured logging + the orchestrator tying every phase together
  pipeline.py            # ingest -> enrich -> chunk -> embed -> index orchestration
streamlit_app.py         # the chat UI (Phase 13)
tests/                   # pytest suite (unit + integration, run locally)
demo_phase*.py           # one runnable, standalone demo per phase
```

The ingestion layer is dataset-agnostic by design: it maps any structured
CSV source into a common document representation with no knowledge of
"vehicles." Everything vehicle-specific (deriving a body type, since the
source data has no such column; parsing multi-valued fuel types; etc.)
lives in the enrichment layer, which inspects the CSV's actual columns at
runtime rather than assuming a fixed schema — this is what would let a
future domain (real estate, other products) reuse the ingestion and
chunking infrastructure by writing a new `BaseEnrichment` subclass.

The LLM never directly controls filtering: every constraint the model
extracts is validated against a plain-Python schema check and converted
into a `Constraints` object before Phase 6's metadata filtering (plain
Python/Pandas-style logic) ever touches it. HyDE's hypothetical vehicle
descriptions are synthetic and only ever used to compute a search vector
— never returned or shown as real data. Relaxed and Fallback results are
always labeled as such, with the specific relaxation steps reported, so
the generation prompt can never claim an unverified constraint was met.

## Known limitations (stated explicitly, not hidden)

- **`evaluation/queries_v1.json` is a draft.** Per the project's locked
  decisions, the agent wrote the first version from direct inspection of
  the real dataset; several entries are flagged `needs_verification` and
  are transparently excluded from retrieval metrics rather than skewing
  results. It should be reviewed/refined by a human before being treated
  as ground truth, and revisions should go in `queries_v2.json` etc.
  rather than editing this file in place.
- **The Body_Type heuristic is approximate.** `src/enrichment/body_type.py`
  documents its keyword rules and one known misclassification (BMW 8
  Series, a real-world coupe, falls back to Sedan) as an accepted
  limitation of deriving a field the source data doesn't provide.
- **No true "sort by field" capability.** Superlative queries ("which car
  has the highest top speed") are answered via retrieval relevance, not
  an explicit numeric sort — flagged in the eval set (q23/q24) as a
  known architectural gap, not silently glossed over.
- **LLM output has real run-to-run variance** on this CPU-only setup even
  at temperature 0 (documented in `ollama_client.py` and `generation.py`)
  — the system is built to degrade gracefully around this, not to assume
  perfectly deterministic model output.

## Roadmap

14-phase build: Basic RAG -> BM25 -> Dense -> Hybrid -> Query
Understanding -> Context -> Expansion -> HyDE -> Deduplication ->
Cross-Encoder -> Relaxation -> Final RAG -> Evaluation -> UI -> optional
Langfuse tracing. Phases 1-13 are complete and tested; Phase 14
(Langfuse) is intentionally not implemented yet, per the spec's own
sequencing (it's an optional addition once everything else already works
end to end, building on the structured logging already in
`src/app/logging_utils.py` rather than starting tracing from scratch).
