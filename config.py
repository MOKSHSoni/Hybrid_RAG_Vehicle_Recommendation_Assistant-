"""Central configuration for the car sales assistant RAG pipeline.

All tunable parameters live here instead of being scattered as magic
numbers through the codebase.
"""

from pathlib import Path

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

RAW_CSV_PATH = RAW_DATA_DIR / "cars_cleaned.csv"
ENRICHED_VEHICLES_PATH = PROCESSED_DATA_DIR / "vehicles_enriched.json"
CHUNK_STORE_PATH = INDEX_DIR / "chunk_store.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# ---- Missing-value handling (generic ingestion) ----
MISSING_VALUE_SENTINELS = {"", "-", "na", "n/a", "null", "none", "nan"}

# ---- Chunking (Phase 1) ----
CHUNK_SIZE = 500  # characters; above the observed max real chunk length (~443 chars)
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
CHUNK_TYPES = ["product_overview", "features"]
FEATURES_COLORS_PREVIEW_COUNT = 3

# ---- Vehicle Body_Type heuristic thresholds (enrichment) ----
BODY_TYPE_SUV_GROUND_CLEARANCE_MM = 180
BODY_TYPE_HATCHBACK_MAX_LENGTH_MM = 4000
BODY_TYPE_SUV_LENGTH_MM = 4700
BODY_TYPE_COUPE_MIN_LENGTH_MM = 4200

# ---- Embeddings (Phase 1) ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 384

# ---- FAISS (Phase 1) ----
FAISS_INDEX_TYPE = "IndexFlatIP"

# ---- BM25 (Phase 1) ----
BM25_K1 = 1.5
BM25_B = 0.75

# ---- Retrieval demo defaults ----
DEFAULT_TOP_K = 5

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Future phases (placeholders — not read by any Phase 1 code) ----
# HYBRID_FUSION_ALPHA = ...        # Phase 4/7
# RERANK_TOP_K = ...               # Phase 9
# QUERY_UNDERSTANDING_MODEL = ...  # Phase 3/5 (Ollama model name)
