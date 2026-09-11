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
EVAL_QUERIES_PATH = PROJECT_ROOT / "evaluation" / "queries_v2.json"

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

# ---- Domain vocabularies (shared by enrichment, extraction, evaluation) ----
VALID_BODY_TYPES = ["SUV", "Sedan", "Hatchback", "MPV", "Coupe", "Convertible"]
VALID_FUEL_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid", "CNG"]
VALID_TRANSMISSIONS = ["Automatic", "Manual"]

# ---- Ollama / Qwen3 (Phase 3+) ----
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:4b"  # Q4_K_M quantization (Ollama's default tag for this size)
OLLAMA_TIMEOUT_SECONDS = 60
OLLAMA_TEMPERATURE = 0.0  # deterministic extraction/rewriting
# Longer timeout for chat_long_form()'s fallback path (Phase 11), which lets
# the model think freely rather than forcing fast schema-constrained output.
# Observed during development: immediately after a long run of back-to-back
# real Ollama calls (e.g. the full test suite), this CPU-only laptop's
# inference noticeably slows down (plausibly thermal throttling under
# sustained load) and can exceed even 120s for a single 5-vehicle
# explanation, though the same call reliably finishes in 15-90s in
# isolation. The graceful-degradation path (generation.py's
# _fallback_message) is the real safety net for this, not the timeout
# value -- this is set generously but a slow environment can still exceed
# it, by design fails safe rather than hanging forever.
GENERATION_FALLBACK_TIMEOUT_SECONDS = 150

# ---- Query understanding (Phase 3) ----
EXTRACTION_RETRY_COUNT = 1  # retry the LLM extraction once on invalid JSON, then fall back to regex-only
CONVERSATION_HISTORY_TURNS = 6  # how many recent turns feed context resolution

# Fields treated as "hard" (functional necessities, relaxed last in Phase 10).
# Everything else extracted is "soft" by default (relaxed first). This list
# doubles as Phase 10's relaxation order, earliest-relaxed first.
HARD_CONSTRAINT_FIELDS = {"seating_capacity"}
CONSTRAINT_RELAXATION_ORDER = [
    # Extended numeric preferences (top speed, boot space, ground clearance,
    # mileage, engine size) relax first -- secondary "nice to have" specs,
    # dropped outright (no quantile-widening; see modes.py) rather than
    # protected the way the primary budget constraint (price) is.
    "top_speed_kmph",
    "boot_space_l",
    "ground_clearance_mm",
    "mileage_max_kmpl",
    "engine_max_cc",
    "price_max_lakhs",
    "price_min_lakhs",
    "body_type",
    "transmission",
    "brand",
    "fuel_types",
    "seating_capacity",
]

# ---- Hybrid retrieval (Phase 7) ----
# Fraction of the fused score from BM25 (dense gets 1 - alpha). NOT assumed
# optimal -- Phase 12 sweeps HYBRID_FUSION_SWEEP_ALPHAS against the eval set
# and picks the winner; this default is just a commonly-cited starting point.
HYBRID_FUSION_ALPHA = 0.3
HYBRID_FUSION_SWEEP_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]

# ---- Multi-query merge & dedup (Phase 8) ----
# Per-extra-match score boost when a vehicle is confirmed by more than one
# (query, chunk) hit -- gentle tie-breaking signal, not enough to let a
# flood of weak matches outrank one genuinely strong unique match.
DEDUP_EVIDENCE_BOOST = 0.05

# ---- Cross-encoder reranking (Phase 9) ----
# Starting checkpoint per the project's locked decision -- kept configurable
# rather than hard-coded; only change if Phase 12 evaluation shows a
# meaningfully better checkpoint for this retrieval task.
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATE_TOP_N = 30  # candidates fed INTO the cross-encoder (never the full 150-vehicle set)
RERANK_TOP_K = 5  # final results shown to the user, after reranking

# ---- Constraint relaxation & fallback (Phase 10) ----
# Price is widened in steps sized from the ACTUAL dataset's price quantiles
# (not an arbitrary percentage bump) -- e.g. a price ceiling relaxes up to
# the next decile boundary of real Price_Lakhs values, one step at a time.
PRICE_RELAXATION_QUANTILE_STEP = 0.1  # decile-sized steps through the real price distribution
PRICE_RELAXATION_MAX_STEPS = 5  # give up widening price after this many decile steps and drop it outright

# ---- Future phases (placeholders — not read by earlier-phase code) ----
# RERANK_TOP_K = ...               # Phase 9
