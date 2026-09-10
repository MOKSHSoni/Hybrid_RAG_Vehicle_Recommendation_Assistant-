import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from src.enrichment.vehicle import VehicleEnrichment
from src.ingestion.csv_loader import load_documents
from src.pipeline import build_knowledge_base
from src.query.ollama_client import OllamaError, chat_text


def _ollama_available() -> bool:
    try:
        # Routed through chat_text (schema-wrapped), not raw chat() --
        # an unconstrained chat() call is subject to Qwen3's ~15s
        # thinking-mode latency (see ollama_client.py) and would make
        # this probe itself flaky/slow. chat_text is reliably fast.
        chat_text(messages=[{"role": "user", "content": "reply with the single word: ok"}], timeout=10)
        return True
    except OllamaError:
        return False


requires_ollama = pytest.mark.skipif(not _ollama_available(), reason="Ollama server/model not available")


@pytest.fixture(scope="session")
def normalized_documents():
    return load_documents(config.RAW_CSV_PATH, source_name="cars_cleaned")


@pytest.fixture(scope="session")
def enriched_vehicles(normalized_documents):
    return VehicleEnrichment().enrich_batch(normalized_documents)


@pytest.fixture(scope="session")
def knowledge_base():
    """The real, full knowledge base (150 vehicles / 300 chunks), built
    once per test session. Not saved to disk -- keeps tests hermetic and
    independent of demo_phase1.py's persisted artifacts."""
    return build_knowledge_base(save=False)
