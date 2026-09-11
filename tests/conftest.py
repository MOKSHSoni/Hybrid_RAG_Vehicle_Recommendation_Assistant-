import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import config
from src.enrichment.vehicle import VehicleEnrichment
from src.ingestion.csv_loader import load_documents
from src.pipeline import build_knowledge_base
from src.query.ollama_client import OllamaError, chat_text


# Generous enough to absorb a cold model load (~70s measured), so tests
# skip only when Ollama is genuinely unavailable.
_OLLAMA_PROBE_TIMEOUT_SECONDS = 150


def _ollama_available() -> bool:
    """Probe Ollama, and in doing so warm the model for the whole session.

    The timeout is deliberately generous. It was 10s, which quietly broke
    the suite: loading qwen3:4b from cold was measured at 70s on this
    hardware, so whenever the model had been evicted the probe failed and
    EVERY live-LLM test silently skipped. A silent skip is worse than a
    failure -- it looks like coverage while providing none.

    Routed through chat_text (schema-wrapped) rather than a raw chat()
    call, for the thinking-suppression reason in ollama_client.py.
    """
    try:
        chat_text(
            messages=[{"role": "user", "content": "reply with the single word: ok"}],
            timeout=_OLLAMA_PROBE_TIMEOUT_SECONDS,
        )
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
