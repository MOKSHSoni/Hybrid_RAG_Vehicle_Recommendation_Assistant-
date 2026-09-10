from unittest.mock import patch

from conftest import requires_ollama

from src.query.hyde import generate_hypothetical_description
from src.query.ollama_client import OllamaError
from src.retrieval.dense.hyde_retriever import HydeRetriever
from src.retrieval.dense.retriever import DenseRetriever


def test_generate_hypothetical_description_returns_none_on_ollama_error():
    with patch("src.query.hyde.chat_text", side_effect=OllamaError("simulated")):
        assert generate_hypothetical_description("affordable SUV") is None


def test_generate_hypothetical_description_strips_whitespace():
    with patch("src.query.hyde.chat_text", return_value="  A spacious SUV.  "):
        assert generate_hypothetical_description("affordable SUV") == "A spacious SUV."


def test_hyde_retriever_falls_back_to_raw_query_on_llm_failure(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    hyde = HydeRetriever(dense)

    with patch("src.retrieval.dense.hyde_retriever.generate_hypothetical_description", return_value=None):
        results = hyde.retrieve("Porsche Cayenne price and top speed", top_k=5)

    assert len(results) == 5
    assert all(r.method == "hyde" for r in results)
    assert "Cayenne" in results[0].chunk.metadata["name"]


def test_hyde_retriever_never_exposes_hypothetical_text(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    hyde = HydeRetriever(dense)

    fake_hypothetical = "THIS-IS-SYNTHETIC-AND-SHOULD-NEVER-APPEAR-IN-RESULTS"
    with patch("src.retrieval.dense.hyde_retriever.generate_hypothetical_description", return_value=fake_hypothetical):
        results = hyde.retrieve("affordable SUV", top_k=5)

    for r in results:
        assert fake_hypothetical not in r.chunk.text
        assert r.chunk.chunk_id.startswith("cars_cleaned_")  # a real chunk, not synthetic


@requires_ollama
def test_generate_hypothetical_description_real_call_is_listing_shaped():
    text = generate_hypothetical_description("affordable 7 seater SUV with good mileage")
    assert text is not None
    assert len(text) > 20


@requires_ollama
def test_hyde_retriever_real_call(knowledge_base):
    dense = DenseRetriever(knowledge_base.faiss_index, knowledge_base.chunk_store, knowledge_base.embedder)
    hyde = HydeRetriever(dense)
    results = hyde.retrieve("Porsche Cayenne price and top speed", top_k=5)
    assert len(results) == 5
    assert all(r.method == "hyde" for r in results)
