import numpy as np

import config
from src.embeddings.embedder import Embedder, normalize_embeddings


def test_normalize_embeddings_l2_norm_is_one():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-6)


def test_normalize_embeddings_guards_zero_rows():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    normalized = normalize_embeddings(vectors)
    assert not np.isnan(normalized).any()


def test_embedder_produces_expected_shape():
    embedder = Embedder()
    vectors = embedder.embed_texts(["a red SUV", "a small hatchback", "an electric sedan"])
    assert vectors.shape == (3, config.EMBEDDING_DIM)
    assert vectors.dtype == np.float32
