import pytest

from src.chunking.splitter import recursive_character_split
from src.chunking.vehicle_chunks import build_vehicle_chunks


def test_short_text_returned_unchanged():
    text = "short text"
    assert recursive_character_split(text, chunk_size=500, chunk_overlap=50) == [text]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        recursive_character_split("some text", chunk_size=10, chunk_overlap=10)


def test_long_text_splits_with_overlap_and_sentence_boundaries():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 20  # 940 chars, well over chunk_size below

    pieces = recursive_character_split(text, chunk_size=200, chunk_overlap=40)

    assert len(pieces) > 1
    for piece in pieces:
        # Ceiling is chunk_size + chunk_overlap, not chunk_size -- see
        # splitter.py docstring (an overlap-seeded chunk can exceed
        # chunk_size by up to chunk_overlap chars).
        assert len(piece) <= 200 + 40

    # Sentence-boundary awareness: every non-final piece ends on ". ".
    for piece in pieces[:-1]:
        assert piece.rstrip().endswith(".")

    # Overlap: each chunk (after the first) starts with exactly the
    # trailing chunk_overlap characters of the previous chunk.
    for prev, nxt in zip(pieces, pieces[1:]):
        assert nxt.startswith(prev[-40:])


def test_recursion_falls_back_to_hard_cut_with_no_separators():
    # A single "word" with no separators at all, longer than chunk_size.
    text = "x" * 250
    pieces = recursive_character_split(text, chunk_size=100, chunk_overlap=10, separators=[])
    assert len(pieces) >= 3
    for p in pieces:
        assert len(p) <= 100 + 10  # ceiling is chunk_size + chunk_overlap, see splitter.py docstring
        assert set(p) == {"x"}


def test_build_vehicle_chunks_porsche_macan(enriched_vehicles):
    macan = next(d for d in enriched_vehicles if d.metadata["name"] == "Porsche Macan")
    chunks = build_vehicle_chunks(macan)

    assert len(chunks) == 2
    assert {c.chunk_type for c in chunks} == {"product_overview", "features"}

    for c in chunks:
        assert c.chunk_id == f"cars_cleaned_0::{c.chunk_type}"
        assert "Porsche Macan" in c.text

    overview = next(c for c in chunks if c.chunk_type == "product_overview")
    assert "SUV" in overview.text


def test_all_real_chunks_no_literal_none_and_unique_ids(enriched_vehicles):
    all_chunks = []
    for doc in enriched_vehicles:
        all_chunks.extend(build_vehicle_chunks(doc))

    assert len(all_chunks) == 300
    chunk_ids = [c.chunk_id for c in all_chunks]
    assert len(set(chunk_ids)) == 300
    for c in all_chunks:
        assert "None" not in c.text
