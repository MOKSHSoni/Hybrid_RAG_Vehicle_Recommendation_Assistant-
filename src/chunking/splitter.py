"""Generic recursive/hierarchical text splitter.

No LangChain per the project's framework constraint -- this is a small,
plain-Python, sentence-boundary-aware splitter used by any future domain,
not just vehicles.

Algorithm: split on the coarsest separator in `separators` that keeps
pieces within chunk_size, recursing to the next-finer separator for any
piece still too long. Pieces are then greedily packed into chunks up to
chunk_size, with each new chunk seeded by the trailing chunk_overlap
characters of the previous one for context continuity. Falls back to a
hard character cut only if a single token exceeds chunk_size with no
separator left to split on.

Note: because of the overlap seed, an individual output chunk can exceed
chunk_size by up to chunk_overlap characters (e.g. an already-chunk_size
piece prefixed with an overlap tail) -- the hard ceiling is
chunk_size + chunk_overlap, not chunk_size. This is standard behavior for
this style of splitter and is asserted precisely (not just "small") in
test_chunking.py.
"""

from typing import List, Optional

import config

DEFAULT_SEPARATORS = list(config.CHUNK_SEPARATORS)


def recursive_character_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    separators = list(separators) if separators is not None else DEFAULT_SEPARATORS
    pieces = _split_recursive(text, chunk_size, separators)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _split_recursive(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    parts = text.split(sep) if sep else list(text)

    pieces = []
    for i, part in enumerate(parts):
        # Re-attach the separator (except after the last part) so pieces
        # concatenate back to the original text and punctuation stays
        # with the sentence it ends.
        fragment = part + sep if i < len(parts) - 1 else part
        if not fragment:
            continue
        if len(fragment) > chunk_size:
            pieces.extend(_split_recursive(fragment, chunk_size, rest_separators))
        else:
            pieces.append(fragment)
    return pieces


def _merge_with_overlap(pieces: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    if not pieces:
        return []

    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            overlap_tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = overlap_tail + piece
        else:
            current += piece
    if current:
        chunks.append(current)
    return chunks
