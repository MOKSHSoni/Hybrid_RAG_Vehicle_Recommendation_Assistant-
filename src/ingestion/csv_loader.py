"""Generic CSV -> NormalizedDocument loader.

Domain-agnostic: works for any structured CSV source. Cleans values
(sentinel detection, type coercion) but never renames or reinterprets
columns -- that is the enrichment layer's job.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.ingestion.models import NormalizedDocument
from src.ingestion.sentinels import is_missing


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with every cell as a raw string.

    dtype=str + keep_default_na=False means WE control sentinel
    interpretation via is_missing(), instead of pandas silently NaN-ing
    values inconsistently (e.g. pandas' default NA list already includes
    'NA', 'null', etc. but not '-', and would coerce numeric-looking
    columns to float64/NaN before we ever get a chance to inspect them).
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def dataframe_to_documents(df: pd.DataFrame, source_name: str) -> List[NormalizedDocument]:
    """Convert a loaded DataFrame into a list of NormalizedDocuments."""
    ingested_at = datetime.now(timezone.utc).isoformat()
    documents = []
    for idx, row in df.iterrows():
        documents.append(
            NormalizedDocument(
                doc_id=f"{source_name}_{idx}",
                raw_fields=_clean_row(row.to_dict()),
                source_metadata={
                    "source_file": source_name,
                    "source_row_index": int(idx),
                    "ingested_at": ingested_at,
                },
            )
        )
    return documents


def load_documents(path: Path, source_name: str) -> List[NormalizedDocument]:
    """Convenience wrapper: load a CSV straight into NormalizedDocuments."""
    df = load_csv(path)
    return dataframe_to_documents(df, source_name)


def _clean_row(raw: Dict[str, str]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in raw.items():
        value = value.strip() if isinstance(value, str) else value
        cleaned[key] = None if is_missing(value) else _coerce_type(value)
    return cleaned


def _coerce_type(value: str) -> Any:
    """Best-effort generic type coercion: bool -> numeric -> trimmed string.

    Compound/messy text (e.g. "251 BHP@5000 RPM", "1,18,950") safely falls
    through to str since float() raises on it.
    """
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        f = float(value)
    except ValueError:
        return value
    if f.is_integer() and "." not in value:
        return int(f)
    return f
