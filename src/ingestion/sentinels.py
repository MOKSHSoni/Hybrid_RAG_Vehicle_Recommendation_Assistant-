"""Generic missing-value detection.

Real-world structured sources use inconsistent sentinels for "missing"
(empty string, '-', 'N/A', ...). This is intentionally generic -- it has
no knowledge of which columns exist in any particular source.
"""

import math
from typing import Any, Optional, Set

import config

DEFAULT_MISSING_SENTINELS: Set[str] = config.MISSING_VALUE_SENTINELS


def is_missing(value: Any, sentinels: Optional[Set[str]] = None) -> bool:
    """Return True if `value` should be treated as missing/null."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    sentinels = sentinels if sentinels is not None else DEFAULT_MISSING_SENTINELS
    return str(value).strip().lower() in sentinels
