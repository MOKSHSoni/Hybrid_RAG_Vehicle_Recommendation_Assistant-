"""Phase 12: query understanding evaluation.

Expected Constraints (from the eval query set) vs Extracted Constraints
(Phase 3's real understand_query output), scored per-field.
"""

from collections import defaultdict
from typing import Any, Dict, List

from src.evaluation.dataset import EvalQuery
from src.query.models import Constraints
from src.query.understanding import understand_query

_SIMPLE_FIELDS = ["brand", "transmission", "seating_capacity", "body_type", "price_max_lakhs", "price_min_lakhs"]


def compare_constraints(extracted: Constraints, expected: Dict[str, Any]) -> Dict[str, bool]:
    """Field-by-field match, only for fields the eval entry actually specifies."""
    comparison = {}
    for field_name in _SIMPLE_FIELDS:
        if field_name in expected:
            comparison[field_name] = getattr(extracted, field_name) == expected[field_name]
    if "fuel_types" in expected:
        comparison["fuel_types"] = set(extracted.fuel_types) == set(expected["fuel_types"])
    return comparison


def evaluate_query_understanding(queries: List[EvalQuery], known_brands: List[str]) -> Dict[str, float]:
    """Per-field accuracy across every eval query that specifies that field."""
    correct = defaultdict(int)
    total = defaultdict(int)

    for q in queries:
        result = understand_query(q.query, q.history, known_brands)
        comparison = compare_constraints(result.constraints, q.expected_constraints)
        for field_name, is_correct in comparison.items():
            total[field_name] += 1
            if is_correct:
                correct[field_name] += 1

    return {field_name: correct[field_name] / total[field_name] for field_name in total if total[field_name] > 0}
