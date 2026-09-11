"""Phase 12: evaluation query dataset loader.

Loads a versioned evaluation/queries_vN.json file (currently
evaluation/queries_v2.json, per config.EVAL_QUERIES_PATH -- a draft
written by the coding agent, to be reviewed/refined by a human before
being treated as ground truth) into typed EvalQuery objects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import config
from src.query.models import ConversationTurn


@dataclass
class EvalQuery:
    id: str
    category: str
    query: str
    history: List[ConversationTurn]
    expected_constraints: Dict[str, Any]
    expected_relevant_vehicles: List[str]
    expected_mode: str
    notes: str = ""


def load_eval_queries(path: Path = config.EVAL_QUERIES_PATH) -> List[EvalQuery]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for raw in data["queries"]:
        history = [ConversationTurn(role=turn["role"], content=turn["content"]) for turn in raw.get("history", [])]
        queries.append(
            EvalQuery(
                id=raw["id"],
                category=raw["category"],
                query=raw["query"],
                history=history,
                expected_constraints=raw.get("expected_constraints", {}),
                expected_relevant_vehicles=raw.get("expected_relevant_vehicles", []),
                expected_mode=raw.get("expected_mode", "unknown"),
                notes=raw.get("notes", ""),
            )
        )
    return queries
