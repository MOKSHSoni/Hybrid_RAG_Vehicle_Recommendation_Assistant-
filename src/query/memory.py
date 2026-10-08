"""Conversational memory: carry earlier constraints into a follow-up.

A refinement like "only 7 seaters" after "mahindra" depends entirely on the
previous turn, and the system was losing that. The LLM rewriter was supposed
to fold the context in and measurably does not: asked to rewrite "only 7
seaters", "what about cheaper ones" and "show me diesel ones" against a real
conversation, it returned all three unchanged. That is not flakiness -- the
calls took 2.8-3.7s each, so the model ran and chose to echo. The prompt ends
with "If the latest message is already standalone, return it unchanged", and
under a JSON schema with thinking suppressed, echoing is the cheapest
compliant answer.

Dropping the schema is not an option: the same prompt without it timed out at
120 seconds for a one-line rewrite (see ollama_client.py's docstring on the
speed/depth trade-off).

So the carry-forward happens here, in Python, on the already-validated
Constraints objects. It is deterministic, instant, and works whether or not
the model cooperates. The prompt fix in context.py is a complement, not the
mechanism: it improves the retrieval *text*, which matters in fallback mode
where no constraint is verified, but correctness no longer depends on it.
"""

from dataclasses import replace
from typing import List, Optional, Tuple

from src.query.models import Constraints, ConversationTurn

# Fields a follow-up inherits when it does not mention them itself. Ordered
# so the "carried over" note reads naturally to the user.
_INHERITABLE = (
    "brand",
    "body_type",
    "fuel_types",
    "transmission",
    "seating_capacity",
    "price_max_lakhs",
    "price_min_lakhs",
)

_LABELS = {
    "brand": "brand",
    "body_type": "body type",
    "fuel_types": "fuel",
    "transmission": "transmission",
    "seating_capacity": "seats",
    "price_max_lakhs": "max price",
    "price_min_lakhs": "min price",
}


def latest_constraints(history: List[ConversationTurn]) -> Optional[Constraints]:
    """The most recent turn's validated constraints, if any."""
    for turn in reversed(history):
        if turn.constraints is not None:
            return turn.constraints
    return None


def carry_forward(
    current: Constraints, previous: Optional[Constraints]
) -> Tuple[Constraints, List[str]]:
    """Merge the previous turn's constraints into this one.

    The rule is deliberately simple, because a rule the user cannot predict
    is worse than no rule: anything this turn specifies wins, anything it
    leaves unset inherits. Saying a brand replaces the old brand; saying
    nothing about brand keeps it.

    There is no cleverness about detecting when the user "meant" to start
    over -- no scanning for "all" or "any" or "instead". That guesswork
    would be wrong some of the time and hard to explain when it was. The
    user controls this with an explicit toggle, and whatever is inherited
    is shown to them, so a wrong inheritance is visible rather than silent.

    Returns the merged constraints and a human-readable list of what was
    inherited, for display.
    """
    if previous is None:
        return current, []

    merged = current
    inherited: List[str] = []
    for name in _INHERITABLE:
        mine = getattr(current, name)
        theirs = getattr(previous, name)
        if _is_set(mine) or not _is_set(theirs):
            continue
        merged = replace(merged, **{name: theirs})
        inherited.append(f"{_LABELS[name]} = {_format(theirs)}")
    return merged, inherited


def _is_set(value) -> bool:
    # fuel_types is a list; every other inheritable field is Optional.
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _format(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)
