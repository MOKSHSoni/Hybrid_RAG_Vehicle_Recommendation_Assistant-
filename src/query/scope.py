"""Scope checks: refuse, rather than invent, when a request is outside the catalogue.

Two distinct cases, both of which the pipeline previously answered with
unrelated vehicles:

1. Off-topic -- "best clothing store". Nothing about a vehicle was asked,
   yet semantic fallback always returns *something*, so three random cars
   came back labelled as recommendations.

2. On-topic but absent -- "Tesla Model 3". A real vehicle question about a
   brand this dataset does not carry. Extraction kept brand="Tesla", the
   filter matched nothing, and relaxation then DROPPED the brand, so the
   user was shown non-Tesla cars with the one thing they asked for
   silently discarded.

Why vocabulary and not a relevance score: the obvious signal is the
cross-encoder's top score, and it was measured and does not work. The
chunks are spec sheets while queries are questions, so genuine vague car
requests match them poorly -- "something fun to drive on weekends" scored
-10.73, BELOW "hello" at -8.46 and level with "recipe for pasta" at
-10.76. No threshold separates those. Vehicle vocabulary does: against the
same query set it rejected 15 of 15 off-topic queries and accepted 12 of
13 car queries, the one miss being a follow-up ("what about cheaper ones")
that context resolution rewrites into a standalone query before this runs.

The off-topic test deliberately errs toward answering. A query is only
refused if it contains no vehicle vocabulary AND extraction found no
constraint at all -- so "something for my family of five" still gets
through on its extracted seating capacity. Wrongly turning a customer
away is the worse failure for a sales assistant.
"""

import re
from typing import FrozenSet, Iterable, List, Optional

from src.query.models import Constraints
from src.retrieval.chunk_store import ChunkStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that mark a request as being about a vehicle. Combined at runtime
# with every brand and model-name token in the dataset, so "is the thar
# good" is recognised without "thar" being listed here.
_GENERIC_VEHICLE_TERMS = frozenset({
    "car", "cars", "vehicle", "vehicles", "auto", "automobile", "suv", "suvs", "sedan", "sedans",
    "hatchback", "hatchbacks", "mpv", "muv", "coupe", "convertible", "drive", "driving", "driver",
    "ride", "mileage", "kmpl", "engine", "petrol", "diesel", "cng", "electric", "ev", "hybrid",
    "automatic", "manual", "transmission", "seater", "seats", "boot", "trunk", "cargo",
    "horsepower", "bhp", "torque", "speed", "sporty", "offroad", "highway", "lakh", "lakhs",
    "emi", "clearance", "wheel", "wheels", "variant", "brand",
})

# Model-name tokens this short are too generic to mark a query as vehicular
# ("go", "rs", "ix"), so they are left out of the vocabulary.
_MIN_MODEL_TOKEN_LEN = 3

OFF_TOPIC_MESSAGE = (
    "I can only help you choose a car from this catalogue, so I don't have anything useful to say "
    "about that. Tell me what you need in a vehicle -- a budget, a body type, how many seats -- and "
    "I'll find the closest matches."
)


def build_vehicle_vocabulary(chunk_store: ChunkStore, known_brands: Iterable[str]) -> FrozenSet[str]:
    vocabulary = set(_GENERIC_VEHICLE_TERMS)
    vocabulary.update(_tokens(" ".join(known_brands)))
    for chunk in chunk_store.all_chunks():
        name = chunk.metadata.get("name") or ""
        vocabulary.update(t for t in _tokens(name) if len(t) >= _MIN_MODEL_TOKEN_LEN)
    return frozenset(vocabulary)


def is_off_topic(query: str, constraints: Constraints, vocabulary: FrozenSet[str]) -> bool:
    if constraints.populated_fields() or constraints.superlative_field:
        return False  # extraction found something vehicular to act on
    return not (set(_tokens(query)) & vocabulary)


def match_known_brand(requested: Optional[str], known_brands: Iterable[str]) -> Optional[str]:
    """The catalogue's own spelling of a requested brand, or None if absent.

    Matched on normalised prefixes in both directions because the dataset
    stores single-token brands ("Mercedes", "Rolls") while the LLM extractor
    returns full names ("Mercedes-Benz", "Rolls Royce"). The metadata filter
    compares brand by exact string, so an unmatched spelling filtered out
    every vehicle -- and relaxation then dropped the brand entirely. Asking
    for "Rolls Royce" returned three Mercedes. When several brands match,
    the longest wins, so a short brand can never capture a longer one.
    """
    if not requested:
        return None
    wanted = _normalise(requested)
    if not wanted:
        return None
    matches = []
    for brand in known_brands:
        have = _normalise(brand)
        if have and (wanted.startswith(have) or have.startswith(wanted)):
            matches.append(brand)
    return max(matches, key=len) if matches else None


def unknown_brand(constraints: Constraints, known_brands: Iterable[str]) -> Optional[str]:
    """The requested brand, if the catalogue carries no vehicles from it."""
    known = list(known_brands)
    if not constraints.brand or not known:
        # No brand list means nothing to compare against, and refusing every
        # brand on a misconfigured pipeline would be far worse than skipping.
        return None
    return None if match_known_brand(constraints.brand, known) else constraints.brand


def unknown_brand_message(brand: str, known_brands: List[str]) -> str:
    return (
        f"This catalogue doesn't include any {brand} vehicles, so I can't recommend one without "
        f"making it up. The brands it does cover are: {', '.join(sorted(known_brands))}. "
        f"Would one of those work for you?"
    )


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalise(text: str) -> str:
    return "".join(_tokens(text))
