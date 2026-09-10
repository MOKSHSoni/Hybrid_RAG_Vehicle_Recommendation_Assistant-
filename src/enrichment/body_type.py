"""Heuristic Body_Type derivation for vehicles.

No Body_Type column exists in the source dataset, so this module derives
one reproducibly from `Name` (keyword match) with a seating/dimension
fallback for names that match no keyword.

Tier 1: keyword match against Name, checked in priority order (SUV before
MPV before Convertible before Coupe before Sedan) so that a more specific
pattern wins over a generic one -- e.g. "Mercedes Benz GLC Coupe" must
resolve to SUV (via the "GLC" SUV keyword), not Coupe (the generic
"Coupe" keyword), because SUV is checked first.

Tier 2: when no keyword matches, fall back to seating capacity + length +
ground clearance thresholds (see config.py).

Every rule below was verified against the actual 150 rows of
cars_cleaned.csv, not written blind. Two corrections were made from a
first draft during that verification:

- "Santro", "Grand i10", "i20" are NOT Sedan keywords, despite being
  plausible-looking model names next to genuine sedans like "Verna" --
  all three are real-world hatchbacks. Removing them lets them resolve
  correctly via the Tier 2 length fallback (all three have
  Length < BODY_TYPE_HATCHBACK_MAX_LENGTH_MM).
- The Tier 2 fallback for 5-seaters (and unknown-seating rows) checks
  LENGTH before GROUND CLEARANCE, not the other way around. Checking
  ground clearance first would misclassify small Indian-market hatchbacks
  with SUV-ish ground clearance (Renault Kwid GC=184mm, Datsun Go
  GC=180mm, Datsun redi-GO GC=187mm -- none of them SUVs) as SUV. Checked
  against every 5-seat row with ground_clearance_mm >= 180 in the real
  dataset: all but those three already resolve via a Tier 1 keyword
  before ever reaching the fallback, so length-first fixes the three
  false positives with no risk of misclassifying a real SUV.

Known accepted limitation: "BMW 8 Series" has no distinguishing keyword
of its own body style and matches the generic "Series" keyword, so it
resolves to Sedan -- even though the real-world 8 Series is a coupe.
Documented here (and in test_enrichment.py) rather than hidden.
"""

import re
from typing import List, Optional, Tuple

import config

# (body_type, [regex patterns checked case-insensitively against Name])
# Priority = list order. First tier with any matching pattern wins.
_KEYWORD_TIERS: List[Tuple[str, List[str]]] = [
    (
        "SUV",
        [
            r"\bX[1-9]\b",  # BMW X1/X3/X4/X5/X6/X7 (incl. "X3 M", "X5 M")
            r"\bQ[2-8]\b",  # Audi Q2/Q8/RS Q8
            r"\bGL[CES]\b",  # Mercedes GLC/GLE/GLS
            r"\bG Class\b",
            r"\bEQC\b",
            r"\bMacan\b",
            r"\bCayenne\b",
            r"\bXC\d{2}\b",  # Volvo XC40/XC60/XC90
            r"\bCross Country\b",
            r"\bNX\b",
            r"\bRX\b",
            r"\bLX\b",  # Lexus
            r"\bF-Pace\b",
            r"\bI-Pace\b",  # Jaguar
            r"\bVenue\b",
            r"\bCreta\b",
            r"\bTucson\b",
            r"\bKona\b",  # Hyundai
            r"\bSonet\b",
            r"\bSeltos\b",  # Kia
            r"\bHector\b",
            r"\bGloster\b",
            r"\bZS EV\b",  # MG
            r"\bNexon\b",
            r"\bHarrier\b",
            r"\bSafari\b",  # Tata
            r"\bXUV\b",
            r"\bScorpio\b",
            r"\bThar\b",
            r"\bBolero\b",
            r"\bAlturas\b",
            r"\bKUV100\b",  # Mahindra
            r"\bCompass\b",
            r"\bWrangler\b",  # Jeep
            r"\bDuster\b",
            r"\bKiger\b",  # Renault
            r"\bKicks\b",
            r"\bMagnite\b",  # Nissan
            r"\bT-Roc\b",
            r"\bTiguan\b",  # Volkswagen
            r"\bAircross\b",  # Citroen
            r"\bWR-V\b",  # Honda
            r"\bUrban Cruiser\b",
            r"\bFortuner\b",  # Toyota
            r"\bUrus\b",  # Lamborghini
            r"\bBentayga\b",  # Bentley
            r"\bLevante\b",  # Maserati
            r"\bCullinan\b",  # Rolls Royce
        ],
    ),
    (
        "MPV",
        [
            r"\bInnova\b",
            r"\bVellfire\b",  # Toyota
            r"\bCarnival\b",  # Kia
            r"\bV-Class\b",  # Mercedes
            r"\bMarazzo\b",  # Mahindra
            r"\bTriber\b",  # Renault
            r"\bGO Plus\b",  # Datsun
        ],
    ),
    (
        "Convertible",
        [
            r"\bConvertible\b",
            r"\bCabrio\b",
            r"\bRoadster\b",
            r"\bSpider\b",
            r"\bSpyder\b",
            r"\bZ4\b",  # BMW
            r"\bDawn\b",  # Rolls Royce
            r"\bPortofino\b",  # Ferrari
        ],
    ),
    (
        "Coupe",
        [
            r"\bCoupe\b",
            r"\bSportback\b",
            r"\bCLS\b",  # Mercedes
            r"\bWraith\b",  # Rolls Royce
            r"\bGranTurismo\b",  # Maserati
            r"\bHuracan\b",  # Lamborghini
            r"\bGT-R\b",  # Nissan
            r"\bAMG GT\b",  # Mercedes
            r"\bTributo\b",  # Ferrari
            r"\b812\b",  # Ferrari
            r"\bLusso\b",  # Ferrari
        ],
    ),
    (
        "Sedan",
        [
            r"\bClass\b",  # Mercedes *-Class (Coupe-named ones already matched above)
            r"\bSeries\b",  # BMW *-Series (ditto)
            r"\bA4\b",
            r"\bA6\b",
            r"\bA8\b",  # Audi
            r"\bPanamera\b",  # Porsche
            r"\bXE\b",
            r"\bXF\b",  # Jaguar
            r"\bS60\b",
            r"\bS90\b",  # Volvo
            r"\bES\b",
            r"\bLS\b",  # Lexus
            r"\bGhibli\b",
            r"\bQuattroporte\b",  # Maserati
            r"\bPhantom\b",  # Rolls Royce
            r"\bCity\b",
            r"\bAmaze\b",  # Honda
            r"\bVerna\b",
            r"\bAura\b",
            r"\bElantra\b",  # Hyundai
            r"\bCamry\b",
            r"\bYaris\b",  # Toyota
            r"\bVento\b",  # Volkswagen
            r"\bTigor\b",  # Tata
        ],
    ),
]


def derive_body_type(
    name: str,
    seating: Optional[int] = None,
    length_mm: Optional[float] = None,
    ground_clearance_mm: Optional[float] = None,
) -> str:
    """Derive a body type (SUV/Sedan/Hatchback/MPV/Coupe/Convertible) for a vehicle."""
    name = name or ""
    for body_type, patterns in _KEYWORD_TIERS:
        for pattern in patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return body_type
    return _fallback_body_type(seating, length_mm, ground_clearance_mm)


def _fallback_body_type(
    seating: Optional[int],
    length_mm: Optional[float],
    ground_clearance_mm: Optional[float],
) -> str:
    length = length_mm or 0
    ground_clearance = ground_clearance_mm or 0

    if seating is not None and seating <= 2:
        return "Coupe"
    if seating == 4:
        return "Hatchback" if length < config.BODY_TYPE_COUPE_MIN_LENGTH_MM else "Coupe"
    if seating is not None and seating >= 6:
        if ground_clearance >= config.BODY_TYPE_SUV_GROUND_CLEARANCE_MM:
            return "SUV"
        if length >= config.BODY_TYPE_SUV_LENGTH_MM:
            return "SUV"
        return "MPV"

    # seating == 5, or seating unknown ('-'/empty sentinel rows).
    # Length checked BEFORE ground clearance -- see module docstring.
    if length and length < config.BODY_TYPE_HATCHBACK_MAX_LENGTH_MM:
        return "Hatchback"
    if ground_clearance >= config.BODY_TYPE_SUV_GROUND_CLEARANCE_MM:
        return "SUV"
    return "Sedan"
