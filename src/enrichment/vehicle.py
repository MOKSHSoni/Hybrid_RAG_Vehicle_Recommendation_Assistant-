"""Vehicle-specific enrichment.

Reads columns defensively via .get() (never []) so this degrades
gracefully if a future CSV variant is missing a column, per the
"inspect available columns at runtime" requirement.
"""

import re
from typing import Any, List, Optional

from src.enrichment.base import BaseEnrichment
from src.enrichment.body_type import derive_body_type
from src.enrichment.models import EnrichedDocument
from src.ingestion.models import NormalizedDocument


class VehicleEnrichment(BaseEnrichment):
    def enrich(self, document: NormalizedDocument) -> EnrichedDocument:
        f = document.raw_fields
        fuel_types = _parse_fuel_types(f.get("FUEL TYPE"))
        seating = f.get("Seating Capacity")
        seating = seating if isinstance(seating, int) else None
        length_mm = _as_float(f.get("Length (mm)"))
        ground_clearance_mm = _as_float(f.get("Ground Clearance (mm)"))

        metadata = {
            "vehicle_id": document.doc_id,
            "name": f.get("Name"),
            "brand": f.get("Brand"),
            "body_type": derive_body_type(
                name=f.get("Name") or "",
                seating=seating,
                length_mm=length_mm,
                ground_clearance_mm=ground_clearance_mm,
            ),
            "price_lakhs": _as_float(f.get("Price_Lakhs")),
            "price_min_lakhs": _as_float(f.get("Price_Min_Lakhs")),
            "price_max_lakhs": _as_float(f.get("Price_Max_Lakhs")),
            "emi_rupees": _as_float(f.get("EMI_Rupees")),
            "fuel_types": fuel_types,
            "is_electric": fuel_types == ["Electric"],
            "has_automatic": bool(f.get("Has_Automatic")),
            "has_manual": bool(f.get("Has_Manual")),
            "seating_capacity": seating,
            "mileage_min_kmpl": _as_float(f.get("Mileage_Min_kmpl")),
            "mileage_max_kmpl": _as_float(f.get("Mileage_Max_kmpl")),
            "engine_min_cc": _as_float(f.get("Engine_Min_cc")),
            "engine_max_cc": _as_float(f.get("Engine_Max_cc")),
            "top_speed_kmph": _parse_leading_number(f.get("Top_Speed")),
            "boot_space_l": _as_float(f.get("Boot Space (L)")),
            "ground_clearance_mm": ground_clearance_mm,
            "length_mm": length_mm,
            "width_mm": _as_float(f.get("Width (mm)")),
            "height_mm": _as_float(f.get("Height (mm)")),
            "wheelbase_mm": _as_float(f.get("Wheelbase (mm)")),
            "turning_radius_m": _as_float(f.get("Turning Radius (m)")),
            "fuel_capacity_l": _as_float(f.get("Fuel Capacity (L)")),
            "color_variants_count": _as_int(f.get("Color Varients")),
            "colors": _parse_list(f.get("Colors")),
        }
        return EnrichedDocument(
            doc_id=document.doc_id,
            raw_fields=f,
            source_metadata=document.source_metadata,
            metadata=metadata,
        )


def _parse_fuel_types(raw: Any) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_leading_number(raw: Any) -> Optional[float]:
    """e.g. "254 Km/h" / "275km/h" -> 254.0 / 275.0."""
    if raw is None:
        return None
    match = re.search(r"[\d.]+", str(raw))
    return float(match.group()) if match else None


def _as_float(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def _as_int(value: Any) -> Optional[int]:
    return int(value) if value is not None else None
