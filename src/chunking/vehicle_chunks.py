"""Domain-specific (vehicle) chunk text templates.

Uses only the clean, precomputed numeric metadata fields (never noisy raw
text like "251 BHP@5000 RPM" or "1,18,950") except for Peak Power / Peak
Torque, which are compound text with no clean numeric equivalent and are
read straight from raw_fields. Every value round-trips through a
missing-value guard that renders "N/A" -- never Python's "None" -- and
every float is rounded to 2 decimals before interpolation (the source
data has float artifacts from Crore->Lakh conversion, e.g.
Price_Max_Lakhs = 243.00000000000003 for one Porsche Panamera trim).
"""

from typing import Any, Dict, List

import config
from src.chunking.models import Chunk
from src.chunking.splitter import recursive_character_split
from src.enrichment.models import EnrichedDocument


def build_vehicle_chunks(enriched: EnrichedDocument) -> List[Chunk]:
    """Build the product_overview and features chunks for one vehicle."""
    overview_text = build_product_overview(enriched.metadata)
    features_text = build_features(enriched.metadata, enriched.raw_fields)

    chunks: List[Chunk] = []
    for chunk_type, text in (
        ("product_overview", overview_text),
        ("features", features_text),
    ):
        pieces = recursive_character_split(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            chunk_id = (
                f"{enriched.doc_id}::{chunk_type}"
                if len(pieces) == 1
                else f"{enriched.doc_id}::{chunk_type}::part{i}"
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=enriched.doc_id,
                    chunk_type=chunk_type,
                    text=piece,
                    metadata=dict(enriched.metadata),
                )
            )
    return chunks


def build_product_overview(metadata: Dict[str, Any]) -> str:
    name = metadata.get("name") or "This vehicle"
    body_type = metadata.get("body_type") or "vehicle"
    brand = metadata.get("brand") or "an unknown brand"
    price = _fmt_num(metadata.get("price_lakhs"))
    price_min = _fmt_num(metadata.get("price_min_lakhs"))
    price_max = _fmt_num(metadata.get("price_max_lakhs"))

    seating = metadata.get("seating_capacity")
    seating_text = f"{seating} people" if seating is not None else "an unspecified number of people"

    fuel_types = metadata.get("fuel_types") or []
    fuel_text = ", ".join(fuel_types) if fuel_types else "N/A"

    transmission_options = []
    if metadata.get("has_automatic"):
        transmission_options.append("Automatic")
    if metadata.get("has_manual"):
        transmission_options.append("Manual")
    transmission_text = " and ".join(transmission_options) if transmission_options else "N/A"

    sentences = [
        f"{name} is a {body_type} manufactured by {brand}, priced at approximately "
        f"Rs {price} Lakh (range: Rs {price_min}-{price_max} Lakh).",
        f"It seats {seating_text} and runs on {fuel_text}.",
        f"Transmission: {transmission_text}.",
    ]

    if not metadata.get("is_electric") and metadata.get("mileage_min_kmpl") is not None:
        mileage_min = _fmt_num(metadata.get("mileage_min_kmpl"))
        mileage_max = _fmt_num(metadata.get("mileage_max_kmpl"))
        sentences.append(f"Fuel efficiency ranges from {mileage_min} to {mileage_max} kmpl.")

    emi = metadata.get("emi_rupees")
    emi_text = f"Rs {emi:,.0f}/month" if emi is not None else "N/A"
    sentences.append(f"Estimated EMI starts at {emi_text}.")

    return " ".join(sentences)


def build_features(metadata: Dict[str, Any], raw_fields: Dict[str, Any]) -> str:
    name = metadata.get("name") or "This vehicle"

    if metadata.get("is_electric"):
        powertrain = "Electric powertrain (no combustion engine)."
    else:
        engine_min = metadata.get("engine_min_cc")
        engine_max = metadata.get("engine_max_cc")
        if engine_min is None or engine_max is None:
            powertrain = "Engine displacement: N/A."
        elif engine_min == engine_max:
            powertrain = f"Engine displacement: {_fmt_num(engine_min)} cc."
        else:
            powertrain = f"Engine displacement: {_fmt_num(engine_min)} to {_fmt_num(engine_max)} cc."

    peak_power = raw_fields.get("Peak Power") or "N/A"
    peak_torque = raw_fields.get("Peak Torque") or "N/A"

    top_speed = _fmt_num(metadata.get("top_speed_kmph"))
    top_speed_text = f"{top_speed} km/h" if top_speed != "N/A" else "N/A"

    length = _fmt_num(metadata.get("length_mm"))
    width = _fmt_num(metadata.get("width_mm"))
    height = _fmt_num(metadata.get("height_mm"))
    wheelbase = _fmt_num(metadata.get("wheelbase_mm"))
    ground_clearance = _fmt_num(metadata.get("ground_clearance_mm"))

    boot_space = _fmt_num(metadata.get("boot_space_l"))
    boot_space_text = f"{boot_space} L" if boot_space != "N/A" else "N/A"
    fuel_capacity = _fmt_num(metadata.get("fuel_capacity_l"))
    fuel_capacity_text = f"{fuel_capacity} L" if fuel_capacity != "N/A" else "N/A"
    turning_radius = _fmt_num(metadata.get("turning_radius_m"))
    turning_radius_text = f"{turning_radius} m" if turning_radius != "N/A" else "N/A"

    colors = metadata.get("colors") or []
    color_variants = metadata.get("color_variants_count")
    if colors:
        preview = colors[: config.FEATURES_COLORS_PREVIEW_COUNT]
        remaining = len(colors) - len(preview)
        colors_text = ", ".join(preview)
        if remaining > 0:
            colors_text += f", and {remaining} more"
    else:
        colors_text = "N/A"
    variants_text = f" ({color_variants} variants total)" if color_variants is not None else ""

    sentences = [
        f"{name} features: {powertrain}",
        f"Peak power: {peak_power}.",
        f"Peak torque: {peak_torque}.",
        f"Top speed: {top_speed_text}.",
        f"Dimensions: Length {length}mm, Width {width}mm, Height {height}mm, "
        f"Wheelbase {wheelbase}mm, Ground clearance {ground_clearance}mm.",
        f"Boot space: {boot_space_text}.",
        f"Fuel tank capacity: {fuel_capacity_text}.",
        f"Turning radius: {turning_radius_text}.",
        f"Available colors: {colors_text}{variants_text}.",
    ]
    return " ".join(sentences)


def _fmt_num(value: Any) -> str:
    if value is None:
        return "N/A"
    rounded = round(float(value), 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"
