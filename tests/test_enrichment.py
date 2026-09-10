import pytest

from src.enrichment.base import BaseEnrichment
from src.enrichment.body_type import derive_body_type
from src.enrichment.vehicle import VehicleEnrichment, _parse_fuel_types


@pytest.mark.parametrize(
    "name,seating,length_mm,ground_clearance_mm,expected",
    [
        ("Hyundai Creta", 5, 4300, 190, "SUV"),
        ("BMW X5", 5, 4922, None, "SUV"),
        ("Toyota Innova Crysta", 7, 4735, 176, "MPV"),
        ("Honda City", 5, 4440, 165, "Sedan"),
        ("Renault Kwid", 5, 3731, 184, "Hatchback"),  # high GC (184mm) must NOT trigger SUV
        ("Datsun redi-GO", 5, 3435, 187, "Hatchback"),  # same high-GC-hatchback trap
        ("Ferrari F8 Tributo", 2, 4611, None, "Coupe"),
        ("BMW Z4", 2, 4324, None, "Convertible"),
        ("Mercedes Benz CLS", None, 4988, None, "Coupe"),  # missing seating must not crash
        ("Tata Nexon EV", None, 3993, 209, "SUV"),  # missing seating must not crash
    ],
)
def test_derive_body_type(name, seating, length_mm, ground_clearance_mm, expected):
    assert derive_body_type(name, seating, length_mm, ground_clearance_mm) == expected


def test_known_heuristic_limitation_bmw_8_series():
    """BMW 8 Series is a real-world coupe, but has no distinguishing
    keyword of its own and matches the generic "Series" pattern, so the
    heuristic classifies it as Sedan. Accepted, documented limitation."""
    result = derive_body_type("BMW 8 Series", seating=5, length_mm=5082, ground_clearance_mm=None)
    assert result == "Sedan"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Petrol", ["Petrol"]),
        ("Petrol, Diesel", ["Petrol", "Diesel"]),
        ("Diesel, Petrol", ["Diesel", "Petrol"]),
        ("Petrol, CNG, Diesel", ["Petrol", "CNG", "Diesel"]),
        ("Electric", ["Electric"]),
        (None, []),
    ],
)
def test_parse_fuel_types(raw, expected):
    assert _parse_fuel_types(raw) == expected


def test_vehicle_enrichment_is_base_enrichment_subclass():
    assert issubclass(VehicleEnrichment, BaseEnrichment)


def test_full_enrichment_run_no_exceptions(enriched_vehicles):
    assert len(enriched_vehicles) == 150
    for doc in enriched_vehicles:
        assert doc.metadata["body_type"]
        assert doc.metadata["brand"]
        assert isinstance(doc.metadata["fuel_types"], list)
        assert len(doc.metadata["fuel_types"]) > 0
        assert doc.metadata["price_lakhs"] > 0


def test_electric_vehicle_count(enriched_vehicles):
    ev = [d for d in enriched_vehicles if d.metadata["is_electric"]]
    assert len(ev) == 6


def test_missing_seating_handled_safely(enriched_vehicles):
    unknown_seating = [d for d in enriched_vehicles if d.metadata["seating_capacity"] is None]
    assert len(unknown_seating) == 2
    for d in unknown_seating:
        assert d.metadata["seating_capacity"] != 0
