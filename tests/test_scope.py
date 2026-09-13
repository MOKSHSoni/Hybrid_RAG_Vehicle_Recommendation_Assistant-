import pytest

from src.query.models import Constraints
from src.query.regex_extraction import known_brands_from_chunk_store
from src.query.scope import (
    build_vehicle_vocabulary,
    is_off_topic,
    match_known_brand,
    unknown_brand,
    unknown_brand_message,
)
from src.query.understanding import _canonicalise_brand


@pytest.fixture(scope="module")
def brands(knowledge_base):
    return known_brands_from_chunk_store(knowledge_base.chunk_store)


@pytest.fixture(scope="module")
def vocabulary(knowledge_base, brands):
    return build_vehicle_vocabulary(knowledge_base.chunk_store, brands)


# These two lists are the query set the vocabulary approach was chosen on.
# A cross-encoder relevance threshold was measured first and could not
# separate them: "something fun to drive on weekends" scored below "hello".
@pytest.mark.parametrize(
    "query",
    [
        "best clothing store",
        "what's the weather today",
        "recipe for pasta",
        "who won the cricket match",
        "best smartphone under 20000",
        "how to learn python",
        "hello",
        "tell me a joke",
        "book a flight to delhi",
        "best laptop for gaming",
        "nearest hospital",
        "buy shoes online",
        "what is the capital of france",
        "good restaurants nearby",
        "movie recommendations",
    ],
)
def test_off_topic_queries_are_refused(query, vocabulary):
    assert is_off_topic(query, Constraints(), vocabulary) is True


@pytest.mark.parametrize(
    "query",
    [
        "something fun to drive on weekends",
        "family car with good mileage",
        "a car for long highway trips",
        "good car for a new driver",
        "comfortable car for my parents",
        "something sporty",
        "car with big boot",
        "which car is best for city driving",
        "i want a reliable vehicle",
        "Porsche Cayenne price",
        "tell me about the creta",
        "is the thar good",
    ],
)
def test_vague_vehicle_queries_are_not_refused(query, vocabulary):
    # The costly failure: turning a genuine customer away.
    assert is_off_topic(query, Constraints(), vocabulary) is False


def test_an_extracted_constraint_keeps_a_query_in_scope(vocabulary):
    # "family of five" has no vehicle word, but extraction found a seat
    # count -- which is itself evidence the request is about a vehicle.
    assert is_off_topic("something for my family of five", Constraints(seating_capacity=5), vocabulary) is False


def test_brand_outside_catalogue_is_reported(brands):
    assert unknown_brand(Constraints(brand="Tesla"), brands) == "Tesla"
    assert unknown_brand(Constraints(brand="Ford"), brands) == "Ford"


@pytest.mark.parametrize("requested", ["BMW", "bmw", "Mercedes-Benz", "Rolls-Royce", "Tata Motors", "MG Motor"])
def test_brand_in_catalogue_is_not_refused(requested, brands):
    # The dataset stores single-token brands ("Mercedes", "Rolls"), while
    # extraction may return the full name. Strict equality would tell a
    # customer the catalogue has no Mercedes -- a false refusal.
    assert unknown_brand(Constraints(brand=requested), brands) is None


def test_no_brand_requested_is_not_a_refusal(brands):
    assert unknown_brand(Constraints(), brands) is None


def test_empty_brand_list_never_refuses():
    # A misconfigured pipeline must not refuse every brand.
    assert unknown_brand(Constraints(brand="BMW"), []) is None


def test_unknown_brand_message_names_the_brand_and_the_alternatives(brands):
    message = unknown_brand_message("Tesla", brands)
    assert "Tesla" in message
    assert "BMW" in message and "Porsche" in message


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("Rolls Royce", "Rolls"),
        ("Rolls-Royce", "Rolls"),
        ("Mercedes-Benz", "Mercedes"),
        ("bmw", "BMW"),
        ("Tata Motors", "Tata"),
        ("Mini Cooper", "MINI"),
        ("Tesla", None),
    ],
)
def test_match_known_brand_returns_the_catalogue_spelling(requested, expected, brands):
    assert match_known_brand(requested, brands) == expected


def test_extracted_brand_is_canonicalised_so_the_filter_matches(brands, knowledge_base):
    # Regression: the LLM extractor returns "Rolls Royce" where the dataset
    # stores "Rolls". The metadata filter compares brand by exact string, so
    # nothing matched, relaxation dropped the brand, and a request for
    # Rolls-Royce came back as three Mercedes.
    from src.query.metadata_filter import filter_vehicles, unique_vehicles

    vehicles = unique_vehicles(knowledge_base.chunk_store)
    assert filter_vehicles(vehicles, Constraints(brand="Rolls Royce")) == []  # the original failure

    constraints = _canonicalise_brand(Constraints(brand="Rolls Royce"), brands)
    assert constraints.brand == "Rolls"
    names = {v.get("name") for v in filter_vehicles(vehicles, constraints)}
    assert names and all(name.startswith("Rolls Royce") for name in names)


def test_brand_outside_catalogue_survives_canonicalisation(brands):
    # Left untouched, not cleared: the scope check needs the name to say
    # "no Tesla here" rather than quietly showing some other make.
    assert _canonicalise_brand(Constraints(brand="Tesla"), brands).brand == "Tesla"
