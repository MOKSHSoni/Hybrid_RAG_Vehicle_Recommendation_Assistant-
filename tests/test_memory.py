"""Conversational memory. Pure Python -- no Ollama needed, which is the
point: the LLM rewriter was measured returning follow-ups unchanged, so
correctness must not depend on it."""

import pytest

from src.query.memory import carry_forward, latest_constraints
from src.query.models import Constraints, ConversationTurn


def test_unspecified_fields_are_inherited():
    # "mahindra" then "only 7 seaters" -- the real case that was losing the
    # brand and returning BMW, Volvo and Volkswagen 7-seaters.
    merged, carried = carry_forward(Constraints(seating_capacity=7), Constraints(brand="Mahindra"))
    assert merged.brand == "Mahindra"
    assert merged.seating_capacity == 7
    assert carried == ["brand = Mahindra"]


def test_the_new_turn_always_wins():
    merged, carried = carry_forward(
        Constraints(brand="Toyota"), Constraints(brand="Mahindra", seating_capacity=7)
    )
    assert merged.brand == "Toyota"
    assert merged.seating_capacity == 7  # untouched by the new turn, so inherited
    assert carried == ["seats = 7"]


def test_nothing_inherited_without_history():
    merged, carried = carry_forward(Constraints(seating_capacity=7), None)
    assert merged.brand is None
    assert carried == []


def test_empty_list_counts_as_unset_not_as_a_value():
    # fuel_types defaults to [] rather than None, so a naive "is not None"
    # check would treat an empty list as a deliberate choice and block
    # inheritance for every follow-up.
    merged, carried = carry_forward(Constraints(), Constraints(fuel_types=["Diesel"]))
    assert merged.fuel_types == ["Diesel"]
    assert carried == ["fuel = Diesel"]


def test_stated_fuel_is_not_overwritten():
    merged, _ = carry_forward(Constraints(fuel_types=["Petrol"]), Constraints(fuel_types=["Diesel"]))
    assert merged.fuel_types == ["Petrol"]


@pytest.mark.parametrize(
    "field,value,label",
    [
        ("brand", "Tata", "brand = Tata"),
        ("body_type", "SUV", "body type = SUV"),
        ("transmission", "Automatic", "transmission = Automatic"),
        ("seating_capacity", 7, "seats = 7"),
        ("price_max_lakhs", 20.0, "max price = 20.0"),
        ("price_min_lakhs", 5.0, "min price = 5.0"),
    ],
)
def test_every_inheritable_field_is_reported_for_display(field, value, label):
    # Whatever is inherited is shown to the user, so the list must name it.
    _, carried = carry_forward(Constraints(), Constraints(**{field: value}))
    assert carried == [label]


def test_superlative_is_not_inherited():
    # A sort order belongs to the question that asked for it. Inheriting
    # "cheapest" would silently re-sort every later follow-up.
    merged, carried = carry_forward(
        Constraints(seating_capacity=7),
        Constraints(superlative_field="price_lakhs", superlative_direction="asc"),
    )
    assert merged.superlative_field is None
    assert carried == []


def test_latest_constraints_walks_back_past_turns_without_any():
    history = [
        ConversationTurn(role="user", content="mahindra", constraints=Constraints(brand="Mahindra")),
        ConversationTurn(role="assistant", content="Here are three Mahindras."),
    ]
    assert latest_constraints(history).brand == "Mahindra"


def test_latest_constraints_prefers_the_most_recent():
    history = [
        ConversationTurn(role="user", content="mahindra", constraints=Constraints(brand="Mahindra")),
        ConversationTurn(role="assistant", content="..."),
        ConversationTurn(role="user", content="toyota", constraints=Constraints(brand="Toyota")),
        ConversationTurn(role="assistant", content="..."),
    ]
    assert latest_constraints(history).brand == "Toyota"


def test_latest_constraints_is_none_for_an_empty_history():
    assert latest_constraints([]) is None
    assert latest_constraints([ConversationTurn(role="user", content="hi")]) is None


def test_inherited_context_cannot_smuggle_an_off_topic_query_through(knowledge_base):
    # Ordering regression: the scope check runs on the turn's OWN constraints,
    # before anything is inherited. If it ran on merged constraints, asking
    # "good clothes" after "mahindra" would arrive carrying brand=Mahindra,
    # count as a populated constraint, and be answered with Mahindras instead
    # of refused.
    from unittest.mock import patch

    from src.app.orchestrator import Pipeline, answer_query
    from src.query.models import QueryUnderstandingResult
    from src.query.regex_extraction import known_brands_from_chunk_store

    pipeline = Pipeline(
        kb=knowledge_base, dense=None, bm25=None, hybrid=None, hyde=None, reranker=None,
        known_brands=known_brands_from_chunk_store(knowledge_base.chunk_store),
    )
    history = [
        ConversationTurn(role="user", content="mahindra", constraints=Constraints(brand="Mahindra")),
        ConversationTurn(role="assistant", content="Here are three Mahindras."),
    ]
    off_topic = QueryUnderstandingResult(
        original_query="good clothes",
        standalone_query="good clothes",
        constraints=Constraints(),  # nothing of its own
        extraction_method="llm",
    )
    with patch("src.app.orchestrator.understand_query", return_value=off_topic):
        result = answer_query(pipeline, "good clothes", history=history, generate=False)

    assert result.log.mode == "out_of_scope"
    assert result.outcome.results == []
    assert result.carried_over == []


def test_memory_can_be_turned_off(knowledge_base):
    from unittest.mock import patch

    from src.app.orchestrator import Pipeline, answer_query
    from src.query.models import QueryUnderstandingResult
    from src.query.regex_extraction import known_brands_from_chunk_store

    pipeline = Pipeline(
        kb=knowledge_base, dense=None, bm25=None, hybrid=None, hyde=None, reranker=None,
        known_brands=known_brands_from_chunk_store(knowledge_base.chunk_store),
    )
    history = [
        ConversationTurn(role="user", content="mahindra", constraints=Constraints(brand="Mahindra")),
        ConversationTurn(role="assistant", content="Here are three Mahindras."),
    ]
    understanding = QueryUnderstandingResult(
        original_query="only 7 seaters",
        standalone_query="only 7 seaters",
        constraints=Constraints(seating_capacity=7),
        extraction_method="llm",
    )
    with patch("src.app.orchestrator.understand_query", return_value=understanding), patch(
        "src.app.orchestrator.transform_query", side_effect=lambda q: q
    ), patch("src.app.orchestrator.retrieve_with_relaxation") as mock_retrieve:
        mock_retrieve.return_value = type(
            "O", (), {"mode": "exact", "results": [], "relaxation_steps": [],
                      "original_constraints": Constraints()}
        )()
        off = answer_query(pipeline, "only 7 seaters", history=history, generate=False, use_memory=False)
        on = answer_query(pipeline, "only 7 seaters", history=history, generate=False, use_memory=True)

    assert off.carried_over == []
    assert on.carried_over == ["brand = Mahindra"]
