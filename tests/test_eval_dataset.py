import config
from src.evaluation.dataset import EvalQuery, load_eval_queries


def test_load_eval_queries_returns_expected_count_and_type():
    queries = load_eval_queries()
    assert 20 <= len(queries) <= 30
    assert all(isinstance(q, EvalQuery) for q in queries)


def test_load_eval_queries_covers_required_categories():
    required = {
        "bm25_friendly", "dense_semantic", "hybrid", "multi_constraint", "exact_model",
        "price", "follow_up_context", "query_expansion", "hyde_suitable", "no_result",
        "relaxation", "feature_specific", "overview_style",
    }
    queries = load_eval_queries()
    present = {q.category for q in queries}
    assert required.issubset(present)


def test_load_eval_queries_unique_ids():
    queries = load_eval_queries()
    ids = [q.id for q in queries]
    assert len(ids) == len(set(ids))


def test_load_eval_queries_follow_up_entries_have_history():
    queries = load_eval_queries()
    follow_ups = [q for q in queries if q.category == "follow_up_context"]
    assert len(follow_ups) > 0
    assert all(len(q.history) > 0 for q in follow_ups)


def test_eval_queries_path_exists():
    assert config.EVAL_QUERIES_PATH.exists()
