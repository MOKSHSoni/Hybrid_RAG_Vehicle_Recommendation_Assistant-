from src.evaluation.metrics import mean, ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k, retrieval_success_rate


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5
    assert precision_at_k(retrieved, relevant, k=2) == 1 / 2
    assert precision_at_k(retrieved, relevant, k=1) == 1.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "z"}  # z never retrieved
    assert recall_at_k(retrieved, relevant, k=3) == 2 / 3
    assert recall_at_k(retrieved, relevant, k=1) == 1 / 3


def test_recall_at_k_no_ground_truth():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 1 / 2
    assert reciprocal_rank(["a", "x"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0  # never found


def test_ndcg_at_k_perfect_ranking():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0  # both relevant items ranked first


def test_ndcg_at_k_worst_ranking_scores_lower():
    relevant = {"c"}
    best = ndcg_at_k(["c", "a", "b"], relevant, k=3)
    worst = ndcg_at_k(["a", "b", "c"], relevant, k=3)
    assert best > worst


def test_ndcg_at_k_no_relevant_items():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_retrieval_success_rate():
    results = [
        (["a", "b"], {"a"}),  # success
        (["x", "y"], {"z"}),  # failure
        (["m", "n"], {"n"}),  # success
    ]
    assert retrieval_success_rate(results) == 2 / 3


def test_retrieval_success_rate_empty():
    assert retrieval_success_rate([]) == 0.0


def test_mean():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
