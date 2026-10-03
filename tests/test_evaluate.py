import numpy as np

from staysearch.evaluate import _metrics_from_rank, bootstrap_ci


def test_unique_rank():
    m = _metrics_from_rank(greater=0, ties=0, n=50)
    assert m["recall@10"] == 1.0 and m["ndcg@10"] == 1.0 and m["mrr"] == 1.0


def test_ties_are_expected_values():
    # Positive tied with 19 others for first place: expected recall@10 is 10/20.
    m = _metrics_from_rank(greater=0, ties=19, n=100)
    assert abs(m["recall@10"] - 0.5) < 1e-12
    assert m["recall@100"] == 1.0
    assert abs(m["rank"] - 10.5) < 1e-12


def test_miss_beyond_k():
    m = _metrics_from_rank(greater=600, ties=0, n=1000)
    assert m["recall@500"] == 0.0 and m["ndcg@10"] == 0.0


def test_bootstrap_brackets_mean():
    x = np.random.default_rng(0).random(500)
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi
