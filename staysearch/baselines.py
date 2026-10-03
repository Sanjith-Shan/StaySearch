"""M1 retrieval baselines, evaluated on the point-in-time eligible sets.

- geo: distance from the search area's centre (closest first).
- popularity: number_of_reviews_ltm as of the snapshot (ties broken at random in expectation).
- linear: the paper's legacy baseline, one query-independent score per listing from a logistic
  regression, retrained "daily": on the eval snapshot's date, from reviewed stays in the 90 days
  before it (positives) against random same-city listings (negatives), with that snapshot's
  features. Nothing after the snapshot date is used.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from staysearch import results
from staysearch.data import CITY_LIST, Snapshot, load_queries, load_snapshot, places, stratified_sample, history_index
from staysearch.evaluate import SPLIT_RANK, EvalSet, evaluate, summarize
from staysearch.qfeatures import haversine_km


def listing_matrix(s: Snapshot) -> np.ndarray:
    room = np.eye(4, dtype=np.float32)[s.room_type]
    return np.concatenate([s.num, s.missing, room, s.amen.astype(np.float32)], 1)


def train_linear(s: Snapshot, seed: int = 0) -> np.ndarray:
    """Fit on reviews in the 90 days before each city's snapshot date; returns a score per row."""
    H = history_index()
    rng = np.random.default_rng(seed)
    snap_by_city = {c: s.snap_ord[s.city == c][0] for c in np.unique(s.city)}
    pos_rows = []
    for c, so in snap_by_city.items():
        m = (H["d"] >= so - 90) & (H["d"] < so)
        rows = s.rows_of(H["listing_id"][m])
        rows = rows[rows >= 0]
        pos_rows.append(rows[s.city[rows] == c])
    pos = np.concatenate(pos_rows)
    neg = _neg_like(s, pos, rng)
    X = listing_matrix(s)
    Xtr = np.concatenate([X[pos], X[neg]])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    lr = LogisticRegression(max_iter=300, C=1.0)
    lr.fit(Xtr, y)
    return lr.decision_function(X)


def _neg_like(s: Snapshot, pos: np.ndarray, rng) -> np.ndarray:
    """One random same-city listing per positive, vectorized."""
    order = np.argsort(s.city, kind="stable")
    sc = s.city[order]
    start = np.searchsorted(sc, s.city[pos], "left")
    end = np.searchsorted(sc, s.city[pos], "right")
    return order[start + np.floor(rng.random(len(pos)) * (end - start)).astype(np.int64)]


def baseline_scorers(es: EvalSet, linear_score: np.ndarray) -> dict:
    P = places()
    s = es.snap
    q = es.q

    def geo(j, rows):
        p = q["place_idx"][es.idx[j]]
        return -haversine_km(s.lat[rows], s.lng[rows], P["lat"][p], P["lng"][p])

    rng = np.random.default_rng(1)
    return {
        "random": lambda j, rows: rng.random(len(rows)),
        "geo_distance": geo,
        "popularity_ltm": lambda j, rows: s.ltm[rows],
        "linear_query_independent": lambda j, rows: linear_score[rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val", "test"])
    ap.add_argument("--per-city", type=int, default=2000)
    args = ap.parse_args()
    results.reset("m1_baselines")
    for split in args.splits:
        s = load_snapshot(SPLIT_RANK[split])
        q = load_queries(split)
        idx = stratified_sample(q, args.per_city, seed=11)
        es = EvalSet(q, idx, s)
        lin = train_linear(s)
        per = evaluate(es, baseline_scorers(es, lin), progress=True)
        has_hist = (q["hist_hi"][idx] - q["hist_lo"][idx]) > 0
        for row in summarize(es, per, has_hist):
            results.append("m1_baselines", {"exp": "exp1", "split": split, "snapshot": f"S{SPLIT_RANK[split] + 1}",
                                            "per_city_sample": args.per_city, **row})
        print(split, {n: round(float(m["recall@100"].mean()), 4) for n, m in per.items()})


if __name__ == "__main__":
    main()
