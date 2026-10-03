"""exp1 (retrieval quality), exp2 (negatives) and exp8 (leakage) on the test split: exact KNN over
each query's point-in-time eligible set, for every trained tower, on the same stratified sample the
M1 baselines use (per_city=2000, seed=11), so the rows compare directly with results/m1_baselines.jsonl.

Usage: python -m staysearch.eval_towers --models mix_l2 random_l2 ...
"""
from __future__ import annotations

import argparse

import numpy as np

from staysearch import results
from staysearch.ann import load_model
from staysearch.baselines import baseline_scorers, train_linear
from staysearch.data import load_queries, load_snapshot, stratified_sample
from staysearch.evaluate import SPLIT_RANK, EvalSet, evaluate, summarize
from staysearch.towers import embed_listings, embed_queries
from staysearch.train_towers import _scorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-city", type=int, default=2000)
    ap.add_argument("--with-baselines", action="store_true")
    ap.add_argument("--out", default="m2_retrieval")
    args = ap.parse_args()
    s = load_snapshot(SPLIT_RANK[args.split])
    q = load_queries(args.split)
    idx = stratified_sample(q, args.per_city, seed=11)
    es = EvalSet(q, idx, s)
    scorers = {}
    meta = {}
    for name in args.models:
        model, cfg = load_model(name)
        emb = embed_listings(model, s)
        qe, _ = embed_queries(model, q, idx, s, emb, drop_guests_nights=cfg["drop_guests_nights"],
                              use_history=cfg["history"])
        scorers[name] = _scorer(model, qe, emb)
        meta[name] = {k: cfg[k] for k in ("sampler", "similarity", "history", "drop_guests_nights", "epochs")}
    if args.with_baselines:
        scorers.update(baseline_scorers(es, train_linear(s)))
    per = evaluate(es, scorers, progress=True)
    has_hist = (q["hist_hi"][idx] - q["hist_lo"][idx]) > 0
    for row in summarize(es, per, has_hist):
        exp = "exp1"
        m = meta.get(row["model"], {})
        if m.get("drop_guests_nights"):
            exp = "exp8"
        results.append(args.out, {"exp": exp, "split": args.split, "snapshot": f"S{SPLIT_RANK[args.split] + 1}",
                                  "per_city_sample": args.per_city, **m, **row})
    for n, m in per.items():
        print(n, {k: round(float(m[k].mean()), 4) for k in ("recall@10", "recall@100", "recall@500", "ndcg@10")})


if __name__ == "__main__":
    main()
