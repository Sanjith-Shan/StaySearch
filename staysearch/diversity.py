"""M6(a): diversity re-ranking after Haldar et al., *Learning To Rank Diversely* (CIKM'23), in its
simplest greedy form: build the top 10 one slot at a time, scoring each remaining candidate by its
ranker score minus lambda times its similarity to the listings already placed. Similarity combines
location (exp(-km / 1.0)) and price (exp(-|log price diff| / 0.25)), averaged.

Reports, per lambda, NDCG@10 and MRR of the reviewed stay against intra-list diversity of the top 10
(mean pairwise distance in km, standard deviation of log price, distinct room types), on the test
split's top K. The trade-off curve is the result; no lambda is claimed as best.

Usage: python -m staysearch.diversity --tower mix_l2 --ranker dnn_hist
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from staysearch import results
from staysearch.position_bias import load_truth
from staysearch.qfeatures import haversine_km
from staysearch.ranking import Candidates, model_scores
from staysearch.evaluate import bootstrap_ci

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def greedy(score, lat, lng, lp, lam, k=10):
    n = len(score)
    z = (score - score.mean()) / (score.std() + 1e-9)
    placed, sim_max = [], np.zeros(n)
    avail = np.ones(n, bool)
    for _ in range(min(k, n)):
        v = np.where(avail, z - lam * sim_max, -np.inf)
        j = int(np.argmax(v))
        placed.append(j)
        avail[j] = False
        loc = np.exp(-haversine_km(lat, lng, lat[j], lng[j]) / 1.0)
        pr = np.exp(-np.abs(lp - lp[j]) / 0.25)
        sim_max = np.maximum(sim_max, 0.5 * (loc + pr))
    rest = np.nonzero(avail)[0]
    return np.concatenate([placed, rest[np.argsort(-score[rest])]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="mix_l2")
    ap.add_argument("--ranker", default="dnn_hist")
    ap.add_argument("--per-city", type=int, default=1000)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results.reset("m6_diversity")
    c = Candidates("test", args.per_city, args.K, args.tower, seed=33, include_positive=False)
    X, y, group = c.matrix(True)
    scores = model_scores(load_truth(args.ranker, X.shape[1]), X.astype(np.float32))
    s = c.snap
    offs = np.concatenate([[0], np.cumsum(group)])
    for lam in LAMBDAS:
        nd, mrr, dist, pstd, rooms = [], [], [], [], []
        for g in range(len(group)):
            sl = slice(offs[g], offs[g + 1])
            rows = c.rows_list[g]
            if len(rows) < 2:
                continue
            lat, lng = s.lat[rows], s.lng[rows]
            lp = np.log1p(np.nan_to_num(s.price[rows], nan=np.nanmedian(s.price[rows]) if np.isfinite(s.price[rows]).any() else 100.0))
            order = greedy(scores[sl], lat, lng, lp, lam)
            yy = y[sl][order]
            hit = np.nonzero(yy > 0)[0]
            r = hit[0] + 1 if len(hit) else None
            nd.append(1 / np.log2(r + 1) if r and r <= 10 else 0.0)
            mrr.append(1 / r if r else 0.0)
            t = order[:10]
            d = haversine_km(lat[t][:, None], lng[t][:, None], lat[t][None, :], lng[t][None, :])
            dist.append(d[np.triu_indices(len(t), 1)].mean())  # lists of 2+ only (skip above)
            pstd.append(lp[t].std())
            rooms.append(len(np.unique(s.room_type[rows][t])))
        row = {"exp": "m6_diversity", "split": "test", "lambda": lam, "K": args.K, "n_queries": len(nd)}
        for name, v in (("ndcg@10", nd), ("mrr", mrr)):
            m, lo, hi = bootstrap_ci(np.array(v))
            row[name], row[f"{name}_ci95"] = m, [lo, hi]
        row.update({"top10_mean_pairwise_km": float(np.mean(dist)), "top10_log_price_std": float(np.mean(pstd)),
                    "top10_distinct_room_types": float(np.mean(rooms))})
        results.append("m6_diversity", row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if "ci95" not in k}, flush=True)


if __name__ == "__main__":
    main()
