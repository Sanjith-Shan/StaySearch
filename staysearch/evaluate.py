"""Point-in-time evaluation of the reviewed stay against its eligible set (DESIGN.md 2.4).

For every evaluation query the eligible set is recomputed from the snapshot the split uses: listings
in the place's bbox, open every night of the stay in that snapshot's calendar, min/max nights
satisfied, capacity at least the guests. Each scorer ranks the eligible set and we record where the
reviewed stay landed. Ties are broken uniformly at random in expectation (popularity has many ties),
so recall@k and NDCG@10 are the expected values under random tie-breaking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from staysearch.data import CITY_LIST, Snapshot, places

Scorer = Callable[[int, np.ndarray], np.ndarray]  # (query position in the eval set, rows) -> scores
KS = (10, 100, 500)
SPLIT_RANK = {"rank_train": 0, "val": 1, "test": 2}


@dataclass
class EvalSet:
    q: dict
    idx: np.ndarray                    # indices into q
    snap: Snapshot
    pos_rows: np.ndarray = field(init=False)
    _place_rows: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.pos_rows = self.snap.rows_of(self.q["listing_id"][self.idx])
        assert (self.pos_rows >= 0).all(), "every kept positive is in its snapshot by construction"

    def place_rows(self, p: int) -> np.ndarray:
        if p not in self._place_rows:
            P = places()
            s = self.snap
            m = ((s.lat >= P["min_lat"][p]) & (s.lat <= P["max_lat"][p]) &
                 (s.lng >= P["min_lng"][p]) & (s.lng <= P["max_lng"][p]))
            self._place_rows[p] = np.nonzero(m)[0]
        return self._place_rows[p]

    def eligible(self, j: int) -> np.ndarray:
        """Rows eligible for evaluation query j (position in self.idx)."""
        i = self.idx[j]
        s = self.snap
        rows = self.place_rows(int(self.q["place_idx"][i]))
        g, n = int(self.q["recon_guests"][i]), int(self.q["recon_nights"][i])
        ok = (s.accommodates[rows] >= g) & (s.min_nights[rows] <= n) & (np.maximum(s.max_nights[rows], 1) >= n)
        rows = rows[ok & s.has_cal[rows]]
        a = self.q["recon_checkin"][i] - s.snap_ord[rows]
        lo = int(a.min()) if len(a) else 0
        # Cities differ in snapshot date, but one place is one city, so the offset is shared.
        assert len(np.unique(a)) <= 1
        win = s.open[rows, lo:lo + n]
        return rows[win.all(1) & (win.shape[1] == n)]

    def __len__(self):
        return len(self.idx)


def _metrics_from_rank(greater: int, ties: int, n: int) -> dict:
    """Expected recall@k, NDCG@10 and reciprocal rank when the positive is tied with `ties` others."""
    out = {}
    for k in KS:
        out[f"recall@{k}"] = 0.0 if greater >= k else min(1.0, (k - greater) / (ties + 1))
    r = np.arange(greater + 1, greater + ties + 2)
    out["ndcg@10"] = float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0.0).mean())
    out["mrr"] = float((1.0 / r).mean())
    out["rank"] = greater + 1 + ties / 2
    out["eligible"] = n
    return out


def evaluate(es: EvalSet, scorers: dict[str, Scorer], progress: bool = False) -> dict[str, dict]:
    """Per-query metric arrays for each scorer: {name: {metric: np.ndarray}}."""
    per = {name: {m: np.zeros(len(es)) for m in [*(f"recall@{k}" for k in KS), "ndcg@10", "mrr", "rank", "eligible"]}
           for name in scorers}
    for j in range(len(es)):
        rows = es.eligible(j)
        pos = es.pos_rows[j]
        assert pos in rows, "the positive is eligible by construction"
        for name, fn in scorers.items():
            s = fn(j, rows)
            sp = s[rows == pos][0]
            greater = int((s > sp).sum())
            ties = int((s == sp).sum()) - 1
            for m, v in _metrics_from_rank(greater, ties, len(rows)).items():
                per[name][m][j] = v
        if progress and j % 2000 == 0:
            print(f"  eval {j}/{len(es)}", flush=True)
    return per


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return (float("nan"),) * 3
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(es: EvalSet, per: dict[str, dict], has_hist: np.ndarray) -> list[dict]:
    """Rows for results/*.jsonl: all cities, per city, and the with-history slice, with 95% CIs."""
    city = es.q["city_idx"][es.idx]
    rows = []
    slices = {"all": np.ones(len(es), bool), "with_history": has_hist, "no_history": ~has_hist}
    for c in np.unique(city):
        slices[f"city={CITY_LIST[c]}"] = city == c
    for name, m in per.items():
        for sl, mask in slices.items():
            rec = {"model": name, "slice": sl, "n_queries": int(mask.sum()),
                   "eligible_median": float(np.median(m["eligible"][mask])) if mask.any() else None}
            for k in [*(f"recall@{k}" for k in KS), "ndcg@10", "mrr"]:
                mean, lo, hi = bootstrap_ci(m[k][mask], n_boot=1000 if sl in ("all", "with_history") else 200)
                rec[k] = mean
                rec[f"{k}_ci95"] = [lo, hi]
            rows.append(rec)
    return rows
