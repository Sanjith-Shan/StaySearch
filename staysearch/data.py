"""Loads a prepared snapshot and the queries into arrays the models and evaluators share."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from staysearch.config import CITIES, HORIZON_BITS
from staysearch.features import PREP

CITY_LIST = list(CITIES)
N_AMEN = 200


@lru_cache(maxsize=1)
def vocab() -> dict:
    return json.loads((PREP / "vocab.json").read_text())


@dataclass
class Snapshot:
    rank: int
    listing_id: np.ndarray
    city: np.ndarray          # int index into CITY_LIST
    nbhd: np.ndarray          # nbhd_idx
    snap_ord: np.ndarray      # per-row ordinal of that city's snapshot date (calendar day 0)
    lat: np.ndarray
    lng: np.ndarray
    room_type: np.ndarray
    property_type: np.ndarray
    accommodates: np.ndarray
    min_nights: np.ndarray
    max_nights: np.ndarray
    price: np.ndarray
    ltm: np.ndarray
    rating: np.ndarray
    last_review_days: np.ndarray  # days from last review to the snapshot date, NaN if never reviewed
    s2: np.ndarray            # [n, 3] vocab indices for levels 10, 12, 14
    num: np.ndarray           # [n, 27] standardized
    missing: np.ndarray       # [n, 3]
    amen: np.ndarray          # [n, 200] uint8 multi-hot
    open: np.ndarray          # [n, HORIZON_BITS] bool, day 0 = the city's snapshot date
    has_cal: np.ndarray

    @property
    def n(self) -> int:
        return len(self.listing_id)

    def rows_of(self, ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_order"):
            self._order = np.argsort(self.listing_id)
            self._sorted = self.listing_id[self._order]
        pos = np.clip(np.searchsorted(self._sorted, ids), 0, self.n - 1)
        return np.where(self._sorted[pos] == ids, self._order[pos], -1)


@lru_cache(maxsize=4)
def load_snapshot(rank: int) -> Snapshot:
    t = pq.read_table(PREP / f"listings_s{rank + 1}.parquet")
    g = lambda c: t.column(c).to_numpy(zero_copy_only=False)
    cities = g("city")
    cmap = {c: i for i, c in enumerate(CITY_LIST)}
    snaps = g("snap")
    snap_ord = np.array([dt.date.fromisoformat(s).toordinal() for s in snaps], dtype=np.int64)
    amen = np.zeros((t.num_rows, N_AMEN), dtype=np.uint8)
    lists = t.column("amenities").to_pylist()
    r = np.repeat(np.arange(t.num_rows), [len(a) for a in lists])
    c = np.concatenate([np.asarray(a, dtype=np.int64) for a in lists]) if r.size else np.zeros(0, np.int64)
    amen[r, c] = 1
    bits = t.column("open_bits").to_pylist()
    nb = HORIZON_BITS // 8
    raw = np.frombuffer(b"".join(b if b is not None else bytes(nb) for b in bits), dtype=np.uint8)
    return Snapshot(
        rank=rank, listing_id=g("listing_id").astype(np.int64),
        city=np.array([cmap[x] for x in cities], dtype=np.int64), nbhd=g("nbhd_idx").astype(np.int64),
        snap_ord=snap_ord, lat=g("lat"), lng=g("lng"), room_type=g("room_type").astype(np.int64),
        property_type=g("property_type").astype(np.int64), accommodates=g("accommodates").astype(np.int64),
        min_nights=g("min_nights").astype(np.int64), max_nights=g("max_nights").astype(np.int64),
        price=g("price").astype(np.float64), ltm=g("number_of_reviews_ltm").astype(np.float64),
        rating=g("review_scores_rating").astype(np.float64),
        last_review_days=(g("days_since_last_review").astype(np.float64) if "days_since_last_review" in t.column_names
                          else np.full(t.num_rows, np.nan)),
        s2=np.stack([g("s2_10_idx"), g("s2_12_idx"), g("s2_14_idx")], 1).astype(np.int64),
        num=np.stack(t.column("num").to_numpy(zero_copy_only=False)).astype(np.float32),
        missing=np.stack(t.column("missing").to_numpy(zero_copy_only=False)).astype(np.float32),
        amen=amen,
        open=np.unpackbits(raw.reshape(-1, nb), axis=1, bitorder="little").astype(bool),
        has_cal=np.array([b is not None for b in bits]),
    )


@lru_cache(maxsize=1)
def places() -> dict:
    t = pq.read_table(PREP / "places.parquet")
    d = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
    return d


@lru_cache(maxsize=1)
def history_index() -> dict:
    t = pq.read_table(PREP / "history_index.parquet")
    return {c: t.column(c).to_numpy() for c in t.column_names}


def load_queries(split: str, columns: list[str] | None = None) -> dict:
    t = pq.read_table(PREP / "queries.parquet", filters=[("split", "=", split)], columns=columns)
    d = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
    if "city" in d:
        cmap = {c: i for i, c in enumerate(CITY_LIST)}
        d["city_idx"] = np.array([cmap[x] for x in d["city"]], dtype=np.int64)
    return d


def stratified_sample(q: dict, per_city: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(q["city_idx"]):
        idx = np.nonzero(q["city_idx"] == c)[0]
        out.append(rng.choice(idx, size=min(per_city, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


def hist_rows(q: dict, snap: Snapshot, idx: np.ndarray | None = None) -> list[np.ndarray]:
    """Each query's earlier reviewed listings, as rows of `snap` (listings absent there are skipped)."""
    H = history_index()
    idx = np.arange(len(q["hist_lo"])) if idx is None else idx
    lo, hi = q["hist_lo"][idx], q["hist_hi"][idx]
    flat = np.concatenate([H["listing_id"][a:b] for a, b in zip(lo, hi)]) if len(idx) else np.zeros(0, np.int64)
    rows = snap.rows_of(flat)
    out, k = [], 0
    for a, b in zip(lo, hi):
        r = rows[k:k + (b - a)]
        out.append(r[r >= 0])
        k += b - a
    return out
