"""Query tower inputs and ranker cross features, exactly as service/BUNDLE_FORMAT.md defines them.

This is the Python reference. The Kotlin service implements the same formulas and a golden-file
test (bundle/golden/*.jsonl) keeps the two from drifting.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

Q_FEAT_DIM = 87
HIST_DIM = 70  # has_history, price mean, 4 room-type shares, 64-d mean embedding
N_CROSS = 15
GAP_CAP = 28
LEAD_EDGES = np.array([1, 7, 14, 30, 60])  # bucket b holds lead <= edge[b]; last bucket is >= 61
ORDINAL_EPOCH = dt.date(1, 1, 1).toordinal()


def lead_bucket(lead_days: np.ndarray) -> np.ndarray:
    return np.searchsorted(LEAD_EDGES, np.maximum(lead_days, 0), side="left")


def dow(ordinal_days: np.ndarray) -> np.ndarray:
    """Monday = 0, from proleptic Gregorian ordinals (date.toordinal())."""
    return (np.asarray(ordinal_days) - 1) % 7


def norm_log_price(price: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(price, dtype=np.float64), nan=0.0)
    return np.where(p > 0, np.log1p(p) - 5.0, 0.0)


def query_base(area_km2, is_city, guests, nights, lead_days, checkin_ord) -> np.ndarray:
    """Features 0..16 for a batch of queries."""
    n = len(guests)
    x = np.zeros((n, 17), dtype=np.float32)
    x[:, 0] = np.log1p(area_km2) / 10.0
    x[:, 1] = np.asarray(is_city, dtype=np.float32)
    x[:, 2] = np.asarray(guests) / 16.0
    x[:, 3] = np.log1p(nights) / np.log(31.0)
    x[np.arange(n), 4 + lead_bucket(np.asarray(lead_days))] = 1.0
    x[np.arange(n), 10 + dow(checkin_ord)] = 1.0
    return x


def history_block(hist_rows: list[np.ndarray], emb: np.ndarray, price: np.ndarray,
                  room_type: np.ndarray) -> np.ndarray:
    """Features 17..86: has_history, mean normalized log price, room-type mix, mean embedding."""
    out = np.zeros((len(hist_rows), HIST_DIM), dtype=np.float32)
    lp = norm_log_price(price)
    for i, rows in enumerate(hist_rows):
        if len(rows) == 0:
            continue
        out[i, 0] = 1.0
        out[i, 1] = lp[rows].mean()
        out[i, 2:6] = np.bincount(room_type[rows], minlength=4)[:4] / len(rows)
        out[i, 6:] = emb[rows].mean(0)
    return out


def haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gaps(open_rows: np.ndarray, a: int, b: int, cap: int = GAP_CAP) -> tuple[np.ndarray, np.ndarray]:
    """Open nights immediately before night a and from night b on (each capped), per row.

    `open_rows` is [n, horizon] bool with day 0 = the calendar's first night; nights outside the
    calendar count as closed, so a stay starting on day 0 has no gap before it.
    """
    n, h = open_rows.shape
    before = np.zeros(n, np.int64)
    alive = np.ones(n, bool)
    for k in range(1, cap + 1):
        d = a - k
        if d < 0:
            break
        alive &= open_rows[:, d] if d < h else False
        before += alive
    after = np.zeros(n, np.int64)
    alive = np.ones(n, bool)
    for k in range(cap):
        d = b + k
        if d >= h:
            break
        alive &= open_rows[:, d]
        after += alive
    return before, after


def cross_features(retrieval_dist, l_lat, l_lng, l_price, l_min_nights, l_accommodates, l_room_type,
                   l_emb, p_lat, p_lng, p_area, guests, nights, lead_days, hist,
                   gap_before=None, gap_after=None, retrieval_rank=None, k: int = 100) -> np.ndarray:
    """The 12 ranker cross features for one query (scalars) against candidate arrays.

    `hist` is the query's 70-d history block (zeros without history).
    """
    n = len(l_lat)
    x = np.zeros((n, N_CROSS), dtype=np.float32)
    price = np.nan_to_num(np.asarray(l_price, dtype=np.float64), nan=0.0)
    has_price = price > 0
    has_hist = hist[0] > 0
    x[:, 0] = retrieval_dist
    x[:, 1] = haversine_km(l_lat, l_lng, p_lat, p_lng) / np.sqrt(1.0 + p_area)
    x[:, 2] = np.where(has_price, np.log1p(price) - np.log(guests), 0.0)
    x[:, 3] = np.log1p(np.maximum(nights - np.asarray(l_min_nights), 0))
    x[:, 4] = guests / np.maximum(np.asarray(l_accommodates, dtype=np.float64), 1.0)
    if has_hist:
        x[:, 5] = l_emb @ hist[6:]
        x[:, 6] = np.where(has_price, (np.log1p(price) - 5.0) - hist[1], 0.0)
        x[:, 11] = hist[2 + np.asarray(l_room_type)]
    x[:, 7] = float(has_hist)
    x[:, 8] = np.log1p(max(lead_days, 0)) / np.log(366.0)
    x[:, 9] = np.log1p(nights) / np.log(31.0)
    x[:, 10] = guests / 16.0
    if gap_before is not None:
        x[:, 12] = np.log1p(gap_before) / np.log1p(GAP_CAP)
        x[:, 13] = np.log1p(gap_after) / np.log1p(GAP_CAP)
    x[:, 14] = np.log1p(retrieval_rank if retrieval_rank is not None else np.argsort(np.argsort(retrieval_dist, kind="stable"), kind="stable")) / np.log1p(k)
    return x
