"""The four negative samplers (DESIGN.md 2.7), vectorized over a batch of positives.

Each sampler takes positive rows (indices into one snapshot's listing table) and returns `k`
negative rows per positive. Any negative equal to the positive, or reviewed by the same guest at
any time, is masked by the caller using `guard_mask`.

These substitute for the behavioural negatives (impressed, viewed, wishlisted) of 2601.06873,
which are not public.
"""
from __future__ import annotations

import numpy as np

CAPACITY_BANDS = np.array([2, 4, 6])  # band 0: 1-2 guests, 1: 3-4, 2: 5-6, 3: 7+


class Groups:
    """Rows bucketed by an integer key, for O(1) uniform draws within a bucket."""

    def __init__(self, keys: np.ndarray):
        self.order = np.argsort(keys, kind="stable")
        self.sorted_keys = keys[self.order]
        uk, start, cnt = np.unique(self.sorted_keys, return_index=True, return_counts=True)
        self.uk, self.start, self.cnt = uk, start, cnt

    def draw(self, keys: np.ndarray, k: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """k draws from each key's bucket; `ok` is False where the bucket has fewer than 2 rows."""
        pos = np.searchsorted(self.uk, keys)
        pos = np.clip(pos, 0, len(self.uk) - 1)
        found = self.uk[pos] == keys
        cnt = np.where(found, self.cnt[pos], 0)
        u = rng.random((len(keys), k))
        off = np.floor(u * np.maximum(cnt, 1)[:, None]).astype(np.int64)
        rows = self.order[np.clip(self.start[pos][:, None] + off, 0, len(self.order) - 1)]
        return rows, cnt >= 2


class Samplers:
    def __init__(self, city_idx: np.ndarray, nbhd_idx: np.ndarray, s2_14: np.ndarray,
                 accommodates: np.ndarray, price: np.ndarray, open_bits: np.ndarray | None = None):
        self.n = len(city_idx)
        self.city = Groups(city_idx.astype(np.int64))
        band = np.searchsorted(CAPACITY_BANDS, accommodates, side="left")
        self.geo_cell = Groups(s2_14.astype(np.int64) * 4 + band)
        self.geo_nbhd = Groups(nbhd_idx.astype(np.int64) * 4 + band)
        self.cell_key = s2_14.astype(np.int64) * 4 + band
        self.nbhd_key = nbhd_idx.astype(np.int64) * 4 + band
        self.city_idx = city_idx
        self.nbhd_idx = nbhd_idx
        lp = np.log1p(np.nan_to_num(price, nan=0.0))
        self.has_price = np.nan_to_num(price, nan=0.0) > 0
        # Price-hard: rows sorted by (neighbourhood, log price); a 20% band is a contiguous range.
        key = nbhd_idx.astype(np.float64) * 100.0 + np.where(self.has_price, lp, -50.0)
        self.p_order = np.argsort(key)
        self.p_key = key[self.p_order]
        self.lp = lp
        self.open_bits = open_bits

    def random(self, pos: np.ndarray, k: int, rng) -> np.ndarray:
        rows, _ = self.city.draw(self.city_idx[pos].astype(np.int64), k, rng)
        return rows

    def geo(self, pos: np.ndarray, k: int, rng, nights: tuple[np.ndarray, np.ndarray] | None = None,
            tries: int = 3) -> np.ndarray:
        """Same S2 level-14 cell and capacity band, else same neighbourhood and band, else city.

        With `nights` = (a, b) offsets into `open_bits`, a draw closed on any of those nights is
        redrawn up to `tries` times (availability is only known for calendar-backed positives).
        """
        rows, ok = self.geo_cell.draw(self.cell_key[pos], k, rng)
        r2, ok2 = self.geo_nbhd.draw(self.nbhd_key[pos], k, rng)
        rows = np.where(ok[:, None], rows, np.where(ok2[:, None], r2, self.random(pos, k, rng)))
        if nights is not None and self.open_bits is not None:
            a, b = nights
            for _ in range(tries):
                bad = ~self._open(rows, a, b)
                if not bad.any():
                    break
                alt, _ = self.geo_nbhd.draw(self.nbhd_key[pos], k, rng)
                rows = np.where(bad, alt, rows)
        return rows

    def _open(self, rows: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ar = np.arange(self.open_bits.shape[1])
        win = (ar[None, :] >= a[:, None]) & (ar[None, :] < b[:, None])
        out = np.empty(rows.shape, bool)
        for j in range(rows.shape[1]):
            out[:, j] = ~(win & ~self.open_bits[rows[:, j]]).any(1)
        return out

    def price(self, pos: np.ndarray, k: int, rng) -> np.ndarray:
        """Same neighbourhood, quote within 20% of the positive's. Falls back to geo-hard."""
        base = self.nbhd_idx[pos].astype(np.float64) * 100.0
        lo = np.searchsorted(self.p_key, base + self.lp[pos] + np.log(0.8))
        hi = np.searchsorted(self.p_key, base + self.lp[pos] + np.log(1.2))
        cnt = hi - lo
        u = rng.random((len(pos), k))
        rows = self.p_order[np.clip(lo[:, None] + np.floor(u * np.maximum(cnt, 1)[:, None]).astype(np.int64),
                                    0, self.n - 1)]
        ok = (cnt >= 2) & self.has_price[pos]
        return np.where(ok[:, None], rows, self.geo(pos, k, rng))

    @staticmethod
    def mined(candidates: np.ndarray, k: int, rng) -> np.ndarray:
        """Uniform draws from each positive's pre-mined top-ranked non-positives ([n, m] rows)."""
        idx = rng.integers(0, candidates.shape[1], size=(len(candidates), k))
        return np.take_along_axis(candidates, idx, 1)


def pair_key(reviewer: np.ndarray, listing: np.ndarray) -> np.ndarray:
    """A 63-bit hash of (reviewer, listing) for the false-negative guard."""
    r = np.asarray(reviewer, dtype=np.uint64)
    l_ = np.asarray(listing, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = r * np.uint64(0x9E3779B97F4A7C15) ^ (l_ + np.uint64(0x632BE59BD9B4E019))
    return (h >> np.uint64(1)).astype(np.int64)


def guard_mask(reviewer: np.ndarray, neg_listing_ids: np.ndarray, pos_listing_ids: np.ndarray,
               reviewed_keys_sorted: np.ndarray) -> np.ndarray:
    """True where a negative is usable: not the positive, and never reviewed by that guest."""
    keys = pair_key(np.repeat(reviewer, neg_listing_ids.shape[1]), neg_listing_ids.ravel())
    pos_ = np.searchsorted(reviewed_keys_sorted, keys)
    pos_ = np.clip(pos_, 0, len(reviewed_keys_sorted) - 1)
    reviewed = (reviewed_keys_sorted[pos_] == keys).reshape(neg_listing_ids.shape)
    return ~reviewed & (neg_listing_ids != pos_listing_ids[:, None])
