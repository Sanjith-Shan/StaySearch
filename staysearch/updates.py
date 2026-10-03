"""The update stream: real changes between two consecutive snapshots of each city.

- avail: a night present in both calendars that flipped open <-> closed (host or guest actions,
  coarsened to a quarter; booked and blocked are indistinguishable).
- price: the listing-level quote changed (S3's quote is carried forward, see DESIGN.md 2.5).
- new: a listing in S4 only (cold start: tower embedding and nearest-centroid cluster at ingest).
- remove: a listing in S3 only.

Nights are written as offsets from a global base date (the earliest S3 date across cities) so one
bitset layout serves every city. Output: data/prepared/updates.parquet, shuffled with a fixed seed.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from staysearch.config import HORIZON_BITS
from staysearch.features import PREP, snapshots
from staysearch.labels import load_snapshot

SEED = 7
BUNDLE_HORIZON = 448
TYPES = {"avail": 0, "price": 1, "new": 2, "remove": 3}


def base_date(rank: int = 2) -> dt.date:
    return min(dt.date.fromisoformat(v[rank]) for v in snapshots().values() if len(v) > rank)


def build(r1: int = 2, r2: int = 3) -> dict:
    snaps = snapshots()
    A, B = load_snapshot(r1), load_snapshot(r2)
    base = base_date(r1)
    out = {"type": [], "listing_id": [], "night": [], "open": [], "price": []}
    counts = {}
    for city, ss in snaps.items():
        d1, d2 = dt.date.fromisoformat(ss[r1]), dt.date.fromisoformat(ss[r2])
        ia = np.nonzero(A["city"] == city)[0]
        ib = np.nonzero(B["city"] == city)[0]
        ida, idb = A["listing_id"][ia].astype(np.int64), B["listing_id"][ib].astype(np.int64)
        common, pa_, pb_ = np.intersect1d(ida, idb, return_indices=True)
        ra, rb = ia[pa_], ib[pb_]
        shift = (d2 - d1).days
        # Nights present in both calendars, expressed in A's offsets: [shift, HORIZON_BITS).
        both = A["has_cal"][ra] & B["has_cal"][rb]
        ra, rb, cl = ra[both], rb[both], common[both]
        n_av = 0
        for j in range(0, len(ra), 20_000):
            oa = A["open"][ra[j:j + 20_000], shift:HORIZON_BITS]
            ob = B["open"][rb[j:j + 20_000], :HORIZON_BITS - shift]
            li, ni = np.nonzero(oa != ob)
            n_av += len(li)
            out["type"].append(np.zeros(len(li), np.int8))
            out["listing_id"].append(cl[j:j + 20_000][li])
            out["night"].append((ni + shift + (d1 - base).days).astype(np.int16))
            out["open"].append(ob[li, ni])
            out["price"].append(np.full(len(li), np.nan))
        pa_price, pb_price = A["price"][ia[pa_]], B["price"][ib[pb_]]
        chg = ~np.isclose(np.nan_to_num(pa_price, nan=-1), np.nan_to_num(pb_price, nan=-1)) & ~np.isnan(pb_price)
        n_new = len(np.setdiff1d(idb, ida))
        new_ids = np.setdiff1d(idb, ida)
        rem_ids = np.setdiff1d(ida, idb)
        for t, ids, prices in (("price", common[chg], pb_price[chg]), ("new", new_ids, np.full(len(new_ids), np.nan)),
                               ("remove", rem_ids, np.full(len(rem_ids), np.nan))):
            out["type"].append(np.full(len(ids), TYPES[t], np.int8))
            out["listing_id"].append(ids)
            out["night"].append(np.full(len(ids), -1, np.int16))
            out["open"].append(np.zeros(len(ids), bool))
            out["price"].append(prices.astype(np.float64))
        counts[city] = {"from": ss[r1], "to": ss[r2], "listings_both": int(len(common)),
                        "avail_events": int(n_av), "price_events": int(chg.sum()),
                        "new_listing_events": int(n_new), "removed_listing_events": int(len(rem_ids)),
                        "overlap_nights": HORIZON_BITS - shift}
        print(f"[updates] {city}: {counts[city]}", flush=True)
    cols = {k: np.concatenate(v) for k, v in out.items()}
    perm = np.random.default_rng(SEED).permutation(len(cols["type"]))
    cols = {k: v[perm] for k, v in cols.items()}
    pq.write_table(pa.table(cols), PREP / "updates.parquet", compression="zstd")
    meta = {"base_date": base.isoformat(), "bundle_horizon": BUNDLE_HORIZON, "per_city": counts,
            "total_events": int(len(perm))}
    (PREP / "updates_summary.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    m = build()
    print("total", m["total_events"])
