"""Reviewed stays, reconstructed searches, splits and point-in-time eligibility (DESIGN.md section 2).

Output, in data/prepared/:
- places.parquet: one row per place (each city, each neighbourhood) with its bbox and area.
- queries.parquet: one row per kept reviewed stay with its reconstructed search, split, place,
  eligibility snapshot and guest history (earlier reviewed listing ids).
- results/m0_data.jsonl rows are written by staysearch.m0_report, not here.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from staysearch.config import AVG_LOS, CITIES, DATA_DIR, HORIZON_BITS
from staysearch.features import PREP, snapshots

SEED = 20261003
TOWER_TRAIN_START = dt.date(2023, 9, 1)
REVIEW_WINDOW = 14
PARTY_P = np.array([0.22, 0.42, 0.11, 0.13, 0.05, 0.04, 0.015, 0.015])
TRIP_GAP_DAYS = 30
HIST_MAX = 20
KM_PER_DEG = 111.32


def bbox_area_km2(min_lat, min_lng, max_lat, max_lng) -> float:
    mid = np.radians((min_lat + max_lat) / 2)
    h = (max_lat - min_lat) * KM_PER_DEG
    w = (max_lng - min_lng) * KM_PER_DEG * np.cos(mid)
    return float(max(h, 0.05) * max(w, 0.05))


def load_snapshot(rank: int) -> dict:
    t = pq.read_table(PREP / f"listings_s{rank + 1}.parquet",
                      columns=["listing_id", "city", "snap", "nbhd", "nbhd_idx", "lat", "lng",
                               "accommodates", "min_nights", "max_nights", "price", "open_bits", "room_type"])
    d = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names if c != "open_bits"}
    bits = t.column("open_bits").to_pylist()
    nbytes = HORIZON_BITS // 8
    raw = np.frombuffer(b"".join(b if b is not None else bytes(nbytes) for b in bits), dtype=np.uint8)
    d["open"] = np.unpackbits(raw.reshape(-1, nbytes), axis=1, bitorder="little").astype(bool)
    d["has_cal"] = np.array([b is not None for b in bits])
    d["row"] = {int(l): i for i, l in enumerate(d["listing_id"])}
    return d


def build_places() -> tuple[list[dict], dict[str, int]]:
    con = duckdb.connect()
    parts = " union all ".join(f"select city, nbhd, lat, lng from '{PREP}/listings_s{r}.parquet'" for r in range(1, 5))
    places = []
    for kind, key in (("city", "city"), ("nbhd", "nbhd")):
        rows = con.execute(f"""select {key}, any_value(city), min(lat), min(lng), max(lat), max(lng),
                                      avg(lat), avg(lng), count(*) from ({parts}) group by {key} order by {key}""").fetchall()
        for name, city, a, b, c, e, clat, clng, n in rows:
            places.append({"idx": len(places), "kind": kind, "city": city, "name": name,
                           "lat": clat, "lng": clng, "min_lat": a, "min_lng": b, "max_lat": c, "max_lng": e,
                           "area_km2": bbox_area_km2(a, b, c, e)})
    index = {(p["kind"], p["name"]): p["idx"] for p in places}
    pq.write_table(pa.Table.from_pylist(places), PREP / "places.parquet")
    return places, index


def _rows_in(snap: dict, lids: np.ndarray) -> np.ndarray:
    """Row of each listing id in a snapshot, -1 if absent."""
    ids = snap["listing_id"].astype(np.int64)
    order = np.argsort(ids)
    pos = np.searchsorted(ids[order], lids)
    pos = np.clip(pos, 0, len(ids) - 1)
    hit = ids[order][pos] == lids
    return np.where(hit, order[pos], -1)


def build() -> dict:
    rng = np.random.default_rng(SEED)
    snaps = snapshots()
    cities = [c for c in snaps if len(snaps[c]) >= 4]
    places, pidx = build_places()
    con = duckdb.connect()
    parts = []
    for ci, city in enumerate(cities):
        t = con.execute(f"""select listing_id, reviewer_id, date_diff('day', DATE '0001-01-01', date) + 1 as d
                            from '{DATA_DIR}/reviews/{city}.parquet'""").fetchnumpy()
        parts.append((t["listing_id"].astype(np.int64), t["reviewer_id"].astype(np.int64),
                      t["d"].astype(np.int32), np.full(len(t["d"]), ci, np.int8)))
    L, R, D, C = (np.concatenate([p[k] for p in parts]) for k in range(4))
    del parts
    counts: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for ci, city in enumerate(cities):
        m = C == ci
        counts[city]["reviews_total"] = int(m.sum())
        counts[city]["reviewers_total"] = int(len(np.unique(R[m])))

    # Cross-city reviewers: reviewer ids that appear in more than one city's review file.
    pair = np.unique(np.stack([R, C.astype(np.int64)], 1), axis=0)
    rid_u, n_c = np.unique(pair[:, 0], return_counts=True)
    multi = set(rid_u[n_c > 1].tolist())
    n_cross_total = len(multi)
    multi_mask = np.isin(pair[:, 0], rid_u[n_c > 1])
    for ci, city in enumerate(cities):
        counts[city]["cross_city_reviewers"] = int((multi_mask & (pair[:, 1] == ci)).sum())
    del pair

    # Every review sorted by (reviewer, date): a guest's history is a contiguous slice.
    order = np.lexsort((D, R))
    HR, HD, HL = R[order], D[order], L[order]
    pq.write_table(pa.table({"reviewer_id": HR, "d": HD, "listing_id": HL}),
                   PREP / "history_index.parquet", compression="zstd")

    S = [load_snapshot(r) for r in range(3)]
    keep_cols = defaultdict(list)
    t0 = TOWER_TRAIN_START.toordinal()
    for ci, city in enumerate(cities):
        sord = np.array([dt.date.fromisoformat(s).toordinal() for s in snaps[city]])
        m = (C == ci) & (D >= t0)
        lid, rid, rd = L[m], R[m], D[m].astype(np.int64)
        n = len(rd)
        checkout = rd - rng.integers(1, REVIEW_WINDOW + 1, size=n)
        s_draw = rng.geometric(1.0 / AVG_LOS[city], size=n)
        coin_city = rng.random(n) < 0.5
        u_party = rng.random(n)
        lead_train = rng.integers(0, 91, size=n)
        # Provisional split from the review date.
        split = np.full(n, 3, np.int8)               # 0 tower_train, 1 rank_train, 2 val, 3 test
        split[rd < sord[0]] = 0
        split[(rd >= sord[0]) & (rd < sord[1])] = 1
        split[(rd >= sord[1]) & (rd < sord[2] + REVIEW_WINDOW + 30)] = 2
        rank = np.array([0, 0, 1, 2])[split]
        # Reconstruction needs the listing's rules from the snapshot the split uses (val rows that
        # turn out to start after S3 move to test below and are re-read from S3).
        rows = np.full(n, -1)
        for r in range(3):
            sel = rank == r
            rows[sel] = _rows_in(S[r], lid[sel])
        for r in range(3):
            sel = (rank == r) & (rows >= 0)
            ok_city = np.zeros(n, bool)
            ok_city[sel] = S[r]["city"][rows[sel]] == city
            rows[sel & ~ok_city] = -1
        mn = np.ones(n, np.int64); mx = np.full(n, 1125, np.int64); acc = np.ones(n, np.int64)
        for r in range(3):
            sel = (rank == r) & (rows >= 0)
            mn[sel] = S[r]["min_nights"][rows[sel]]
            mx[sel] = np.maximum(S[r]["max_nights"][rows[sel]], 1)
            acc[sel] = np.maximum(S[r]["accommodates"][rows[sel]], 1)
        nights = np.minimum(np.maximum(mn, s_draw), mx)
        ci_ = checkout - nights
        # Val rows whose reconstructed stay starts after S3 belong to test.
        move = (split == 2) & (ci_ > sord[2])
        split[move] = 3; rank[move] = 2
        rows[move] = _rows_in(S[2], lid[move])
        sel = move & (rows >= 0)
        okc = np.zeros(n, bool); okc[sel] = S[2]["city"][rows[sel]] == city
        rows[move & ~okc] = -1
        for arr, col in ((mn, "min_nights"), (mx, "max_nights"), (acc, "accommodates")):
            arr[sel & okc] = S[2][col][rows[sel & okc]] if col != "max_nights" else np.maximum(S[2][col][rows[sel & okc]], 1)
        nights[move] = np.minimum(np.maximum(mn[move], s_draw[move]), mx[move])
        ci_ = checkout - nights
        names = ["tower_train", "rank_train", "val", "test"]
        keep = rows >= 0
        for k in range(4):
            counts[city][f"{names[k]}_drop_not_in_snapshot"] += int(((split == k) & ~keep).sum())
        # Stays must sit inside their window.
        win = np.ones(n, bool)
        win[split == 1] = (ci_ > sord[0])[split == 1] & (rd < sord[1])[split == 1]
        win[split == 2] = (ci_ > sord[1])[split == 2] & (checkout <= sord[2])[split == 2]
        win[split == 3] = (ci_ > sord[2])[split == 3]
        counts[city]["straddle_drop"] += int((keep & ~win).sum())
        keep &= win
        # Point-in-time eligibility of the positive itself.
        a = ci_ - sord[rank]
        b = checkout - sord[rank]
        for r, k in ((0, 1), (1, 2), (2, 3)):
            sel = keep & (split == k)
            idx = np.nonzero(sel)[0]
            no_cal = (b[idx] > HORIZON_BITS) | ~S[r]["has_cal"][rows[idx]]
            counts[city][f"{names[k]}_drop_no_calendar"] += int(no_cal.sum())
            keep[idx[no_cal]] = False
            idx = idx[~no_cal]
            ar = np.arange(HORIZON_BITS)
            for j in range(0, len(idx), 50_000):
                part = idx[j:j + 50_000]
                op = S[r]["open"][rows[part]]
                inwin = (ar[None, :] >= a[part, None]) & (ar[None, :] < b[part, None])
                closed = (inwin & ~op).any(1)
                counts[city][f"{names[k]}_drop_closed_in_snapshot"] += int(closed.sum())
                keep[part[closed]] = False
        guests = np.ones(n, np.int64)
        for av in np.unique(acc):
            sel = acc == av
            p = PARTY_P[:min(int(av), len(PARTY_P))]
            guests[sel] = np.searchsorted(np.cumsum(p / p.sum()), u_party[sel]) + 1
        guests = np.minimum(guests, acc)
        search = np.where(split == 0, ci_ - lead_train, sord[rank])
        nb = np.empty(n, dtype=object)
        for r in range(3):
            sel = (rank == r) & keep
            nb[sel] = S[r]["nbhd"][rows[sel]]
        place = np.full(n, pidx[("city", city)])
        for i in np.nonzero(keep & ~coin_city)[0]:
            place[i] = pidx[("nbhd", nb[i])]
        # History slice: the guest's reviews strictly before the search date, last HIST_MAX.
        h_lo = np.searchsorted(HR, rid, side="left")
        key_hi = np.searchsorted(HR, rid, side="right")
        h_hi = h_lo.copy()
        for i in np.nonzero(keep)[0]:
            seg = HD[h_lo[i]:key_hi[i]]
            h_hi[i] = h_lo[i] + np.searchsorted(seg, search[i], side="left")
        h_lo = np.maximum(h_lo, h_hi - HIST_MAX)
        for k in range(4):
            counts[city][f"{names[k]}_kept"] += int((keep & (split == k)).sum())
        idx = np.nonzero(keep)[0]
        for col, arr in (("split", split), ("reviewer_id", rid), ("listing_id", lid), ("review_date", rd),
                         ("recon_checkin", ci_), ("recon_checkout", checkout), ("recon_nights", nights),
                         ("recon_guests", guests), ("place_idx", place), ("is_city_search", coin_city),
                         ("search_date", search), ("lead_days", ci_ - search), ("snap_rank", rank),
                         ("hist_lo", h_lo), ("hist_hi", h_hi)):
            keep_cols[col].append(arr[idx])
        keep_cols["city"].append(np.full(len(idx), city, dtype=object))
        print(f"[labels] {city}: " + ", ".join(f"{names[k]} {int((keep & (split == k)).sum())}" for k in range(4)), flush=True)

    cols = {k: np.concatenate(v) for k, v in keep_cols.items()}
    # Trips: a guest's stays in one city whose review dates are within 30 days of the previous one.
    o = np.lexsort((cols["review_date"], cols["city"].astype(str), cols["reviewer_id"]))
    cols = {k: v[o] for k, v in cols.items()}
    same = np.zeros(len(o), bool)
    same[1:] = ((cols["reviewer_id"][1:] == cols["reviewer_id"][:-1]) & (cols["city"][1:] == cols["city"][:-1])
                & (cols["review_date"][1:] - cols["review_date"][:-1] <= TRIP_GAP_DAYS))
    cols["trip_id"] = np.cumsum(~same) - 1
    cols["qid"] = np.arange(len(o))
    cols["split"] = np.array(["tower_train", "rank_train", "val", "test"])[cols["split"]]
    cols["city"] = cols["city"].astype(str)
    pq.write_table(pa.table(cols), PREP / "queries.parquet", compression="zstd")
    _, sizes = np.unique(cols["trip_id"], return_counts=True)
    summary = {"counts": {c: dict(v) for c, v in counts.items()}, "cross_city_reviewers_total": n_cross_total,
               "reviewers_total": int(len(np.unique(R))), "reviews_total": int(len(R)), "trips": int(len(sizes)),
               "trips_single_review_share": float((sizes == 1).mean()), "n_places": len(places),
               "queries_total": int(len(o))}
    (PREP / "labels_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps({k: v for k, v in s.items() if k != "counts"}, indent=1))
