"""Listing features per snapshot, point in time.

Builds one table per snapshot rank (S1..S4, see DESIGN.md 2.5) across all cities, with:
- parsed numeric fields and missing indicators,
- the price quote carried forward from the latest snapshot at or before this one that has a quote,
- S2 cell tokens at levels 10, 12 and 14,
- amenities mapped to a fixed vocabulary of the 200 most common (built from S1 only),
- the snapshot's calendar bitset (day 0 = that city's snapshot date).

Output: data/prepared/listings_s{1..4}.parquet and data/prepared/vocab.json.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import s2sphere

from staysearch.config import CITIES, DATA_DIR

PREP = DATA_DIR / "prepared"
ROOM_TYPES = {"Entire home/apt": 0, "Private room": 1, "Shared room": 2, "Hotel room": 3}
N_AMENITIES = 200
N_PROPERTY_TYPES = 40
S2_LEVELS = (10, 12, 14)

# Numeric columns fed to the towers and the ranker, in a fixed order. Each is log1p'd when flagged,
# then standardized with S1 statistics; a missing value becomes 0 after standardizing, with an
# indicator column for the ones that are often missing.
NUMERIC = [
    ("accommodates", False), ("bedrooms", False), ("beds", False), ("bathrooms", False),
    ("log_price", False), ("minimum_nights", True), ("maximum_nights", True),
    ("availability_30", False), ("availability_60", False), ("availability_90", False),
    ("availability_365", False), ("number_of_reviews", True), ("number_of_reviews_ltm", True),
    ("number_of_reviews_l30d", True), ("estimated_occupancy_l365d", True),
    ("review_scores_rating", False), ("review_scores_accuracy", False),
    ("review_scores_cleanliness", False), ("review_scores_checkin", False),
    ("review_scores_communication", False), ("review_scores_location", False),
    ("review_scores_value", False), ("host_tenure_days", True), ("host_listings_count", True),
    ("listing_age_days", True), ("host_is_superhost", False), ("reviews_per_month", True),
]
MISSING_FLAGS = ["log_price", "review_scores_rating", "bedrooms"]


def snapshots() -> dict[str, list[str]]:
    man = json.loads((DATA_DIR / "snapshots.json").read_text())
    return {c: man[c]["snapshots"] for c in CITIES if c in man}


def _num(x):
    if x is None:
        return np.nan
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "N/A", "nan"):
        return np.nan
    if s in ("t", "f"):
        return 1.0 if s == "t" else 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _bathrooms(row):
    v = _num(row["bathrooms"])
    if np.isnan(v) and row["bathrooms_text"]:
        t = str(row["bathrooms_text"]).lower()
        if "half" in t:
            return 0.5
        try:
            return float(t.split()[0])
        except ValueError:
            return np.nan
    return v


def _s2_tokens(lat: np.ndarray, lng: np.ndarray) -> dict[int, np.ndarray]:
    out = {lv: np.zeros(len(lat), dtype=np.int64) for lv in S2_LEVELS}
    for i, (a, b) in enumerate(zip(lat, lng)):
        leaf = s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(float(a), float(b)))
        for lv in S2_LEVELS:
            # CellId ids are uint64; keep the signed bit pattern so it fits int64.
            out[lv][i] = np.int64(np.uint64(leaf.parent(lv).id()).astype(np.int64))
    return out


def _parse_amenities(s) -> list[str]:
    if not s:
        return []
    try:
        return [a.strip().lower() for a in json.loads(s)]
    except Exception:
        return []


def load_raw(city: str, snap: str):
    con = duckdb.connect()
    return con.execute(f"select * from '{DATA_DIR}/listings/{city}/{snap}.parquet'").fetch_arrow_table().to_pylist()


def build() -> None:
    PREP.mkdir(parents=True, exist_ok=True)
    snaps = snapshots()
    # Vocabularies from S1 only, so nothing later leaks into them.
    amen_counts, prop_counts = Counter(), Counter()
    for city, ss in snaps.items():
        for row in load_raw(city, ss[0]):
            amen_counts.update(set(_parse_amenities(row["amenities"])))
            prop_counts[row["property_type"]] += 1
    amen_vocab = [a for a, _ in amen_counts.most_common(N_AMENITIES)]
    prop_vocab = [p for p, _ in prop_counts.most_common(N_PROPERTY_TYPES - 1)]
    amen_idx = {a: i for i, a in enumerate(amen_vocab)}
    prop_idx = {p: i + 1 for i, p in enumerate(prop_vocab)}  # 0 = other

    tables = {r: [] for r in range(4)}
    last_price: dict[tuple[str, int], float] = {}
    for city, ss in snaps.items():
        last_price.clear()
        for rank, snap in enumerate(ss):
            snap_d = dt.date.fromisoformat(snap)
            rows = load_raw(city, snap)
            cal = pq.read_table(DATA_DIR / "calendar" / city / f"{snap}.parquet").to_pydict()
            cal_map = dict(zip(cal["listing_id"], cal["open_bits"]))
            recs = []
            for r in rows:
                try:
                    lid = int(r["id"])
                    lat, lng = float(r["latitude"]), float(r["longitude"])
                except (TypeError, ValueError):
                    continue
                if r["room_type"] not in ROOM_TYPES:
                    continue
                price = _num(r["price_quote_price_per_night"])
                if np.isnan(price):
                    price = _num(r["price"])
                if not np.isnan(price) and price > 0:
                    last_price[(city, lid)] = price
                    price_src = "this_snapshot"
                elif (city, lid) in last_price:
                    price = last_price[(city, lid)]
                    price_src = "carried"
                else:
                    price, price_src = np.nan, "missing"
                host_since = r["host_since"]
                first_review = r["first_review"]
                rec = {
                    "listing_id": lid, "city": city, "snap": snap, "rank": rank,
                    "nbhd": f"{city}|{r['neighbourhood_cleansed'] or 'unknown'}",
                    "lat": lat, "lng": lng,
                    "room_type": ROOM_TYPES[r["room_type"]],
                    "property_type": prop_idx.get(r["property_type"], 0),
                    "accommodates": _num(r["accommodates"]), "bedrooms": _num(r["bedrooms"]),
                    "beds": _num(r["beds"]), "bathrooms": _bathrooms(r),
                    "price": price, "price_src": price_src,
                    "log_price": np.log1p(price) if not np.isnan(price) else np.nan,
                    "minimum_nights": _num(r["minimum_nights"]), "maximum_nights": _num(r["maximum_nights"]),
                    "host_tenure_days": ((snap_d - dt.date.fromisoformat(host_since)).days
                                         if host_since else np.nan),
                    "listing_age_days": ((snap_d - dt.date.fromisoformat(first_review)).days
                                         if first_review else 0.0),
                    "amenities": sorted({amen_idx[a] for a in _parse_amenities(r["amenities"]) if a in amen_idx}),
                    "open_bits": cal_map.get(lid),
                }
                for col, _ in NUMERIC:
                    if col not in rec:
                        rec[col] = _num(r.get(col))
                recs.append(rec)
            if not recs:
                continue
            lat = np.array([x["lat"] for x in recs])
            lng = np.array([x["lng"] for x in recs])
            s2 = _s2_tokens(lat, lng)
            for lv in S2_LEVELS:
                for x, v in zip(recs, s2[lv]):
                    x[f"s2_{lv}"] = int(v)
            tables[rank].extend(recs)
            print(f"[features] {city} S{rank + 1} {snap}: {len(recs)} listings", flush=True)

    # Standardization statistics from S1.
    stats = {}
    for col, use_log in NUMERIC:
        v = np.array([x[col] for x in tables[0]], dtype=np.float64)
        if use_log:
            v = np.log1p(np.clip(v, 0, None))
        mu, sd = float(np.nanmean(v)), float(np.nanstd(v) + 1e-6)
        stats[col] = {"log": use_log, "mean": mu, "std": sd}
    # Categorical vocabularies over all snapshots for ids that only identify places (no outcome).
    s2_vocab = {lv: {} for lv in S2_LEVELS}
    nbhd_vocab = {}
    for rank in range(4):
        for x in tables[rank]:
            for lv in S2_LEVELS:
                s2_vocab[lv].setdefault(x[f"s2_{lv}"], len(s2_vocab[lv]) + 1)
            nbhd_vocab.setdefault(x["nbhd"], len(nbhd_vocab))
    vocab = {"amenities": amen_vocab, "property_types": prop_vocab, "numeric": NUMERIC,
             "missing_flags": MISSING_FLAGS, "stats": stats,
             "s2_sizes": {lv: len(s2_vocab[lv]) + 1 for lv in S2_LEVELS},
             "nbhds": sorted(nbhd_vocab, key=nbhd_vocab.get)}
    (PREP / "vocab.json").write_text(json.dumps(vocab))

    for rank in range(4):
        recs = tables[rank]
        num = np.zeros((len(recs), len(NUMERIC)), dtype=np.float32)
        for j, (col, use_log) in enumerate(NUMERIC):
            v = np.array([x[col] for x in recs], dtype=np.float64)
            if use_log:
                v = np.log1p(np.clip(v, 0, None))
            z = (v - stats[col]["mean"]) / stats[col]["std"]
            num[:, j] = np.nan_to_num(np.clip(z, -5, 5), nan=0.0)
        miss = np.array([[float(np.isnan(x[c])) for c in MISSING_FLAGS] for x in recs], dtype=np.float32)
        tbl = pa.table({
            "listing_id": pa.array([x["listing_id"] for x in recs], pa.int64()),
            "city": [x["city"] for x in recs], "snap": [x["snap"] for x in recs],
            "nbhd": [x["nbhd"] for x in recs],
            "nbhd_idx": pa.array([nbhd_vocab[x["nbhd"]] for x in recs], pa.int32()),
            "lat": [x["lat"] for x in recs], "lng": [x["lng"] for x in recs],
            "room_type": pa.array([x["room_type"] for x in recs], pa.int8()),
            "property_type": pa.array([x["property_type"] for x in recs], pa.int16()),
            "accommodates": pa.array([int(np.nan_to_num(x["accommodates"], nan=1)) for x in recs], pa.int16()),
            "min_nights": pa.array([int(min(np.nan_to_num(x["minimum_nights"], nan=1), 10000)) for x in recs], pa.int32()),
            "max_nights": pa.array([int(min(np.nan_to_num(x["maximum_nights"], nan=1125), 100000)) for x in recs], pa.int32()),
            "price": pa.array([x["price"] for x in recs], pa.float64()),
            "price_src": [x["price_src"] for x in recs],
            "number_of_reviews_ltm": pa.array([float(np.nan_to_num(x["number_of_reviews_ltm"])) for x in recs]),
            "review_scores_rating": pa.array([x["review_scores_rating"] for x in recs], pa.float64()),
            **{f"s2_{lv}_idx": pa.array([s2_vocab[lv][x[f"s2_{lv}"]] for x in recs], pa.int32()) for lv in S2_LEVELS},
            "num": pa.array(list(num), pa.list_(pa.float32(), len(NUMERIC))),
            "missing": pa.array(list(miss), pa.list_(pa.float32(), len(MISSING_FLAGS))),
            "amenities": pa.array([x["amenities"] for x in recs], pa.list_(pa.int16())),
            "open_bits": pa.array([x["open_bits"] for x in recs], pa.binary()),
        })
        pq.write_table(tbl, PREP / f"listings_s{rank + 1}.parquet", compression="zstd")
        print(f"[features] S{rank + 1}: {len(recs)} listings total", flush=True)


if __name__ == "__main__":
    build()
