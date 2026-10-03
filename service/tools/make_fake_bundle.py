#!/usr/bin/env python3
"""Write a small synthetic serving bundle that follows service/BUNDLE_FORMAT.md exactly.

The bundle is fake (random listings, random embeddings, random-weight models). It exists so the
Kotlin service can be built and tested before the real offline pipeline produces a bundle. Nothing
measured on it is a result.

The section "Reference feature code" below is the Python definition of the query features
(qfeat 0..16) and the 12 ranker cross features. It is deliberately plain and self-contained so the
offline pipeline can copy it; the golden files it writes are what the Kotlin tests compare against.

Usage: python make_fake_bundle.py --out DIR [--listings 5000] [--seed 7]
Needs numpy, torch and onnx.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------------------------
# Constants of the contract
# ---------------------------------------------------------------------------------------------

DIM = 64
Q_FEAT_DIM = 87
LISTING_STATIC_DIM = 28
RANKER_FEAT_DIM = LISTING_STATIC_DIM + 12
HORIZON_DAYS = 448
GUEST_BLOCK = 70  # has_history, hist_log_price_mean_norm, 4 room-type mix, 64 hist_vec
KM_PER_DEG_LAT = 111.32
EARTH_RADIUS_KM = 6371.0
LEAD_BUCKETS = [(-10**9, 1), (2, 7), (8, 14), (15, 30), (31, 60), (61, 10**9)]


# ---------------------------------------------------------------------------------------------
# Reference feature code (the offline pipeline must compute exactly this)
# ---------------------------------------------------------------------------------------------

def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km on a sphere of radius 6371.0 km."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bbox_area_km2(min_lat, min_lng, max_lat, max_lng) -> float:
    """Planar area of a lat/lng box: latitude at 111.32 km/deg, longitude scaled by cos(centre lat)."""
    centre_lat = (min_lat + max_lat) / 2
    h = (max_lat - min_lat) * KM_PER_DEG_LAT
    w = (max_lng - min_lng) * KM_PER_DEG_LAT * math.cos(math.radians(centre_lat))
    return abs(h * w)


def resolve_search_area(req: dict, places: list[dict]) -> dict:
    """Turns a request into the search area used by every feature.

    A place search (place_idx set, no bbox) uses the place: its bbox as the map box, its (lat, lng)
    as the centre, its area_km2, and is_city = (kind == "city").
    A bbox search uses the bbox: its centre and planar area, is_city = 0, and as tower place the
    city place whose (lat, lng) is nearest (haversine) to the bbox centre (ties: lowest index).
    If both are given, the explicit bbox wins for the box, centre and area, place_idx is kept as the
    tower place, and is_city = 0.
    """
    bbox = req.get("bbox")
    if bbox is None:
        p = places[req["place_idx"]]
        return {
            "place_idx": p["idx"],
            "is_city": 1.0 if p["kind"] == "city" else 0.0,
            "area_km2": p["area_km2"],
            "centre_lat": p["lat"],
            "centre_lng": p["lng"],
            "box": (p["min_lat"], p["min_lng"], p["max_lat"], p["max_lng"]),
        }
    box = (bbox["min_lat"], bbox["min_lng"], bbox["max_lat"], bbox["max_lng"])
    clat, clng = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if req.get("place_idx") is not None:
        place_idx = req["place_idx"]
    else:
        best, place_idx = float("inf"), -1
        for p in places:
            if p["kind"] != "city":
                continue
            d = haversine_km(clat, clng, p["lat"], p["lng"])
            if d < best:
                best, place_idx = d, p["idx"]
    return {
        "place_idx": place_idx,
        "is_city": 0.0,
        "area_km2": bbox_area_km2(*box),
        "centre_lat": clat,
        "centre_lng": clng,
        "box": box,
    }


def guest_block(guest_id, guest_ids: np.ndarray, guest_feats: np.ndarray) -> np.ndarray:
    """qfeat 17..86: the guest's row of guest_history.f32, or 70 zeros if absent."""
    if guest_id is None:
        return np.zeros(GUEST_BLOCK, dtype=np.float32)
    i = int(np.searchsorted(guest_ids, guest_id))
    if i < len(guest_ids) and guest_ids[i] == guest_id:
        return guest_feats[i].astype(np.float32)
    return np.zeros(GUEST_BLOCK, dtype=np.float32)


def lead_days(req: dict, as_of: str) -> int:
    """checkin minus search date in days; the search date defaults to manifest.as_of."""
    sd = parse_date(req.get("search_date") or as_of)
    return (parse_date(req["checkin"]) - sd).days


def query_features(req: dict, places, guest_ids, guest_feats, as_of: str) -> tuple[int, np.ndarray]:
    """Returns (tower place_idx, qfeat[87]) for a request."""
    area = resolve_search_area(req, places)
    checkin, checkout = parse_date(req["checkin"]), parse_date(req["checkout"])
    nights = (checkout - checkin).days
    lead = lead_days(req, as_of)
    q = np.zeros(Q_FEAT_DIM, dtype=np.float64)
    q[0] = math.log1p(area["area_km2"]) / 10.0
    q[1] = area["is_city"]
    q[2] = req["guests"] / 16.0
    q[3] = math.log1p(nights) / math.log(31.0)
    for b, (lo, hi) in enumerate(LEAD_BUCKETS):  # lead <= 1 (negative included) is bucket 0
        if lo <= lead <= hi:
            q[4 + b] = 1.0
    q[10 + checkin.weekday()] = 1.0  # Monday = 0
    q[17:] = guest_block(req.get("guest_id"), guest_ids, guest_feats)
    return area["place_idx"], q.astype(np.float32)


def retrieval_distance(qemb: np.ndarray, emb: np.ndarray, similarity: str) -> float:
    """Smaller is better: squared L2 under "l2", minus the dot product under "dot"."""
    q, e = qemb.astype(np.float64), emb.astype(np.float64)
    if similarity == "l2":
        return float(np.sum((q - e) ** 2))
    return float(-np.dot(q, e))


def ranker_cross_features(req, listing, emb, qemb, places, guest_ids, guest_feats, as_of,
                          similarity) -> np.ndarray:
    """The 12 cross features appended to ranker_static[i]. `listing` is a listings.jsonl dict."""
    area = resolve_search_area(req, places)
    checkin, checkout = parse_date(req["checkin"]), parse_date(req["checkout"])
    nights = (checkout - checkin).days
    guests = req["guests"]
    lead = max(0, lead_days(req, as_of))  # a negative lead is clamped to 0 for feature 8
    g = guest_block(req.get("guest_id"), guest_ids, guest_feats).astype(np.float64)
    has_hist = g[0] > 0.5
    price = listing["price"]
    x = np.zeros(12, dtype=np.float64)
    x[0] = retrieval_distance(qemb, emb, similarity)
    x[1] = haversine_km(listing["lat"], listing["lng"], area["centre_lat"], area["centre_lng"]) \
        / math.sqrt(1.0 + area["area_km2"])
    x[2] = (math.log1p(price) - math.log(guests)) if price > 0 else 0.0
    x[3] = math.log1p(max(0, nights - listing["min_nights"]))
    x[4] = guests / listing["accommodates"]
    x[5] = float(np.dot(g[6:], emb.astype(np.float64))) if has_hist else 0.0
    x[6] = ((math.log1p(price) - 5.0) / 1.0 - g[1]) if (has_hist and price > 0) else 0.0
    x[7] = 1.0 if has_hist else 0.0
    x[8] = math.log1p(lead) / math.log(366.0)
    x[9] = math.log1p(nights) / math.log(31.0)
    x[10] = guests / 16.0
    x[11] = g[2 + listing["room_type"]] if has_hist else 0.0
    return x.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Models (random weights)
# ---------------------------------------------------------------------------------------------

class QueryTower(nn.Module):
    def __init__(self, n_places: int):
        super().__init__()
        self.place = nn.Embedding(n_places, 16)
        self.mlp = nn.Sequential(nn.Linear(16 + Q_FEAT_DIM, 64), nn.ReLU(), nn.Linear(64, DIM))

    def forward(self, place_idx, qfeat):
        return self.mlp(torch.cat([self.place(place_idx), qfeat], dim=1))


class Ranker(nn.Module):
    def __init__(self, squeeze: bool):
        super().__init__()
        self.squeeze = squeeze
        self.mlp = nn.Sequential(nn.Linear(RANKER_FEAT_DIM, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        s = self.mlp(x)
        return s.squeeze(1) if self.squeeze else s


def export_onnx(model, args, path, input_names, output_names):
    dyn = {n: {0: "batch"} for n in input_names + output_names}
    torch.onnx.export(model, args, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dyn, opset_version=17, dynamo=False)


# ---------------------------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------------------------

def make_places(rng):
    cities = [("fakeville", 47.60, -122.33), ("mockton", 32.72, -117.16)]
    places = []
    for name, lat, lng in cities:
        places.append({"kind": "city", "city": name, "name": name, "lat": lat, "lng": lng})
    for ci, (name, lat, lng) in enumerate(cities):
        for j in range(5):
            places.append({"kind": "nbhd", "city": name, "name": f"{name}-n{j}",
                           "lat": lat + rng.uniform(-0.08, 0.08), "lng": lng + rng.uniform(-0.1, 0.1)})
    for i, p in enumerate(places):
        p["idx"] = i
    return cities, places


def random_calendar(rng) -> np.ndarray:
    """Alternating open and closed runs, roughly half the nights open."""
    bits = np.zeros(HORIZON_DAYS, dtype=bool)
    d, open_ = 0, rng.random() < 0.5
    while d < HORIZON_DAYS:
        n = int(rng.geometric(1 / 6))
        bits[d:d + n] = open_
        d += n
        open_ = not open_
    return bits


def pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


def make_listing(rng, lid, city_idx, cities, places, n_cities):
    name, clat, clng = cities[city_idx]
    nbhd = n_cities + city_idx * 5 + int(rng.integers(0, 5))
    np_ = places[nbhd]
    min_n = int(rng.choice([1, 1, 1, 2, 2, 3, 5, 7]))
    return {
        "id": int(lid), "city": name, "place_city": city_idx, "place_nbhd": nbhd,
        "lat": float(np_["lat"] + rng.normal(0, 0.015)), "lng": float(np_["lng"] + rng.normal(0, 0.02)),
        "accommodates": int(rng.integers(1, 9)), "min_nights": min_n,
        "max_nights": int(rng.choice([14, 30, 60, 365, 1125])),
        "price": 0.0 if rng.random() < 0.05 else float(round(rng.lognormal(5.0, 0.5), 2)),
        "room_type": int(rng.choice([0, 0, 0, 1, 1, 2, 3])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--listings", type=int, default=5000)
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--guests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    out = args.out
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "golden"))

    as_of, base_date = "2026-03-20", "2026-03-01"
    base = parse_date(base_date)
    similarity = "l2"
    cities, places = make_places(rng)
    n_cities = len(cities)
    N = args.listings

    listings = [make_listing(rng, 10_000 + 3 * i, i % n_cities, cities, places, n_cities) for i in range(N)]

    # Places: bbox and area from their listings, as the contract says.
    for p in places:
        if p["kind"] == "city":
            members = [l for l in listings if l["place_city"] == p["idx"]]
        else:
            members = [l for l in listings if l["place_nbhd"] == p["idx"]]
        lats = [l["lat"] for l in members]
        lngs = [l["lng"] for l in members]
        p.update(min_lat=min(lats), min_lng=min(lngs), max_lat=max(lats), max_lng=max(lngs))
        p["lat"], p["lng"] = float(np.mean(lats)), float(np.mean(lngs))
        p["area_km2"] = bbox_area_km2(p["min_lat"], p["min_lng"], p["max_lat"], p["max_lng"])

    # Embeddings with some cluster structure, then a few Lloyd iterations for the centroids.
    true_centres = rng.normal(0, 1, size=(args.clusters, DIM)).astype(np.float32)
    emb = (true_centres[rng.integers(0, args.clusters, N)] + rng.normal(0, 0.6, size=(N, DIM))).astype(np.float32)
    cent = emb[rng.choice(N, args.clusters, replace=False)].copy()
    for _ in range(10):
        d2 = ((emb[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(1)
        for c in range(args.clusters):
            m = assign == c
            if m.any():
                cent[c] = emb[m].mean(0)
    d2 = ((emb[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    cluster_ids = d2.argmin(1).astype(np.int32)

    cal = np.stack([random_calendar(rng) for _ in range(N)])
    static = rng.normal(0, 1, size=(N, LISTING_STATIC_DIM)).astype(np.float32)

    # Guest history: sorted ids, rows of 70 floats.
    gids = np.sort(rng.choice(np.arange(1, 10**7), args.guests, replace=False)).astype(np.int64)
    gfeat = np.zeros((args.guests, GUEST_BLOCK), dtype=np.float32)
    gfeat[:, 0] = 1.0
    gfeat[:, 1] = rng.normal(0, 0.5, args.guests)
    mix = rng.dirichlet(np.ones(4), args.guests)
    gfeat[:, 2:6] = mix
    gfeat[:, 6:] = rng.normal(0, 0.3, size=(args.guests, DIM))

    # Write per-listing files.
    with open(os.path.join(out, "listings.jsonl"), "w") as f:
        for l in listings:
            f.write(json.dumps(l) + "\n")
    emb.astype("<f4").tofile(os.path.join(out, "embeddings.f32"))
    cluster_ids.astype("<i4").tofile(os.path.join(out, "cluster_ids.i32"))
    cent.astype("<f4").tofile(os.path.join(out, "centroids.f32"))
    with open(os.path.join(out, "availability.bin"), "wb") as f:
        for i in range(N):
            f.write(pack_bits(cal[i]))
    static.astype("<f4").tofile(os.path.join(out, "ranker_static.f32"))
    with open(os.path.join(out, "places.jsonl"), "w") as f:
        for p in places:
            f.write(json.dumps({k: p[k] for k in ["idx", "kind", "city", "name", "lat", "lng", "min_lat",
                                                  "min_lng", "max_lat", "max_lng", "area_km2"]}) + "\n")
    gids.astype("<i8").tofile(os.path.join(out, "guest_history.i64"))
    gfeat.astype("<f4").tofile(os.path.join(out, "guest_history.f32"))

    # Models.
    qt = QueryTower(len(places)).eval()
    export_onnx(qt, (torch.zeros(2, dtype=torch.long), torch.zeros(2, Q_FEAT_DIM)),
                os.path.join(out, "query_tower.onnx"), ["place_idx", "qfeat"], ["qemb"])
    dnn = Ranker(squeeze=True).eval()
    export_onnx(dnn, (torch.zeros(2, RANKER_FEAT_DIM),), os.path.join(out, "ranker.onnx"), ["x"], ["score"])
    # Stand-in for the LightGBM export: same interface, [B, 1] output to exercise that shape.
    gbdt = Ranker(squeeze=False).eval()
    export_onnx(gbdt, (torch.zeros(2, RANKER_FEAT_DIM),), os.path.join(out, "ranker_gbdt.onnx"), ["x"], ["score"])

    # Queries: each has a positive listing that is open for the window and fits the request.
    queries = []
    qid = 0
    while len(queries) < args.queries:
        i = int(rng.integers(0, N))
        l = listings[i]
        nights = int(max(l["min_nights"], rng.integers(1, 8)))
        if nights > l["max_nights"]:
            continue
        start = int(rng.integers(19, HORIZON_DAYS - 40))
        if not cal[i, start:start + nights].all():
            continue
        checkin = base + dt.timedelta(days=start)
        lead = int(rng.choice([0, 1, 3, 10, 20, 45, 90]))
        guests = int(rng.integers(1, l["accommodates"] + 1))
        place = l["place_nbhd"] if rng.random() < 0.5 else l["place_city"]
        gid = int(gids[rng.integers(0, len(gids))]) if rng.random() < 0.3 else None
        queries.append({
            "qid": qid, "place_idx": place, "checkin": checkin.isoformat(),
            "checkout": (checkin + dt.timedelta(days=nights)).isoformat(), "guests": guests,
            "guest_id": gid, "search_date": (checkin - dt.timedelta(days=lead)).isoformat(),
            "positive_id": l["id"],
        })
        qid += 1
    with open(os.path.join(out, "queries_test.jsonl"), "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # Updates: mostly availability flips, some price changes, a few removals and new listings.
    next_id = 10_000 + 3 * N + 1
    with open(os.path.join(out, "updates.jsonl"), "w") as f:
        for _ in range(args.updates):
            r = rng.random()
            if r < 0.93:
                ev = {"t": "avail", "id": listings[int(rng.integers(0, N))]["id"],
                      "night": int(rng.integers(0, HORIZON_DAYS)), "open": bool(rng.random() < 0.5)}
            elif r < 0.985:
                ev = {"t": "price", "id": listings[int(rng.integers(0, N))]["id"],
                      "price": float(round(rng.lognormal(5.0, 0.5), 2))}
            elif r < 0.99:
                ev = {"t": "remove", "id": listings[int(rng.integers(0, N))]["id"]}
            else:
                nl = make_listing(rng, next_id, int(rng.integers(0, n_cities)), cities, places, n_cities)
                next_id += 1
                ev = {"t": "new", "listing": nl,
                      "emb": [float(v) for v in (true_centres[rng.integers(0, args.clusters)]
                                                 + rng.normal(0, 0.6, DIM)).astype(np.float32)],
                      "static": [float(v) for v in rng.normal(0, 1, LISTING_STATIC_DIM).astype(np.float32)],
                      "avail_b64": base64.b64encode(pack_bits(random_calendar(rng))).decode("ascii")}
            f.write(json.dumps(ev) + "\n")

    # Golden files.
    golden_reqs = []
    for q in queries[:60]:
        golden_reqs.append({k: q[k] for k in ["place_idx", "checkin", "checkout", "guests", "guest_id",
                                              "search_date"]})
    for j in range(20):  # bbox searches, some with a guest, some without a search date
        p = places[int(rng.integers(0, len(places)))]
        lat0 = rng.uniform(p["min_lat"], p["max_lat"])
        lng0 = rng.uniform(p["min_lng"], p["max_lng"])
        ci = base + dt.timedelta(days=int(rng.integers(19, 300)))
        golden_reqs.append({
            "place_idx": None,
            "bbox": {"min_lat": lat0 - 0.03, "min_lng": lng0 - 0.04, "max_lat": lat0 + 0.03, "max_lng": lng0 + 0.04},
            "checkin": ci.isoformat(), "checkout": (ci + dt.timedelta(days=int(rng.integers(1, 15)))).isoformat(),
            "guests": int(rng.integers(1, 7)), "guest_id": int(gids[j]) if j % 2 == 0 else None,
            "search_date": None if j % 3 == 0 else (ci - dt.timedelta(days=int(rng.integers(0, 120)))).isoformat(),
        })

    with open(os.path.join(out, "golden", "query_features.jsonl"), "w") as fq, \
            open(os.path.join(out, "golden", "ranker_features.jsonl"), "w") as fr, \
            open(os.path.join(out, "golden", "parity_queries.jsonl"), "w") as fp:
        for req in golden_reqs:
            pidx, qf = query_features(req, places, gids, gfeat, as_of)
            fq.write(json.dumps({"request": req, "place_idx": pidx, "qfeat": [float(v) for v in qf]}) + "\n")
            with torch.no_grad():
                qemb = qt(torch.tensor([pidx]), torch.tensor(qf[None, :]))[0].numpy()
            fp.write(json.dumps({"place_idx": pidx, "qfeat": [float(v) for v in qf],
                                 "qemb": [float(v) for v in qemb]}) + "\n")
            for i in rng.choice(N, 3, replace=False):
                l = listings[int(i)]
                if req["guests"] > l["accommodates"]:
                    continue  # the ranker only ever sees listings that fit the party
                cross = ranker_cross_features(req, l, emb[i], qemb, places, gids, gfeat, as_of, similarity)
                x = np.concatenate([static[i], cross])
                fr.write(json.dumps({"request": req, "qemb": [float(v) for v in qemb], "listing_id": l["id"],
                                     "x": [float(v) for v in x]}) + "\n")
        # Extra parity rows over random inputs (any place, random qfeat).
        for _ in range(200):
            pidx = int(rng.integers(0, len(places)))
            qf = rng.normal(0, 1, Q_FEAT_DIM).astype(np.float32)
            with torch.no_grad():
                qemb = qt(torch.tensor([pidx]), torch.tensor(qf[None, :]))[0].numpy()
            fp.write(json.dumps({"place_idx": pidx, "qfeat": [float(v) for v in qf],
                                 "qemb": [float(v) for v in qemb]}) + "\n")

    manifest = {
        "version": 1, "as_of": as_of, "base_date": base_date, "horizon_days": HORIZON_DAYS,
        "n_listings": N, "dim": DIM, "similarity": similarity, "n_clusters": args.clusters,
        "n_places": len(places), "q_feat_dim": Q_FEAT_DIM, "listing_static_dim": LISTING_STATIC_DIM,
        "ranker_feat_dim": RANKER_FEAT_DIM,
        "models": {"query_tower": "query_tower.onnx", "ranker": "ranker.onnx", "ranker_gbdt": "ranker_gbdt.onnx"},
        "cities": [c[0] for c in cities],
        "synthetic": True,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote fake bundle to {out}: {N} listings, {len(places)} places, {len(queries)} queries")


if __name__ == "__main__":
    main()
