"""Writes the serving bundle (service/BUNDLE_FORMAT.md) for the test snapshot S3 (March 2026).

The listing tower runs here as the "daily offline batch"; only the query tower ships to the service.
Guest history is computed with a cutoff of the bundle's base date (the earliest S3 date across
cities), which is at or before every test search date, so nothing after a search leaks into it.

Usage: python -m staysearch.export --tower mix_l2 --ranker dnn_hist --gbdt lambdamart_hist
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import shutil

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch

from staysearch import qfeatures as qf
from staysearch import results
from staysearch.ann import load_model
from staysearch.config import ARTIFACTS, CITIES
from staysearch.data import CITY_LIST, history_index, load_queries, load_snapshot, places, stratified_sample
from staysearch.features import PREP
from staysearch.ranking import listing_static
from staysearch.towers import embed_listings
from staysearch.updates import BUNDLE_HORIZON, TYPES, base_date

RANK_K = 100  # the K the rankers were trained with; feature 14 clamps rank at it


def iso(ordinal: int) -> str:
    return dt.date.fromordinal(int(ordinal)).isoformat()


def shifted_bits(snap, rows: np.ndarray, base_ord: int) -> np.ndarray:
    """Bitsets re-based to the bundle's base date, packed little-endian, BUNDLE_HORIZON bits."""
    out = np.zeros((len(rows), BUNDLE_HORIZON), dtype=bool)
    for city_ord in np.unique(snap.snap_ord[rows]):
        sel = np.nonzero(snap.snap_ord[rows] == city_ord)[0]
        sh = int(city_ord - base_ord)
        src = snap.open[rows[sel]] & snap.has_cal[rows[sel], None]
        if sh >= 0:
            w = min(src.shape[1], BUNDLE_HORIZON - sh)
            out[sel, sh:sh + w] = src[:, :w]
        else:
            out[sel, :src.shape[1] + sh] = src[:, -sh:]
    return np.packbits(out, axis=1, bitorder="little")


def history_blocks(guest_ids: np.ndarray, cutoff_ord: int, snap, emb) -> np.ndarray:
    H = history_index()
    out = np.zeros((len(guest_ids), qf.HIST_DIM), dtype=np.float32)
    lo = np.searchsorted(H["reviewer_id"], guest_ids, "left")
    hi = np.searchsorted(H["reviewer_id"], guest_ids, "right")
    rows_list = []
    for a, b in zip(lo, hi):
        seg = slice(a, b)
        d, l_ = H["d"][seg], H["listing_id"][seg]
        l_ = l_[d < cutoff_ord][-20:]
        r = snap.rows_of(l_)
        rows_list.append(r[r >= 0])
    return qf.history_block(rows_list, emb, snap.price, snap.room_type)


def listing_json(s, r: int, P_index: dict) -> dict:
    city = CITY_LIST[s.city[r]]
    return {"id": int(s.listing_id[r]), "city": city, "place_city": P_index[("city", city)],
            "place_nbhd": P_index.get(("nbhd", nbhd_name(s, r)), P_index[("city", city)]),
            "lat": float(s.lat[r]), "lng": float(s.lng[r]), "accommodates": int(s.accommodates[r]),
            "min_nights": int(s.min_nights[r]), "max_nights": int(max(s.max_nights[r], 1)),
            "price": float(np.nan_to_num(s.price[r], nan=0.0)), "room_type": int(s.room_type[r])}


_NB = {}


def nbhd_name(s, r) -> str:
    key = s.rank
    if key not in _NB:
        _NB[key] = pq.read_table(PREP / f"listings_s{s.rank + 1}.parquet", columns=["nbhd"]).column("nbhd").to_pylist()
    return _NB[key][r]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="mix_l2")
    ap.add_argument("--ranker", default="dnn_hist")
    ap.add_argument("--gbdt", default="lambdamart_hist")
    ap.add_argument("--out", default=str(ARTIFACTS / "bundle"))
    ap.add_argument("--n-queries", type=int, default=10_000)
    ap.add_argument("--n-updates", type=int, default=3_000_000)
    ap.add_argument("--n-parity", type=int, default=10_000)
    ap.add_argument("--nlist", type=int, default=128)
    args = ap.parse_args()
    results.require_disk(2.0)
    out = ARTIFACTS / "bundle" if args.out is None else __import__("pathlib").Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "golden").mkdir(parents=True)
    model, cfg = load_model(args.tower)
    s3, s4 = load_snapshot(2), load_snapshot(3)
    base = base_date(2)
    base_ord = base.toordinal()
    P = places()
    P_index = {(k, n): int(i) for k, n, i in zip(P["kind"], P["name"], P["idx"])}
    n = s3.n
    rows = np.arange(n)

    # Listings, embeddings, clusters, availability, static features.
    emb = embed_listings(model, s3)
    metric = cfg["similarity"]
    quant = faiss.IndexFlatIP(emb.shape[1]) if metric == "dot" else faiss.IndexFlatL2(emb.shape[1])
    km = faiss.Kmeans(emb.shape[1], args.nlist, niter=25, seed=0, spherical=False)
    km.train(emb)
    quant.add(km.centroids)
    _, assign = quant.search(emb, 1)
    with open(out / "listings.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(listing_json(s3, r, P_index)) + "\n")
    emb.astype("<f4").tofile(out / "embeddings.f32")
    km.centroids.astype("<f4").tofile(out / "centroids.f32")
    assign[:, 0].astype("<i4").tofile(out / "cluster_ids.i32")
    shifted_bits(s3, rows, base_ord).tofile(out / "availability.bin")
    static = listing_static(s3)
    static.astype("<f4").tofile(out / "ranker_static.f32")
    with open(out / "places.jsonl", "w") as f:
        for i in range(len(P["idx"])):
            f.write(json.dumps({"idx": int(P["idx"][i]), "kind": P["kind"][i], "city": P["city"][i],
                                "name": P["name"][i], "lat": float(P["lat"][i]), "lng": float(P["lng"][i]),
                                "min_lat": float(P["min_lat"][i]), "min_lng": float(P["min_lng"][i]),
                                "max_lat": float(P["max_lat"][i]), "max_lng": float(P["max_lng"][i]),
                                "area_km2": float(P["area_km2"][i])}) + "\n")

    # Guest history for every guest in the test split (the online KV map).
    q = load_queries("test")
    guests = np.unique(q["reviewer_id"])
    hb = history_blocks(guests, base_ord, s3, emb)
    keep = hb[:, 0] > 0
    guests, hb = guests[keep], hb[keep]
    guests.astype("<i8").tofile(out / "guest_history.i64")
    hb.astype("<f4").tofile(out / "guest_history.f32")
    hmap = {int(g): i for i, g in enumerate(guests)}

    # Test queries for the service-side experiments.
    idx = stratified_sample(q, args.n_queries // len(CITIES) + 1, seed=41)[:args.n_queries]
    reqs = []
    with open(out / "queries_test.jsonl", "w") as f:
        for i in idx:
            r = {"qid": int(q["qid"][i]), "place_idx": int(q["place_idx"][i]), "checkin": iso(q["recon_checkin"][i]),
                 "checkout": iso(q["recon_checkout"][i]), "guests": int(q["recon_guests"][i]),
                 "guest_id": int(q["reviewer_id"][i]), "search_date": iso(q["search_date"][i]),
                 "positive_id": int(q["listing_id"][i])}
            reqs.append(r)
            f.write(json.dumps(r) + "\n")

    # Golden query features and ONNX parity inputs, computed exactly as the service must.
    def qfeat_for(r: dict) -> np.ndarray:
        p = r["place_idx"]
        ci = dt.date.fromisoformat(r["checkin"]).toordinal()
        co = dt.date.fromisoformat(r["checkout"]).toordinal()
        sd = dt.date.fromisoformat(r["search_date"]).toordinal()
        x = np.zeros(qf.Q_FEAT_DIM, np.float32)
        x[:17] = qf.query_base(np.array([P["area_km2"][p]]), np.array([P["kind"][p] == "city"]),
                               np.array([r["guests"]]), np.array([co - ci]), np.array([ci - sd]), np.array([ci]))[0]
        g = hmap.get(r["guest_id"])
        if g is not None:
            x[17:] = hb[g]
        return x

    Q = np.stack([qfeat_for(r) for r in reqs])
    PI = np.array([r["place_idx"] for r in reqs], dtype=np.int64)
    with torch.no_grad():
        QE = model.query(torch.as_tensor(PI), torch.as_tensor(Q)).numpy()
    with open(out / "golden" / "query_features.jsonl", "w") as f:
        for r, x in list(zip(reqs, Q))[:500]:
            req = {k: r[k] for k in ("place_idx", "checkin", "checkout", "guests", "guest_id", "search_date")}
            f.write(json.dumps({"request": {**req, "bbox": None}, "place_idx": r["place_idx"], "qfeat": x.tolist()}) + "\n")
    with open(out / "golden" / "parity_queries.jsonl", "w") as f:
        for j in range(min(args.n_parity, len(reqs))):
            f.write(json.dumps({"place_idx": int(PI[j]), "qfeat": Q[j].tolist(), "qemb": QE[j].tolist()}) + "\n")
    # Golden ranker features: for some queries, a handful of listings in the place.
    rng = np.random.default_rng(3)
    with open(out / "golden" / "ranker_features.jsonl", "w") as f:
        for j in range(200):
            r = reqs[j]
            p = r["place_idx"]
            cand = np.nonzero((s3.lat >= P["min_lat"][p]) & (s3.lat <= P["max_lat"][p]) &
                              (s3.lng >= P["min_lng"][p]) & (s3.lng <= P["max_lng"][p]))[0]
            for c in rng.choice(cand, min(5, len(cand)), replace=False):
                ci = dt.date.fromisoformat(r["checkin"]).toordinal()
                co = dt.date.fromisoformat(r["checkout"]).toordinal()
                sd = dt.date.fromisoformat(r["search_date"]).toordinal()
                d = float(((emb[c].astype(np.float64) - QE[j]) ** 2).sum()) if metric == "l2" else -float(emb[c] @ QE[j])
                rank = int(rng.integers(0, 130))
                a0 = ci - int(s3.snap_ord[c])
                gb, ga = qf.gaps(s3.open[[c]], a0, a0 + (co - ci))
                cross = qf.cross_features(np.array([d]), s3.lat[[c]], s3.lng[[c]], s3.price[[c]], s3.min_nights[[c]],
                                          s3.accommodates[[c]], s3.room_type[[c]], emb[[c]], P["lat"][p], P["lng"][p],
                                          P["area_km2"][p], r["guests"], co - ci, ci - sd, Q[j, 17:], gb, ga,
                                          np.array([min(rank, RANK_K)]), RANK_K)[0]
                req = {k: r[k] for k in ("place_idx", "checkin", "checkout", "guests", "guest_id", "search_date")}
                f.write(json.dumps({"request": {**req, "bbox": None}, "qemb": QE[j].tolist(),
                                    "listing_id": int(s3.listing_id[c]), "rank": rank,
                                    "x": np.concatenate([static[c], cross]).tolist()}) + "\n")

    # Models.
    torch.onnx.export(model.query, (torch.zeros(2, dtype=torch.long), torch.zeros(2, qf.Q_FEAT_DIM)),
                      str(out / "query_tower.onnx"), input_names=["place_idx", "qfeat"], output_names=["qemb"],
                      dynamic_axes={"place_idx": {0: "B"}, "qfeat": {0: "B"}, "qemb": {0: "B"}},
                      opset_version=17, dynamo=False)
    rk = ARTIFACTS / "rankers"
    shutil.copy(rk / f"{args.ranker}.onnx", out / "ranker.onnx")
    shutil.copy(rk / f"{args.gbdt}.onnx", out / "ranker_gbdt.onnx")

    # Update stream: the first n events of the (already shuffled) S3 -> S4 diff.
    U = pq.read_table(PREP / "updates.parquet").slice(0, args.n_updates).to_pydict()
    new_ids = np.array([l for t, l in zip(U["type"], U["listing_id"]) if t == TYPES["new"]], dtype=np.int64)
    r4 = s4.rows_of(new_ids)
    e4 = embed_listings(model, s4)
    st4 = listing_static(s4)
    b4 = shifted_bits(s4, np.maximum(r4, 0), base_ord)
    new_map = {int(l): k for k, l in enumerate(new_ids)}
    n_written = {k: 0 for k in TYPES}
    inv = {v: k for k, v in TYPES.items()}
    with open(out / "updates.jsonl", "w") as f:
        for t, l, night, op, price in zip(U["type"], U["listing_id"], U["night"], U["open"], U["price"]):
            name = inv[t]
            if name == "avail":
                if not 0 <= night < BUNDLE_HORIZON:
                    continue
                ev = {"t": "avail", "id": int(l), "night": int(night), "open": bool(op)}
            elif name == "price":
                ev = {"t": "price", "id": int(l), "price": float(price)}
            elif name == "remove":
                ev = {"t": "remove", "id": int(l)}
            else:
                k = new_map[int(l)]
                if r4[k] < 0:
                    continue
                ev = {"t": "new", "listing": listing_json(s4, r4[k], P_index), "emb": e4[r4[k]].tolist(),
                      "static": st4[r4[k]].tolist(), "avail_b64": base64.b64encode(b4[k].tobytes()).decode()}
            n_written[name] += 1
            f.write(json.dumps(ev) + "\n")

    manifest = {"version": 1, "as_of": base.isoformat(), "base_date": base.isoformat(),
                "horizon_days": BUNDLE_HORIZON, "n_listings": int(n), "dim": int(emb.shape[1]),
                "similarity": metric, "n_clusters": args.nlist, "n_places": int(len(P["idx"])),
                "q_feat_dim": qf.Q_FEAT_DIM, "listing_static_dim": int(static.shape[1]),
                "ranker_feat_dim": int(static.shape[1] + qf.N_CROSS),
                "models": {"query_tower": "query_tower.onnx", "ranker": "ranker.onnx", "ranker_gbdt": "ranker_gbdt.onnx"},
                "cities": list(CITIES), "synthetic": False, "tower": args.tower, "ranker_name": args.ranker,
                "gbdt_name": args.gbdt, "snapshot": "S3", "ranker_rank_k": RANK_K, "n_guests_with_history": int(len(guests)),
                "updates_written": n_written}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
