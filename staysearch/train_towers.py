"""Two-tower training (M2): in-batch sampled softmax with the logQ correction plus explicit hard
negatives from one of the samplers (DESIGN.md 2.7).

Training stays are the tower_train split (reviews before S1), listing features from S1. Each epoch
draws one positive per trip. Batches are drawn within a city so in-batch negatives are plausible.
The best epoch by validation recall@100 (exact KNN over the S2 eligible sets) is kept.

Usage: python -m staysearch.train_towers --name mix_l2 --sampler mix --similarity l2
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from staysearch import results
from staysearch.config import ARTIFACTS
from staysearch.data import hist_rows, load_queries, load_snapshot, stratified_sample, history_index
from staysearch.evaluate import EvalSet, evaluate
from staysearch import qfeatures as qf
from staysearch.samplers import Samplers, guard_mask, pair_key
from staysearch.towers import TwoTower, embed_listings, embed_queries, listing_inputs, query_features

SAMPLERS = ("none", "random", "geo", "price", "mined", "mix")


def reviewed_keys() -> np.ndarray:
    H = history_index()
    return np.sort(pair_key(H["reviewer_id"], H["listing_id"]))


def one_per_trip(q: dict, rng) -> np.ndarray:
    """One random stay per trip (trip_id is sorted contiguous in queries.parquet)."""
    tid = q["trip_id"]
    u = rng.random(len(tid))
    order = np.lexsort((u, tid))
    first = np.ones(len(tid), bool)
    first[1:] = tid[order][1:] != tid[order][:-1]
    return order[first]


def mine(model, q, idx, snap, emb, pos_rows, top: int = 20, chunk: int = 4096) -> np.ndarray:
    """Top-`top` same-city non-positive listings for each query under the current model (ANCE-style)."""
    out = np.zeros((len(idx), top), dtype=np.int64)
    qe, _ = embed_queries(model, q, idx, snap, emb)
    city = q["city_idx"][idx]
    E = torch.as_tensor(emb)
    with torch.no_grad():
        for c in np.unique(city):
            sel = np.nonzero(city == c)[0]
            rows = np.nonzero(snap.city == c)[0]
            Ec = E[rows]
            for j in range(0, len(sel), chunk):
                part = sel[j:j + chunk]
                sc = model.logits(torch.as_tensor(qe[part]), Ec)
                sc[torch.as_tensor(rows)[None, :] == torch.as_tensor(pos_rows[part])[:, None]] = -1e9
                k = min(top, len(rows))
                t = torch.topk(sc, k, dim=1).indices.numpy()
                out[part, :k] = rows[t]
                if k < top:
                    out[part, k:] = out[part, :1]
    return out


def train(args) -> dict:
    results.require_disk()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    s1 = load_snapshot(0)
    q = load_queries("tower_train")
    pos_rows_all = s1.rows_of(q["listing_id"])
    keep = pos_rows_all >= 0
    q = {k: v[keep] for k, v in q.items()}
    pos_rows_all = pos_rows_all[keep]
    samp = Samplers(s1.city, s1.nbhd, s1.s2[:, 2], s1.accommodates, s1.price)
    rkeys = reviewed_keys()
    # logQ: each listing's share of positives within its city (batches are within a city).
    cnt = np.bincount(pos_rows_all, minlength=s1.n).astype(np.float64) + 1.0
    city_tot = np.bincount(s1.city, weights=cnt)
    logq = torch.as_tensor(np.log(cnt / city_tot[s1.city]), dtype=torch.float32)

    model = TwoTower(args.similarity)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    s2 = load_snapshot(1)
    vq = load_queries("val")
    v_idx = stratified_sample(vq, args.val_per_city, seed=5)
    v_es = EvalSet(vq, v_idx, s2)
    best, best_state, history = -1.0, None, []
    emb = embed_listings(model, s1)
    mined = None
    for epoch in range(args.epochs):
        t0 = time.time()
        idx = one_per_trip(q, rng)
        if len(idx) > args.per_epoch:
            idx = rng.choice(idx, args.per_epoch, replace=False)
        pos = pos_rows_all[idx]
        if args.sampler in ("mined", "mix") and epoch > 0:
            mined = mine(model, q, idx, s1, emb, pos)
        # Stop-gradient history features from the current listing tower.
        hist = qf.history_block(hist_rows(q, s1, idx), emb, s1.price, s1.room_type) if args.history \
            else np.zeros((len(idx), qf.HIST_DIM), np.float32)
        base = query_features(q, idx, s1, None, drop_guests_nights=args.drop_guests_nights, use_history=False)
        base[:, 17:] = hist
        # Hard negatives for the whole epoch.
        H = args.n_hard
        if args.sampler == "none":
            hard = np.zeros((len(idx), 0), np.int64)
        elif args.sampler == "random":
            hard = samp.random(pos, H, rng)
        elif args.sampler == "geo":
            hard = samp.geo(pos, H, rng)
        elif args.sampler == "price":
            hard = samp.price(pos, H, rng)
        elif args.sampler == "mined":
            hard = samp.mined(mined, H, rng) if mined is not None else samp.random(pos, H, rng)
        else:  # mix: one of each, geo standing in for mined in the first epoch
            hard = np.concatenate([samp.random(pos, 1, rng), samp.geo(pos, 1, rng), samp.price(pos, 1, rng),
                                   samp.mined(mined, 1, rng) if mined is not None else samp.geo(pos, 1, rng)], 1)
        hmask = guard_mask(q["reviewer_id"][idx], s1.listing_id[hard], q["listing_id"][idx], rkeys) \
            if hard.shape[1] else np.zeros((len(idx), 0), bool)
        # Batches within a city.
        city = q["city_idx"][idx]
        order = np.lexsort((rng.random(len(idx)), city))
        batches = []
        for c in np.unique(city):
            sel = order[city[order] == c]
            batches += [sel[i:i + args.batch] for i in range(0, len(sel), args.batch) if len(sel[i:i + args.batch]) > 8]
        rng.shuffle(batches)
        model.train()
        tot = 0.0
        for b in batches:
            pr = pos[b]
            qe = model.query(torch.as_tensor(q["place_idx"][idx[b]], dtype=torch.long), torch.as_tensor(base[b]))
            le = model.listing(*listing_inputs(s1, pr))
            lg = model.logits(qe, le) - logq[pr][None, :]
            same = torch.as_tensor(pr[:, None] == pr[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            lg = lg.masked_fill(same, -1e9)
            if hard.shape[1]:
                hb = hard[b]
                he = model.listing(*listing_inputs(s1, hb.ravel())).view(len(b), hb.shape[1], -1)
                hl = model.logits(qe, he).masked_fill(~torch.as_tensor(hmask[b]), -1e9)
                lg = torch.cat([lg, hl], 1)
            loss = F.cross_entropy(lg, torch.arange(len(b)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        emb = embed_listings(model, s1)
        # Validation: exact KNN over the S2 eligible sets.
        e2 = embed_listings(model, s2)
        qe2, _ = embed_queries(model, vq, v_idx, s2, e2, drop_guests_nights=args.drop_guests_nights,
                               use_history=args.history)
        sc = _scorer(model, qe2, e2)
        per = evaluate(v_es, {"m": sc})["m"]
        r100 = float(per["recall@100"].mean())
        history.append({"epoch": epoch, "loss": tot / len(idx), "val_recall@100": r100,
                        "val_recall@10": float(per["recall@10"].mean()), "secs": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if r100 > best:
            best = r100
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = ARTIFACTS / "towers" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    cfg = vars(args) | {"best_val_recall@100": best, "history": history}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    results.append("m2_training", {"name": args.name, "sampler": args.sampler, "similarity": args.similarity,
                                   "history": args.history, "drop_guests_nights": args.drop_guests_nights,
                                   "epochs": args.epochs, "per_epoch": args.per_epoch,
                                   "best_val_recall@100": best, "curve": history})
    return cfg


def _scorer(model, qe: np.ndarray, emb: np.ndarray):
    """Exact KNN scorer under the model's similarity (higher is better)."""
    if model.similarity == "dot":
        return lambda j, rows: emb[rows] @ qe[j]
    return lambda j, rows: -((emb[rows] - qe[j]) ** 2).sum(1)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sampler", choices=SAMPLERS, default="mix")
    ap.add_argument("--similarity", choices=("l2", "dot"), default="l2")
    ap.add_argument("--no-history", dest="history", action="store_false")
    ap.add_argument("--drop-guests-nights", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--per-epoch", type=int, default=600_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-hard", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--val-per-city", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
