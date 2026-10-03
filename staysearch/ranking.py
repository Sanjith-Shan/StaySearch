"""M4 ranking: LightGBM LambdaMART, a pairwise DNN, and the DNN with guest-history features, over
the top K the two-tower retrieves from each query's point-in-time eligible set (exp6), plus the
replay recall metric of 2601.06873 section 5.2 (exp7).

Rankers train on the rank_train split (stays after S1, eligible sets from S1's calendar), which the
towers never saw, select on val (S2), and report on test (S3).

Ranker input = listing_static (num, missing flags, room type one-hot, amenity count) followed by the
12 cross features of qfeatures.cross_features, which the Kotlin service computes identically.
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

from staysearch import qfeatures as qf
from staysearch import results
from staysearch.ann import load_model
from staysearch.config import ARTIFACTS
from staysearch.data import Snapshot, hist_rows, load_queries, load_snapshot, places, stratified_sample
from staysearch.evaluate import SPLIT_RANK, EvalSet, bootstrap_ci
from staysearch.towers import embed_listings, query_features

HIST_CROSS = [5, 6, 7, 11]  # cross-feature offsets that carry guest history


def listing_static(s: Snapshot) -> np.ndarray:
    room = np.eye(4, dtype=np.float32)[s.room_type]
    amen = (s.amen.sum(1, keepdims=True) / 50.0).astype(np.float32)
    return np.concatenate([s.num, s.missing, room, amen], 1).astype(np.float32)


class Candidates:
    """Top-K retrieval candidates and ranker features for a set of queries."""

    def __init__(self, split: str, idx_per_city: int, K: int, tower: str, seed: int, include_positive: bool):
        model, cfg = load_model(tower)
        self.split = split
        s = load_snapshot(SPLIT_RANK[split])
        q = load_queries(split)
        idx = stratified_sample(q, idx_per_city, seed=seed)
        self.es = EvalSet(q, idx, s)
        emb = embed_listings(model, s)
        qfeat = query_features(q, idx, s, emb)
        with torch.no_grad():
            qe = model.query(torch.as_tensor(q["place_idx"][idx], dtype=torch.long), torch.as_tensor(qfeat)).numpy()
        static = listing_static(s)
        P = places()
        self.K = K
        self.X, self.y, self.group, self.qpos, self.in_k, self.rows_list = [], [], [], [], [], []
        self.full = []  # (rows, dist) of the full eligible set, for replay recall
        for j in range(len(idx)):
            i = idx[j]
            rows = self.es.eligible(j)
            if cfg["similarity"] == "dot":
                d = -(emb[rows] @ qe[j])
            else:
                d = ((emb[rows] - qe[j]) ** 2).sum(1)
            self.full.append((rows, d))
            top = np.argsort(d, kind="stable")[:K]
            cand, cd = rows[top], d[top]
            pos = self.es.pos_rows[j]
            hit = pos in cand
            self.in_k.append(hit)
            if include_positive and not hit:
                cand = np.append(cand[:-1] if len(cand) == K else cand, pos)
                cd = np.append(cd[:-1] if len(cd) == K else cd, d[rows == pos][0])
            p = q["place_idx"][i]
            cross = qf.cross_features(cd, s.lat[cand], s.lng[cand], s.price[cand], s.min_nights[cand],
                                      s.accommodates[cand], s.room_type[cand], emb[cand], P["lat"][p], P["lng"][p],
                                      P["area_km2"][p], int(q["recon_guests"][i]), int(q["recon_nights"][i]),
                                      int(q["lead_days"][i]), qfeat[j, 17:])
            self.X.append(np.concatenate([static[cand], cross], 1))
            self.y.append((cand == pos).astype(np.float32))
            self.group.append(len(cand))
            self.rows_list.append(cand)
        self.static_dim = static.shape[1]
        self.has_hist = qfeat[:, 17] > 0
        self.qfeat, self.emb, self.static, self.snap, self.qe = qfeat, emb, static, s, qe
        self.in_k = np.array(self.in_k)

    def matrix(self, history: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.concatenate(self.X)
        if not history:
            X = X.copy()
            X[:, [self.static_dim + c for c in HIST_CROSS]] = 0.0
        return X, np.concatenate(self.y), np.array(self.group)


class PairwiseDNN(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dnn(X, y, group, Xv, yv, gv, epochs=30, seed=0, threads=4) -> PairwiseDNN:
    """Pairwise logistic loss: the reviewed stay against every other candidate in its list (KDD'19)."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = PairwiseDNN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    offs = np.concatenate([[0], np.cumsum(group)])
    lists = [(offs[g], offs[g + 1]) for g in range(len(group)) if y[offs[g]:offs[g + 1]].sum() > 0]
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    rng = np.random.default_rng(seed)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(lists))
        for b in range(0, len(order), 64):
            idx = np.concatenate([np.arange(*lists[k]) for k in order[b:b + 64]])
            lens = [lists[k][1] - lists[k][0] for k in order[b:b + 64]]
            s = model(Xt[idx])
            yy = yt[idx]
            seg = torch.repeat_interleave(torch.arange(len(lens)), torch.as_tensor(lens))
            pos_s = torch.zeros(len(lens)).index_add_(0, seg, s * yy)
            diff = pos_s[seg] - s
            loss = (torch.nn.functional.softplus(-diff) * (1 - yy)).sum() / (1 - yy).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = ndcg_of(model_scores(model, Xv), yv, gv)["ndcg@10"].mean()
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def model_scores(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(X[i:i + 200_000])).numpy() for i in range(0, len(X), 200_000)])


def ndcg_of(scores, y, group) -> dict:
    """Per-list NDCG@10 and MRR of the reviewed stay (0 when it is not in the list); ties broken by
    expected position."""
    offs = np.concatenate([[0], np.cumsum(group)])
    nd, mrr = np.zeros(len(group)), np.zeros(len(group))
    for g in range(len(group)):
        s, yy = scores[offs[g]:offs[g + 1]], y[offs[g]:offs[g + 1]]
        if yy.sum() == 0:
            continue
        sp = s[yy > 0][0]
        gt, ties = int((s > sp).sum()), int((s == sp).sum()) - 1
        r = np.arange(gt + 1, gt + ties + 2)
        nd[g] = np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()
        mrr[g] = (1 / r).mean()
    return {"ndcg@10": nd, "mrr": mrr}


def baseline_scores(c: Candidates) -> dict:
    s = c.snap
    rows = np.concatenate(c.rows_list)
    X = np.concatenate(c.X)
    rng = np.random.default_rng(0)
    return {
        "retrieval_order": -X[:, c.static_dim + 0],
        "popularity_ltm": s.ltm[rows],
        "price_ascending": -np.nan_to_num(s.price[rows], nan=1e9),
        "review_score": np.nan_to_num(s.rating[rows], nan=0.0),
        "random": rng.random(len(rows)),
    }


def export_dnn(model: PairwiseDNN, d_in: int, path) -> None:
    model.eval()
    torch.onnx.export(model, torch.zeros(2, d_in), str(path), input_names=["x"], output_names=["score"],
                      dynamic_axes={"x": {0: "B"}, "score": {0: "B"}}, opset_version=17, dynamo=False)


def export_lgbm(booster: lgb.Booster, d_in: int, path) -> None:
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_lightgbm(booster, initial_types=[("x", FloatTensorType([None, d_in]))], target_opset=17,
                           zipmap=False)
    # Rename the regressor's output to the contract's name.
    for o in onx.graph.output:
        if o.name == "variable":
            o.name = "score"
    for node in onx.graph.node:
        node.output[:] = ["score" if x == "variable" else x for x in node.output]
    path.write_bytes(onx.SerializeToString())


def report(name, scores, c: Candidates, split, extra=None):
    X, y, g = c.matrix(True)
    m = ndcg_of(scores, y, g)
    slices = {"all": np.ones(len(g), bool), "with_history": c.has_hist, "no_history": ~c.has_hist,
              "positive_in_topK": c.in_k}
    for sl, mask in slices.items():
        row = {"exp": "exp6", "split": split, "model": name, "slice": sl, "K": c.K, "n_queries": int(mask.sum()),
               "positive_in_topK_share": float(c.in_k[mask].mean()) if mask.any() else None}
        for k in ("ndcg@10", "mrr"):
            mean, lo, hi = bootstrap_ci(m[k][mask])
            row[k], row[f"{k}_ci95"] = mean, [lo, hi]
        results.append("m4_ranking", {**row, **(extra or {})})
    return m


def replay_recall(c: Candidates, ranker_fn, retrieval: dict, history: bool) -> dict:
    """exp7: recall@100 of each retrieval method against the ranker's top 100 over the FULL eligible set."""
    s, P = c.snap, places()
    q, idx = c.es.q, c.es.idx
    out = {k: [] for k in retrieval}
    for j, (rows, d) in enumerate(c.full):
        i = idx[j]
        p = q["place_idx"][i]
        cross = qf.cross_features(d, s.lat[rows], s.lng[rows], s.price[rows], s.min_nights[rows],
                                  s.accommodates[rows], s.room_type[rows], c.emb[rows], P["lat"][p], P["lng"][p],
                                  P["area_km2"][p], int(q["recon_guests"][i]), int(q["recon_nights"][i]),
                                  int(q["lead_days"][i]), c.qfeat[j, 17:])
        X = np.concatenate([c.static[rows], cross], 1)
        if not history:
            X[:, [c.static_dim + h for h in HIST_CROSS]] = 0.0
        truth = rows[np.argsort(-ranker_fn(X), kind="stable")[:100]]
        denom = min(100, len(rows))
        for name, fn in retrieval.items():
            got = rows[np.argsort(-fn(j, rows, d), kind="stable")[:100]]
            out[name].append(len(np.intersect1d(got, truth)) / denom)
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="mix_l2")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--train-per-city", type=int, default=2500)
    ap.add_argument("--eval-per-city", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results.reset("m4_ranking")
    results.reset("m4_replay_recall")
    tr = Candidates("rank_train", args.train_per_city, args.K, args.tower, seed=31, include_positive=True)
    va = Candidates("val", args.eval_per_city // 2, args.K, args.tower, seed=32, include_positive=False)
    te = Candidates("test", args.eval_per_city, args.K, args.tower, seed=33, include_positive=False)
    print("candidates built", flush=True)
    out_dir = ARTIFACTS / "rankers"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for split_c in (va, te):
        for name, sc in baseline_scores(split_c).items():
            if split_c is te:
                report(name, sc, te, "test")
    for history in (False, True):
        tag = "hist" if history else "nohist"
        X, y, g = tr.matrix(history)
        Xv, yv, gv = va.matrix(history)
        Xt, yt, gt = te.matrix(history)
        # LambdaMART
        dtr = lgb.Dataset(X, y, group=g)
        dva = lgb.Dataset(Xv, yv, group=gv)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": 0.05,
                  "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "num_threads": args.threads, "verbose": -1, "lambdarank_truncation_level": 20}
        booster = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        report(f"lambdamart_{tag}", booster.predict(Xt), te, "test", {"best_iteration": booster.best_iteration})
        models[f"lambdamart_{tag}"] = (lambda B: (lambda Z: B.predict(Z)))(booster)
        booster.save_model(str(out_dir / f"lambdamart_{tag}.txt"))
        export_lgbm(booster, X.shape[1], out_dir / f"lambdamart_{tag}.onnx")
        # Pairwise DNN (inputs standardized with training statistics folded into a first layer)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        dnn = train_dnn((X - mu) / sd, y, g, (Xv - mu) / sd, yv, gv, threads=args.threads)
        wrapped = nn.Sequential(Standardize(mu, sd), dnn)
        report(f"dnn_{tag}", model_scores(wrapped, Xt.astype(np.float32)), te, "test")
        models[f"dnn_{tag}"] = (lambda W: (lambda Z: model_scores(W, Z.astype(np.float32))))(wrapped)
        export_dnn(wrapped, X.shape[1], out_dir / f"dnn_{tag}.onnx")
        torch.save(wrapped.state_dict(), out_dir / f"dnn_{tag}.pt")
        print(f"trained {tag}", flush=True)
    # exp7 replay recall against each ranker's top 100 over the full eligible set.
    from staysearch.baselines import train_linear
    lin = train_linear(te.snap)
    s = te.snap

    def geo(j, rows, d):
        p = te.es.q["place_idx"][te.es.idx[j]]
        from staysearch.qfeatures import haversine_km
        return -haversine_km(s.lat[rows], s.lng[rows], places()["lat"][p], places()["lng"][p])

    retrieval = {"two_tower": lambda j, rows, d: -d, "geo_distance": geo,
                 "linear_query_independent": lambda j, rows, d: lin[rows],
                 "popularity_ltm": lambda j, rows, d: s.ltm[rows]}
    for rname in ("lambdamart_hist", "dnn_hist"):
        rr = replay_recall(te, models[rname], retrieval, history=True)
        for name, v in rr.items():
            mean, lo, hi = bootstrap_ci(v)
            results.append("m4_replay_recall", {"exp": "exp7", "split": "test", "ground_truth_ranker": rname,
                                                "retrieval": name, "replay_recall@100": mean,
                                                "ci95": [lo, hi], "n_queries": len(v),
                                                "paper_note": "2601.06873 Table 1 replay recall@100 is not like for like"})
        print(rname, {k: round(float(v.mean()), 3) for k, v in rr.items()}, flush=True)
    (out_dir / "ranker_meta.json").write_text(json.dumps({"static_dim": te.static_dim, "n_cross": qf.N_CROSS,
                                                          "K": args.K, "tower": args.tower}))


class Standardize(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mu) / self.sd


if __name__ == "__main__":
    main()
