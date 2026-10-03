"""exp3 and the FAISS side of exp4: clustering balance, IVF nprobe recall, HNSW efSearch recall,
memory per vector and removal support, for a dot-product and a Euclidean model.

Reproduces the shape of 2601.06873 Table 2 and Figure 8 on our own embeddings. Recall@100 vs exact
is reported two ways:
- unfiltered: global top 100 over all listings in the snapshot (the paper's table compares against
  true KNN; its filtering setup is not specified, so this is the closest like-for-like);
- filtered: top 100 within each query's point-in-time eligible set, IVF restricted to the probed
  clusters (what the Lucene IVF path does) and HNSW searched with an ID selector.

Usage: python -m staysearch.ann --l2 mix_l2 --dot mix_dot
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import faiss
import numpy as np
import torch

from staysearch import results
from staysearch.config import ARTIFACTS
from staysearch.data import load_queries, load_snapshot, stratified_sample
from staysearch.evaluate import EvalSet
from staysearch.towers import TwoTower, embed_listings, embed_queries

PAPER_TABLE2 = {"dot": {2: 0.43, 4: 0.47, 16: 0.66, 32: 0.84}, "l2": {2: 0.71, 4: 0.75, 16: 0.98, 32: 1.00}}
NPROBES = (1, 2, 4, 8, 16, 32, 64)
EFS = (16, 32, 64, 128, 256, 512)


def load_model(name: str) -> TwoTower:
    import json
    cfg = json.loads((ARTIFACTS / "towers" / name / "config.json").read_text())
    m = TwoTower(cfg["similarity"])
    m.load_state_dict(torch.load(ARTIFACTS / "towers" / name / "model.pt"))
    m.eval()
    return m, cfg


def gini(sizes: np.ndarray) -> float:
    x = np.sort(sizes.astype(np.float64))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def bachrach(x: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inner product -> Euclidean: append sqrt(M^2 - |x|^2) to items and 0 to queries."""
    n2 = (x ** 2).sum(1)
    M2 = n2.max()
    xa = np.concatenate([x, np.sqrt(np.maximum(M2 - n2, 0))[:, None]], 1).astype(np.float32)
    qa = np.concatenate([q, np.zeros((len(q), 1))], 1).astype(np.float32)
    return xa, qa


def build_ivf(x: np.ndarray, nlist: int, metric: str, seed: int = 0):
    d = x.shape[1]
    if metric == "dot":
        quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        quant = faiss.IndexFlatL2(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
    idx.cp.seed = seed
    idx.train(x)
    idx.add(x)
    _, assign = quant.search(x, 1)
    return idx, assign[:, 0], faiss.vector_to_array(quant.xb).reshape(nlist, d) if hasattr(quant, "xb") else None


def index_bytes(index) -> int:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    faiss.write_index(index, path)
    n = os.path.getsize(path)
    os.unlink(path)
    return n


def exact_topk(x, q, metric, k=100):
    if metric == "dot":
        s = q @ x.T
    else:
        s = -((q ** 2).sum(1)[:, None] - 2 * q @ x.T + (x ** 2).sum(1)[None, :])
    return np.argsort(-s, 1)[:, :k]


def run(name: str, args) -> None:
    model, cfg = load_model(name)
    metric = cfg["similarity"]
    s = load_snapshot(2)
    q = load_queries("test")
    idx = stratified_sample(q, args.per_city, seed=21)
    es = EvalSet(q, idx, s)
    x = embed_listings(model, s)
    qe, _ = embed_queries(model, q, idx, s, x)
    n = len(x)
    faiss.omp_set_num_threads(args.threads)
    norms = np.linalg.norm(x, axis=1)
    results.append("m2_ann", {"exp": "exp3", "model": name, "similarity": metric, "kind": "embedding_norms",
                              "n_listings": n, "norm_mean": float(norms.mean()), "norm_p01": float(np.percentile(norms, 1)),
                              "norm_p99": float(np.percentile(norms, 99))})
    variants = [(metric, x, qe)]
    if metric == "dot" and args.bachrach:
        xa, qa = bachrach(x, qe)
        variants.append(("dot_bachrach_l2", xa, qa))
    for label, X, Q in variants:
        imetric = "dot" if label == "dot" else "l2"
        gt = exact_topk(X, Q, imetric)
        # Eligible sets for the filtered measurements.
        elig = [es.eligible(j) for j in range(len(es))]
        if imetric == "dot":
            gt_f = [e[np.argsort(-(X[e] @ Q[j]))[:100]] for j, e in enumerate(elig)]
        else:
            gt_f = [e[np.argsort(((X[e] - Q[j]) ** 2).sum(1))[:100]] for j, e in enumerate(elig)]
        for nlist in args.nlists:
            t0 = time.time()
            ivf, assign, _ = build_ivf(X, nlist, imetric)
            build_s = time.time() - t0
            sizes = np.bincount(assign, minlength=nlist)
            top = np.sort(sizes)[::-1]
            results.append("m2_ann", {
                "exp": "exp3", "model": name, "similarity": label, "kind": "cluster_sizes", "nlist": nlist,
                "largest_cluster_share": float(top[0] / n), "top2_share": float(top[:2].sum() / n),
                "gini": gini(sizes), "empty_clusters": int((sizes == 0).sum()),
                "sizes_sorted_desc": top.tolist(), "build_secs": build_s,
                "index_bytes": index_bytes(ivf), "bytes_per_vector": index_bytes(ivf) / n})
            # Cluster of each listing and each query's probe list, for the filtered recall.
            qd, qprobe = ivf.quantizer.search(Q, max(NPROBES))
            for npb in NPROBES:
                if npb > nlist:
                    continue
                ivf.nprobe = npb
                _, I = ivf.search(Q, 100)
                rec_u = np.mean([len(np.intersect1d(I[j], gt[j])) / 100 for j in range(len(Q))])
                rec_f, scanned = [], []
                for j, e in enumerate(elig):
                    probed = np.isin(assign[e], qprobe[j, :npb])
                    cand = e[probed]
                    scanned.append(len(cand) / max(len(e), 1))
                    if imetric == "dot":
                        top_c = cand[np.argsort(-(X[cand] @ Q[j]))[:100]]
                    else:
                        top_c = cand[np.argsort(((X[cand] - Q[j]) ** 2).sum(1))[:100]]
                    denom = min(100, len(gt_f[j]))
                    rec_f.append(len(np.intersect1d(top_c, gt_f[j])) / denom if denom else 1.0)
                rec_f = np.array(rec_f)
                row = {"exp": "exp3", "model": name, "similarity": label, "kind": "ivf_recall", "nlist": nlist,
                       "nprobe": npb, "recall@100_vs_exact_unfiltered": float(rec_u),
                       "recall@100_vs_exact_filtered": float(rec_f.mean()),
                       "filtered_share_of_eligible_scanned": float(np.mean(scanned)), "n_queries": len(Q)}
                if nlist == 128 and npb in PAPER_TABLE2.get("dot" if label.startswith("dot") else "l2", {}):
                    row["paper_table2"] = PAPER_TABLE2["dot" if label.startswith("dot") else "l2"][npb]
                results.append("m2_ann", row)
                print(label, nlist, npb, round(rec_u, 3), round(float(rec_f.mean()), 3), flush=True)
        if label != metric:
            continue
        # HNSW: efSearch sweep, unfiltered and filtered with an ID selector.
        t0 = time.time()
        hnsw = faiss.IndexHNSWFlat(X.shape[1], 32, faiss.METRIC_INNER_PRODUCT if imetric == "dot" else faiss.METRIC_L2)
        hnsw.hnsw.efConstruction = 128
        hnsw.add(X)
        build_s = time.time() - t0
        hb = index_bytes(hnsw)
        for ef in EFS:
            hnsw.hnsw.efSearch = ef
            t1 = time.time()
            _, I = hnsw.search(Q, 100)
            us = (time.time() - t1) / len(Q) * 1e6
            rec_u = np.mean([len(np.intersect1d(I[j], gt[j])) / 100 for j in range(len(Q))])
            rec_f = []
            t2 = time.time()
            for j, e in enumerate(elig):
                sel = faiss.IDSelectorBatch(e.astype(np.int64))
                params = faiss.SearchParametersHNSW(sel=sel, efSearch=ef)
                _, If = hnsw.search(Q[j:j + 1], 100, params=params)
                got = If[0][If[0] >= 0]
                denom = min(100, len(gt_f[j]))
                rec_f.append(len(np.intersect1d(got, gt_f[j])) / denom if denom else 1.0)
            us_f = (time.time() - t2) / len(Q) * 1e6
            results.append("m2_ann", {"exp": "exp3", "model": name, "similarity": label, "kind": "hnsw_recall",
                                      "M": 32, "efConstruction": 128, "efSearch": ef,
                                      "recall@100_vs_exact_unfiltered": float(rec_u),
                                      "recall@100_vs_exact_filtered": float(np.mean(rec_f)),
                                      "us_per_query_unfiltered": us, "us_per_query_filtered": us_f,
                                      "build_secs": build_s, "index_bytes": hb, "bytes_per_vector": hb / n})
            print("hnsw", ef, round(rec_u, 3), round(float(np.mean(rec_f)), 3), flush=True)
        # Removal support, verified rather than assumed (exp4 FAISS side-check).
        ivf, _, _ = build_ivf(X, 128, imetric)
        out = {"exp": "exp4_faiss", "model": name, "similarity": label, "kind": "faiss_memory_and_removal",
               "n_vectors": n, "dim": X.shape[1], "ivf_bytes_per_vector": index_bytes(ivf) / n,
               "hnsw_bytes_per_vector": hb / n, "raw_vector_bytes": 4 * X.shape[1]}
        for nm, ix in (("ivf_flat", ivf), ("hnsw_flat", hnsw)):
            try:
                removed = ix.remove_ids(np.arange(100, dtype=np.int64))
                out[f"{nm}_remove_ids"] = f"supported (removed {removed})"
            except Exception as ex:  # HNSW raises: it cannot unlink nodes from the graph
                out[f"{nm}_remove_ids"] = "not supported: " + str(ex).strip().splitlines()[-1][:160]
        results.append("m2_ann", out)
        print(out, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--per-city", type=int, default=200)
    ap.add_argument("--nlists", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bachrach", action="store_true", default=True)
    args = ap.parse_args()
    for m in args.models:
        run(m, args)


if __name__ == "__main__":
    main()
