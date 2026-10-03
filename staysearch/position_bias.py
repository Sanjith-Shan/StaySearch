"""M5 / exp10: position bias handled the KDD'20 way, on SIMULATED logs. Every number this module
writes is labelled `"simulated": true` and never reaches the ledger as anything else.

Real candidate lists (the two-tower's top K over real point-in-time eligible sets, val split) are
re-used, but the clicks are simulated:
- The "true" utility of each candidate is the trained M4 ranker's score (dnn_hist).
- A logging policy shows a ranked list: it mixes the true utility with popularity and noise, so
  what lands on top is only partly what guests want (otherwise position and utility coincide and
  there is no bias to correct).
- A simulated guest examines position p with probability p^-eta (eta = bias strength) and chooses
  one examined listing with probability proportional to exp(utility / tau).
- Two pairwise DNNs train on the logs: one without position, one with log(1 + position) as a feature
  (dropped to 0 with probability 0.15 during training, set to 0 at inference), as in KDD'20.
- Two labelled extensions beyond the spec: position dropout 0.5, and a separate shallow additive
  position tower dropped at inference.
- Recovery is measured on held-out lists against the true utility: Kendall tau over all K
  candidates and NDCG@10 with graded gains from the true utility.

Usage: python -m staysearch.position_bias --tower mix_l2 --truth dnn_hist
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import kendalltau
from torch import nn

from staysearch import results
from staysearch.config import ARTIFACTS
from staysearch.ranking import Candidates, PairwiseDNN, Standardize, model_scores

ETAS = {"weak": 0.5, "medium": 1.0, "strong": 2.0}


def load_truth(name: str, d_in: int) -> nn.Module:
    m = nn.Sequential(Standardize(np.zeros(d_in), np.ones(d_in)), PairwiseDNN(d_in))
    m.load_state_dict(torch.load(ARTIFACTS / "rankers" / f"{name}.pt"))
    m.eval()
    return m


def simulate(lists, util, pop, eta, rng, shown=25, sessions=4, tau=0.5, mix=0.5, noise=0.5):
    """Logged (features row index, position, label) for simulated sessions over each list."""
    rows, pos, lab, grp = [], [], [], []
    for li, idx in enumerate(lists):
        u, p = util[idx], pop[idx]
        zu = (u - u.mean()) / (u.std() + 1e-9)
        zp = (p - p.mean()) / (p.std() + 1e-9)
        for _ in range(sessions):
            policy = mix * zu + (1 - mix) * zp + noise * rng.standard_normal(len(idx))
            order = np.argsort(-policy)[:shown]
            ranks = np.arange(1, len(order) + 1)
            examined = rng.random(len(order)) < ranks ** (-eta)
            if not examined.any():
                continue
            w = np.exp((u[order] - u[order].max()) / tau) * examined
            choice = rng.choice(len(order), p=w / w.sum())
            y = np.zeros(len(order))
            y[choice] = 1
            rows.append(idx[order])
            pos.append(ranks)
            lab.append(y)
            grp.append(len(order))
    return np.concatenate(rows), np.concatenate(pos), np.concatenate(lab), np.array(grp)


class ShallowPositionModel(nn.Module):
    """score = f(listing, query) + g(position): position only enters through a separate additive
    tower, which is dropped at inference. A common variant of the KDD'20 idea, kept additive so the
    relevance tower cannot entangle position with the features the logging policy ranked by."""

    def __init__(self, d_in: int, n_pos: int = 64):
        super().__init__()
        self.f = PairwiseDNN(d_in)
        self.g = nn.Embedding(n_pos, 1)
        nn.init.zeros_(self.g.weight)

    def forward(self, x, pos=None):
        s = self.f(x)
        return s if pos is None else s + self.g(pos).squeeze(-1)


def train_shallow(X, pos, y, group, epochs=8, seed=0):
    torch.manual_seed(seed)
    model = ShallowPositionModel(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    offs = np.concatenate([[0], np.cumsum(group)])
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    Pt = torch.as_tensor(np.minimum(pos - 1, 63), dtype=torch.long)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(group))
        for b in range(0, len(order), 128):
            gs = order[b:b + 128]
            idx = np.concatenate([np.arange(offs[g], offs[g + 1]) for g in gs])
            lens = torch.as_tensor([group[g] for g in gs])
            s = model(Xt[idx], Pt[idx])
            yy = yt[idx]
            seg = torch.repeat_interleave(torch.arange(len(gs)), lens)
            ps = torch.zeros(len(gs)).index_add_(0, seg, s * yy)
            loss = (nn.functional.softplus(-(ps[seg] - s)) * (1 - yy)).sum() / (1 - yy).sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.f


def train_pairwise(X, y, group, epochs=8, seed=0, pos_col=None, pos_dropout=0.15):
    torch.manual_seed(seed)
    model = PairwiseDNN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    offs = np.concatenate([[0], np.cumsum(group)])
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(group))
        for b in range(0, len(order), 128):
            gs = order[b:b + 128]
            idx = np.concatenate([np.arange(offs[g], offs[g + 1]) for g in gs])
            lens = torch.as_tensor([group[g] for g in gs])
            xb = Xt[idx].clone()
            if pos_col is not None:
                drop = torch.rand(len(idx)) < pos_dropout
                xb[drop, pos_col] = 0.0
            s = model(xb)
            yy = yt[idx]
            seg = torch.repeat_interleave(torch.arange(len(gs)), lens)
            ps = torch.zeros(len(gs)).index_add_(0, seg, s * yy)
            loss = (nn.functional.softplus(-(ps[seg] - s)) * (1 - yy)).sum() / (1 - yy).sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def recovery(scores, util, lists) -> dict:
    taus, ndcgs = [], []
    for idx in lists:
        s, u = scores[idx], util[idx]
        taus.append(kendalltau(s, u).statistic)
        # Graded gains: true-utility quintile within the list (0..4).
        g = np.searchsorted(np.quantile(u, [0.2, 0.4, 0.6, 0.8]), u)
        gain = 2.0 ** g - 1
        top = np.argsort(-s)[:10]
        ideal = np.sort(gain)[::-1][:10]
        disc = 1 / np.log2(np.arange(2, 12))
        ndcgs.append((gain[top] * disc[:len(top)]).sum() / max((ideal * disc[:len(ideal)]).sum(), 1e-9))
    return {"kendall_tau": float(np.nanmean(taus)), "ndcg@10_true_utility": float(np.mean(ndcgs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="mix_l2")
    ap.add_argument("--truth", default="dnn_hist")
    ap.add_argument("--per-city", type=int, default=600)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    results.reset("m5_position_bias")
    c = Candidates("val", args.per_city, args.K, args.tower, seed=51, include_positive=False)
    X, _, group = c.matrix(True)
    X = X.astype(np.float32)
    truth = load_truth(args.truth, X.shape[1])
    util = model_scores(truth, X)
    pop = X[:, 12]  # standardized log number_of_reviews_ltm (NUMERIC index 12)
    offs = np.concatenate([[0], np.cumsum(group)])
    lists = [np.arange(offs[g], offs[g + 1]) for g in range(len(group)) if group[g] >= 10]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(lists))
    train_l = [lists[i] for i in perm[: int(0.7 * len(lists))]]
    test_l = [lists[i] for i in perm[int(0.7 * len(lists)):]]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    for strength, eta in ETAS.items():
        for seed in range(args.seeds):
            r = np.random.default_rng(100 + seed)
            rows, pos, y, g = simulate(train_l, util, pop, eta, r)
            base = Xs[rows]
            m_nopos = train_pairwise(base, y, g, seed=seed)
            with_pos = np.concatenate([base, np.log1p(pos)[:, None].astype(np.float32)], 1)
            m_pos = train_pairwise(with_pos, y, g, seed=seed, pos_col=base.shape[1])
            m_pos50 = train_pairwise(with_pos, y, g, seed=seed, pos_col=base.shape[1], pos_dropout=0.5)
            m_shallow = train_shallow(base, pos, y, g, seed=seed)
            Xz = np.concatenate([Xs, np.zeros((len(Xs), 1), np.float32)], 1)
            variants = (("no_position_feature", model_scores(m_nopos, Xs)),
                        ("position_feature_zero_at_inference", model_scores(m_pos, Xz)),
                        ("position_feature_dropout0.5_extension", model_scores(m_pos50, Xz)),
                        ("shallow_position_tower_extension", model_scores(m_shallow, Xs)))
            for name, s in variants:
                rec = recovery(s, util, test_l)
                results.append("m5_position_bias", {"exp": "exp10", "simulated": True, "bias_strength": strength,
                                                    "eta": eta, "seed": seed, "model": name,
                                                    "n_logged_sessions": int(len(g)), "n_test_lists": len(test_l),
                                                    **rec})
                print(strength, seed, name, {k: round(v, 4) for k, v in rec.items()}, flush=True)


if __name__ == "__main__":
    main()
