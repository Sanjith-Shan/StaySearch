"""The two towers (2601.06873 section 4.5): an offline listing tower and a small online query tower."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from staysearch import qfeatures as qf
from staysearch.data import Snapshot, hist_rows, places, vocab

DIM = 64


class ListingTower(nn.Module):
    def __init__(self, s2_sizes: list[int], n_num: int, n_missing: int, n_amen: int = 200,
                 n_prop: int = 40, dim: int = DIM):
        super().__init__()
        self.s2 = nn.ModuleList([nn.Embedding(n, 16) for n in s2_sizes])
        self.room = nn.Embedding(4, 4)
        self.prop = nn.Embedding(n_prop, 8)
        d_in = 16 * len(s2_sizes) + 4 + 8 + n_num + n_missing + n_amen
        self.mlp = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, dim))

    def forward(self, s2, room, prop, num, missing, amen):
        x = torch.cat([*(e(s2[:, i]) for i, e in enumerate(self.s2)), self.room(room), self.prop(prop),
                       num, missing, amen], 1)
        return self.mlp(x)


class QueryTower(nn.Module):
    def __init__(self, n_places: int, q_dim: int = qf.Q_FEAT_DIM, dim: int = DIM):
        super().__init__()
        self.place = nn.Embedding(n_places, 32)
        self.mlp = nn.Sequential(nn.Linear(32 + q_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, dim))

    def forward(self, place_idx, qfeat):
        return self.mlp(torch.cat([self.place(place_idx), qfeat], 1))


class TwoTower(nn.Module):
    def __init__(self, similarity: str = "l2"):
        super().__init__()
        v = vocab()
        sizes = [v["s2_sizes"][str(lv)] if str(lv) in v["s2_sizes"] else v["s2_sizes"][lv] for lv in (10, 12, 14)]
        self.listing = ListingTower(sizes, len(v["numeric"]), len(v["missing_flags"]))
        self.query = QueryTower(len(places()["idx"]))
        self.similarity = similarity
        self.log_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def logits(self, q: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """q [B, d], l [B, M, d] or [M, d] -> [B, M]; higher is better."""
        s = self.log_scale.clamp(-2, 5).exp()
        if l.dim() == 2:
            if self.similarity == "dot":
                return s * q @ l.T
            return -s * torch.cdist(q, l).pow(2) / q.shape[1]
        if self.similarity == "dot":
            return s * torch.einsum("bd,bmd->bm", q, l)
        return -s * (q[:, None, :] - l).pow(2).sum(-1) / q.shape[1]


def listing_inputs(snap: Snapshot, rows: np.ndarray, device="cpu") -> tuple:
    t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    return (t(snap.s2[rows], torch.long), t(snap.room_type[rows], torch.long),
            t(snap.property_type[rows], torch.long), t(snap.num[rows]), t(snap.missing[rows]),
            t(snap.amen[rows]))


@torch.no_grad()
def embed_listings(model: TwoTower, snap: Snapshot, batch: int = 65536) -> np.ndarray:
    """The daily offline batch job: every listing in the snapshot through the listing tower."""
    model.eval()
    out = np.zeros((snap.n, DIM), dtype=np.float32)
    for i in range(0, snap.n, batch):
        rows = np.arange(i, min(i + batch, snap.n))
        out[rows] = model.listing(*listing_inputs(snap, rows)).numpy()
    return out


def query_features(q: dict, idx: np.ndarray, snap: Snapshot, emb: np.ndarray | None,
                   drop_guests_nights: bool = False, use_history: bool = True) -> np.ndarray:
    """The 87-d qfeat for queries `idx` (see service/BUNDLE_FORMAT.md)."""
    P = places()
    p = q["place_idx"][idx]
    base = qf.query_base(P["area_km2"][p], P["kind"][p] == "city", q["recon_guests"][idx],
                         q["recon_nights"][idx], q["lead_days"][idx], q["recon_checkin"][idx])
    if drop_guests_nights:
        base[:, 2] = 0.0
        base[:, 3] = 0.0
    x = np.zeros((len(idx), qf.Q_FEAT_DIM), dtype=np.float32)
    x[:, :17] = base
    if use_history and emb is not None:
        x[:, 17:] = qf.history_block(hist_rows(q, snap, idx), emb, snap.price, snap.room_type)
    return x


@torch.no_grad()
def embed_queries(model: TwoTower, q: dict, idx: np.ndarray, snap: Snapshot, emb: np.ndarray,
                  **kw) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    qfeat = query_features(q, idx, snap, emb, **kw)
    out = model.query(torch.as_tensor(q["place_idx"][idx], dtype=torch.long), torch.as_tensor(qfeat))
    return out.numpy(), qfeat
