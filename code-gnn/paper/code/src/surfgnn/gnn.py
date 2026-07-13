"""Correlated-noise graph-neural-network decoder.

The decoder is a message-passing network over the code's *detector graph*: one
node per detector, with an edge between two detectors whenever some error
mechanism in the detector error model can flip both.  Correlated (hyperedge)
faults therefore appear as denser local subgraphs, giving the network the
structural signal that minimum-weight matching -- which only ever sees a sum of
independent graphlike edges -- cannot represent.

Per shot, the only dynamic input is each detector's 0/1 value; static node
features are the detector's spacetime coordinates.  The network outputs the
probability that the logical observable has flipped; thresholding at 0.5 gives
the decode, and the probability itself is a calibrated confidence usable for
selective decoding.  The graph topology is fixed for a given (distance, regime),
so message passing is a few normalized-adjacency matmuls -- fast on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim
import torch
import torch.nn as nn

from .baselines import dem_to_matrices


def build_detector_graph(dem_full: stim.DetectorErrorModel):
    """Return (A_norm [n_det,n_det] float32, static_feats [n_det,3] float32).

    A_norm is the symmetric-normalized adjacency (with self-loops) of the graph
    whose edges join detectors co-occurring in an error mechanism.
    """
    n_det = dem_full.num_detectors
    adj = np.zeros((n_det, n_det), dtype=np.float64)
    for inst in dem_full.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        for a in range(len(dets)):
            for b in range(a + 1, len(dets)):
                adj[dets[a], dets[b]] += 1.0
                adj[dets[b], dets[a]] += 1.0
    adj = (adj > 0).astype(np.float64)
    adj += np.eye(n_det)  # self-loops
    deg = adj.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    a_norm = (adj * dinv[:, None]) * dinv[None, :]

    coords = dem_full.get_detector_coordinates()
    feats = np.zeros((n_det, 3), dtype=np.float64)
    for i in range(n_det):
        xy = coords.get(i, [0.0, 0.0, 0.0])
        feats[i, : len(xy)] = xy[:3]
    # normalize each coordinate column to roughly unit scale
    for k in range(3):
        col = feats[:, k]
        rng = col.max() - col.min()
        if rng > 0:
            feats[:, k] = (col - col.min()) / rng
    return a_norm.astype(np.float32), feats.astype(np.float32)


class GINLayer(nn.Module):
    """GIN-style message passing with an MLP and a residual connection."""

    def __init__(self, dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        agg = torch.einsum("ij,bjf->bif", a_norm, h)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return self.norm(h + torch.relu(out))  # residual


class GCNDecoder(nn.Module):
    def __init__(self, n_det: int, hidden: int = 96, layers: int = 6):
        super().__init__()
        self.in_dim = 1 + 3  # detector value + (x,y,t)
        self.embed = nn.Linear(self.in_dim, hidden)
        self.layers = nn.ModuleList(GINLayer(hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: torch.Tensor, a_norm: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        # values: [B, n_det]; static: [n_det, 3]; a_norm: [n_det, n_det]
        b = values.shape[0]
        h = torch.cat([values.unsqueeze(-1), static.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        h = self.embed(h)
        for layer in self.layers:
            h = layer(h, a_norm)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values, h.sum(dim=1)], dim=-1)
        return self.head(pooled).squeeze(-1)  # [B] logit


@dataclass
class TrainConfig:
    epochs: int = 80
    batch: int = 1024
    lr: float = 2e-3
    hidden: int = 96
    layers: int = 6
    patience: int = 12
    weight_decay: float = 1e-5


class GNNDecoder:
    """Trainable wrapper exposing the same predict() interface as the baselines."""

    def __init__(self, dem_full: stim.DetectorErrorModel, cfg: TrainConfig | None = None,
                 device: str = "cpu", seed: int = 0):
        self.cfg = cfg or TrainConfig()
        self.device = device
        a_norm, static = build_detector_graph(dem_full)
        self.n_det = a_norm.shape[0]
        self.a_norm = torch.tensor(a_norm, device=device)
        self.static = torch.tensor(static, device=device)
        torch.manual_seed(seed)
        self.model = GCNDecoder(self.n_det, self.cfg.hidden, self.cfg.layers).to(device)
        self.threshold = 0.5

    def fit(self, det_train: np.ndarray, obs_train: np.ndarray,
            det_val: np.ndarray, obs_val: np.ndarray, verbose: bool = False) -> dict:
        cfg = self.cfg
        Xtr = torch.tensor(det_train.astype(np.float32), device=self.device)
        ytr = torch.tensor(obs_train[:, 0].astype(np.float32), device=self.device)
        Xva = torch.tensor(det_val.astype(np.float32), device=self.device)
        yva = torch.tensor(obs_val[:, 0].astype(np.float32), device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        # Raw BCE keeps the sigmoid output a proper posterior estimate (so it
        # stays calibrated and 0.5 is Bayes-optimal); we additionally calibrate
        # the *decision* threshold on validation to squeeze out point LER.
        lossf = nn.BCEWithLogitsLoss()
        yv = obs_val[:, 0].astype(np.uint8)

        def val_ler_and_thr() -> tuple[float, float]:
            proba = self.predict_proba(det_val)
            best_thr, best_err = 0.5, 1.0
            for thr in np.linspace(0.05, 0.95, 91):
                err = float(np.mean((proba >= thr).astype(np.uint8) != yv))
                if err < best_err:
                    best_err, best_thr = err, float(thr)
            return best_err, best_thr

        # Select the model by validation *LER* (the actual metric), not by BCE
        # loss -- the loss minimum need not be the logical-error minimum.
        best_ler = 1.0; best_state = None; best_thr = 0.5; bad = 0
        n = Xtr.shape[0]
        g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(cfg.epochs):
            self.model.train()
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, cfg.batch):
                idx = perm[s:s + cfg.batch]
                opt.zero_grad()
                logit = self.model(Xtr[idx], self.a_norm, self.static)
                loss = lossf(logit, ytr[idx])
                loss.backward(); opt.step()
            vler, vthr = val_ler_and_thr()
            if verbose:
                print(f"   ep{ep:02d} val_ler={vler:.4f} thr={vthr:.2f}")
            if vler < best_ler - 1e-4:
                best_ler = vler; best_thr = vthr
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.threshold = best_thr
        return {"best_val_ler": best_ler, "epochs_run": ep + 1,
                "threshold": self.threshold}

    @torch.no_grad()
    def predict_proba(self, detectors: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.model.eval()
        out = np.zeros(detectors.shape[0], dtype=np.float64)
        for s in range(0, detectors.shape[0], batch):
            X = torch.tensor(detectors[s:s + batch].astype(np.float32), device=self.device)
            p = torch.sigmoid(self.model(X, self.a_norm, self.static)).cpu().numpy()
            out[s:s + batch] = p
        return out

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        return (self.predict_proba(detectors) >= self.threshold).astype(np.uint8)[:, None]
