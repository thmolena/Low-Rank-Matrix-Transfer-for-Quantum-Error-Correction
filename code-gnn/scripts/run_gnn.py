#!/usr/bin/env python3
"""Proof-of-concept: a GNN decoder for the rotated surface-code Z-memory, compared
head-to-head against MWPM and BP+OSD under i.i.d. vs. correlated circuit noise.

Hypothesis (the genuine, falsifiable claim):
  * Under i.i.d. circuit noise, MWPM is near-optimal, so GNN ~ MWPM ~ BP+OSD.
  * Under correlated XX noise, MWPM (restricted to the decomposed/graphlike DEM)
    is provably blind to the joint detector structure; a learned decoder and
    BP+OSD (full DEM) can exploit it, so GNN should BEAT MWPM and approach BP+OSD.

The GNN runs on the detector graph whose edges are the co-firing cliques of the
FULL detector error model (so correlated hyperedges are visible to the network).
Every number is from real stim shots; LERs carry bootstrap 95% CIs. This script
reports whatever it finds -- including a null -- and is the d=3 POC; scale via CLI.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)

import stim  # noqa: E402
from surfgnn import circuits as C  # noqa: E402
from surfgnn.baselines import MWPMDecoder, BPOSDDecoder, logical_error_rate  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


# ---------------------------------------------------------------------------
# Detector graph from the FULL DEM: nodes = detectors, edges = co-firing cliques.
# ---------------------------------------------------------------------------
def detector_graph(benchmark: C.Benchmark):
    dem = benchmark.dem(decompose=False)
    n_det = dem.num_detectors
    A = np.zeros((n_det, n_det), dtype=np.float64)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        for a in range(len(dets)):
            for b in range(a + 1, len(dets)):
                A[dets[a], dets[b]] = 1.0
                A[dets[b], dets[a]] = 1.0
    # detector coordinates (x, y, t) for static node features
    coords = benchmark.circuit.get_detector_coordinates()
    feat = np.zeros((n_det, 3), dtype=np.float64)
    for di in range(n_det):
        c = coords.get(di, [0, 0, 0])
        feat[di, :len(c)] = c[:3]
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
    # symmetric normalized adjacency with self loops
    A = A + np.eye(n_det)
    deg = A.sum(1, keepdims=True)
    A_norm = A / np.clip(deg, 1e-9, None)
    return A_norm.astype(np.float32), feat.astype(np.float32), n_det


# ---------------------------------------------------------------------------
# A small message-passing GNN over the fixed detector graph.
# ---------------------------------------------------------------------------
class GNNDecoder(nn.Module):
    def __init__(self, n_static: int, n_obs: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.embed = nn.Linear(n_static + 1, hidden)   # +1 for the per-shot detector bit
        self.gnn = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                           nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_obs))

    def forward(self, det, A, static):
        # det [B, N], A [N, N], static [N, Fs]
        B, N = det.shape
        x = torch.cat([static.unsqueeze(0).expand(B, -1, -1), det.unsqueeze(-1)], dim=-1)
        h = torch.relu(self.embed(x))                  # [B, N, H]
        for mlp, ln in zip(self.gnn, self.norm):
            agg = torch.einsum("ij,bjh->bih", A, h)     # neighbour mean
            h = ln(h + torch.relu(mlp(torch.cat([h, agg], dim=-1))))
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=-1)
        return self.head(pooled)                        # [B, n_obs] logits


def train_gnn(det_tr, obs_tr, A, static, n_obs, *, epochs=40, bs=512, lr=2e-3,
              hidden=64, layers=3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = GNNDecoder(static.shape[1], n_obs, hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    At = torch.tensor(A, device=device); St = torch.tensor(static, device=device)
    X = torch.tensor(det_tr, dtype=torch.float32, device=device)
    Y = torch.tensor(obs_tr, dtype=torch.float32, device=device)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logit = model(X[idx], At, St)
            loss = lossf(logit, Y[idx])
            loss.backward(); opt.step()
    return model, At, St


@torch.no_grad()
def gnn_predict(model, det, At, St, device="cpu", bs=4096):
    out = []
    X = torch.tensor(det, dtype=torch.float32, device=device)
    for i in range(0, X.shape[0], bs):
        logit = model(X[i:i + bs], At, St)
        out.append((logit > 0).to(torch.uint8).cpu().numpy())
    return np.concatenate(out, axis=0)


def bootstrap_ci(pred, obs, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    fails = np.any(pred != obs, axis=1).astype(np.float64)
    n = len(fails)
    means = [fails[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(fails.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_regime(d, rounds, p, p_corr, n_train, n_test, seed, epochs):
    t0 = time.time()
    bench = C.build_benchmark(d=d, rounds=rounds, p=p, p_corr=p_corr)
    A, static, n_det = detector_graph(bench)
    det_tr, obs_tr = C.sample(bench, n_train, seed=seed)
    det_te, obs_te = C.sample(bench, n_test, seed=seed + 99999)
    det_tr = det_tr.astype(np.float32); det_te_f = det_te.astype(np.float32)
    obs_tr = obs_tr.astype(np.float32)
    n_obs = obs_tr.shape[1]

    # --- GNN ---
    model, At, St = train_gnn(det_tr, obs_tr, A, static, n_obs, epochs=epochs, seed=seed)
    gnn_pred = gnn_predict(model, det_te_f, At, St)
    gnn = bootstrap_ci(gnn_pred, obs_te, seed=seed)

    # --- MWPM (decomposed DEM) ---
    mwpm = MWPMDecoder(bench.dem(decompose=True))
    mwpm_pred = mwpm.predict(det_te)
    mw = bootstrap_ci(mwpm_pred, obs_te, seed=seed)

    # --- BP+OSD (full DEM) ---
    try:
        bposd = BPOSDDecoder(bench.dem(decompose=False))
        bposd_pred = bposd.predict(det_te[:min(len(det_te), 4000)])
        bp = bootstrap_ci(bposd_pred, obs_te[:min(len(obs_te), 4000)], seed=seed)
    except Exception as e:
        bp = (float("nan"), float("nan"), float("nan"))
        print(f"    [bposd skipped: {e}]", flush=True)

    print(f"  d={d} p={p} p_corr={p_corr} seed={seed} n_det={n_det} "
          f"({time.time()-t0:.0f}s): "
          f"GNN={gnn[0]:.4f}[{gnn[1]:.4f},{gnn[2]:.4f}] "
          f"MWPM={mw[0]:.4f}[{mw[1]:.4f},{mw[2]:.4f}] "
          f"BPOSD={bp[0]:.4f}", flush=True)
    return dict(d=d, p=p, p_corr=p_corr, seed=seed, n_det=n_det,
                gnn=gnn, mwpm=mw, bposd=bp, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--p", type=float, default=0.006)
    ap.add_argument("--p_corr", type=float, nargs="+", default=[0.0, 0.03])
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "gnn_poc.json"))
    args = ap.parse_args()
    rounds = args.rounds if args.rounds is not None else args.d
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"GNN-vs-MWPM POC: d={args.d} rounds={rounds} p={args.p} "
          f"p_corr={args.p_corr} seeds={args.seeds}", flush=True)
    rows = []
    for pc in args.p_corr:
        for sd in args.seeds:
            rows.append(run_regime(args.d, rounds, args.p, pc, args.n_train,
                                   args.n_test, sd, args.epochs))
    json.dump({"config": vars(args), "rounds": rounds, "rows": rows},
              open(args.out, "w"), indent=2)
    # honest verdict per p_corr (mean over seeds)
    print("\n=== verdict (mean over seeds) ===")
    for pc in args.p_corr:
        sub = [r for r in rows if r["p_corr"] == pc]
        g = np.mean([r["gnn"][0] for r in sub]); m = np.mean([r["mwpm"][0] for r in sub])
        b = np.nanmean([r["bposd"][0] for r in sub])
        tag = "GNN<MWPM (beats)" if g < m else "GNN>=MWPM (no win)"
        print(f"  p_corr={pc}: GNN={g:.4f} MWPM={m:.4f} BPOSD={b:.4f} -> {tag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
