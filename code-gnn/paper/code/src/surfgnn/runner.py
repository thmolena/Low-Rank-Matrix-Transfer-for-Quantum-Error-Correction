"""End-to-end head-to-head runner -> single source-of-truth summary.json.

For every (distance, noise regime) we run several independent seeds.  Within a
seed all three decoders (MWPM, BP+OSD, GNN) decode the *identical* test shots
(paired), so McNemar and paired-bootstrap tests are valid; across seeds we get
the training-and-sampling variance that the headline confidence intervals use.

No number is ever typed by hand: this module computes everything and writes it
to results/summary.json, from which the paper's tables and figures are built.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, asdict

import numpy as np

from .circuits import build_benchmark, sample
from .baselines import MWPMDecoder, BPOSDDecoder, logical_error_rate
from .gnn import GNNDecoder, TrainConfig
from . import stats


@dataclass
class RunConfig:
    distances: tuple[int, ...] = (3, 5, 7)
    p: float = 0.002
    p_corr: float = 0.01
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_train: int = 40000
    n_val: int = 8000
    n_test: int = 20000
    gnn_epochs: int = 60
    gnn_hidden: int = 64
    gnn_layers: int = 4


def _agg(per_seed: list[float]) -> dict:
    a = np.array(per_seed, dtype=np.float64)
    n = len(a)
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": float(a.mean()), "sem": se, "per_seed": [float(x) for x in a]}


def run_cell(d: int, regime: str, cfg: RunConfig) -> dict:
    p_corr = cfg.p_corr if regime == "correlated" else 0.0
    bm = build_benchmark(d=d, p=cfg.p, p_corr=p_corr)
    demF, demD = bm.dem(decompose=False), bm.dem(decompose=True)

    decoders = ["mwpm", "bposd", "gnn"]
    seed_ler = {k: [] for k in decoders}
    # pooled per-shot failure indicators (concatenated across seeds, paired)
    pooled_fail = {k: [] for k in decoders}
    pooled_conf_gnn, pooled_correct_gnn = [], []
    epochs_run = []

    for si, seed in enumerate(cfg.seeds):
        base = 1000 * (seed + 1)
        dtr, otr = sample(bm, cfg.n_train, seed=base + 1)
        dva, ova = sample(bm, cfg.n_val, seed=base + 2)
        dte, ote = sample(bm, cfg.n_test, seed=base + 3)

        mwpm = MWPMDecoder(demD)
        bposd = BPOSDDecoder(demF)
        gnn = GNNDecoder(demF, TrainConfig(epochs=cfg.gnn_epochs,
                                           hidden=cfg.gnn_hidden,
                                           layers=cfg.gnn_layers), seed=seed)
        info = gnn.fit(dtr, otr, dva, ova)
        epochs_run.append(info["epochs_run"])

        preds = {"mwpm": mwpm.predict(dte), "bposd": bposd.predict(dte),
                 "gnn": gnn.predict(dte)}
        for k in decoders:
            fail = np.any(preds[k] != ote, axis=1).astype(np.uint8)
            seed_ler[k].append(float(fail.mean()))
            pooled_fail[k].append(fail)

        proba = gnn.predict_proba(dte)
        conf = np.maximum(proba, 1.0 - proba)
        correct = (preds["gnn"][:, 0] == ote[:, 0]).astype(np.uint8)
        pooled_conf_gnn.append(conf)
        pooled_correct_gnn.append(correct)

    pooled = {k: np.concatenate(pooled_fail[k]) for k in decoders}
    n_pooled = len(pooled["mwpm"])
    ler = {}
    for k in decoders:
        kfail = int(pooled[k].sum())
        phat, lo, hi = stats.wilson_interval(kfail, n_pooled)
        ler[k] = {"across_seed": _agg(seed_ler[k]),
                  "pooled": {"ler": phat, "wilson_lo": lo, "wilson_hi": hi,
                             "n": n_pooled}}

    # paired tests vs MWPM and vs BP+OSD
    tests = {}
    for ref in ("mwpm", "bposd"):
        tests[f"gnn_vs_{ref}"] = {
            "mcnemar": stats.mcnemar(pooled[ref], pooled["gnn"]),
            "bootstrap_diff_ler": stats.paired_bootstrap_diff(pooled[ref], pooled["gnn"]),
        }

    conf_all = np.concatenate(pooled_conf_gnn)
    correct_all = np.concatenate(pooled_correct_gnn)
    ece_gnn = stats.expected_calibration_error(conf_all, correct_all)
    rc_gnn = stats.risk_coverage(conf_all, pooled["gnn"])

    return {
        "d": d, "regime": regime, "p": cfg.p, "p_corr": p_corr,
        "rounds": bm.rounds, "num_detectors": demF.num_detectors,
        "num_neighbour_pairs": len(bm.pairs),
        "ler": ler, "paired_tests": tests, "gnn_ece": ece_gnn,
        "gnn_risk_coverage": rc_gnn, "gnn_epochs_run": epochs_run,
    }


def _provenance(t0: float) -> dict:
    import scipy, stim, pymatching, ldpc, torch
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "stim": stim.__version__, "pymatching": pymatching.__version__,
        "ldpc": ldpc.__version__, "torch": torch.__version__,
        "runtime_s": round(time.time() - t0, 1),
    }


def run(cfg: RunConfig, out_path: str) -> dict:
    t0 = time.time()
    cells = []
    for d in cfg.distances:
        for regime in ("independent", "correlated"):
            print(f"[run] d={d} regime={regime} ...", flush=True)
            cell = run_cell(d, regime, cfg)
            g = cell["ler"]["gnn"]["pooled"]["ler"]
            m = cell["ler"]["mwpm"]["pooled"]["ler"]
            b = cell["ler"]["bposd"]["pooled"]["ler"]
            print(f"      MWPM={m:.4f} BP+OSD={b:.4f} GNN={g:.4f}", flush=True)
            cells.append(cell)
            # incremental save: a long run under load is not lost if interrupted
            with open(out_path, "w") as f:
                json.dump({"config": asdict(cfg), "provenance": _provenance(t0),
                           "cells": cells, "complete": False}, f, indent=2)
    summary = {
        "complete": True,
        "config": asdict(cfg),
        "provenance": _provenance(t0),
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] wrote {out_path} in {summary['provenance']['runtime_s']}s")
    return summary
