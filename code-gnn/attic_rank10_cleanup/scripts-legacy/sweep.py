"""Correlation-strength sweep at fixed distance -> results/sweep.json.

Sweeps the correlated-fault rate p_corr and records MWPM / BP+OSD / GNN logical
error rates, to show the learned decoder's advantage over matching grows with
correlation strength. Single seed per point (this is an illustrative trend, not
the headline significance test, which lives in summary.json).

Usage: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/sweep.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from surfgnn.circuits import build_benchmark, sample  # noqa: E402
from surfgnn.baselines import MWPMDecoder, BPOSDDecoder, logical_error_rate  # noqa: E402
from surfgnn.gnn import GNNDecoder, TrainConfig  # noqa: E402

D = 3
P = 0.004
PCORRS = [0.0, 0.004, 0.008, 0.012, 0.016]
N_TRAIN, N_VAL, N_TEST = 60000, 10000, 12000

if __name__ == "__main__":
    rows = []
    for pc in PCORRS:
        bm = build_benchmark(d=D, p=P, p_corr=pc)
        demF, demD = bm.dem(decompose=False), bm.dem(decompose=True)
        dtr, otr = sample(bm, N_TRAIN, seed=100)
        dva, ova = sample(bm, N_VAL, seed=101)
        dte, ote = sample(bm, N_TEST, seed=102)
        lm = logical_error_rate(MWPMDecoder(demD).predict(dte), ote)
        lb = logical_error_rate(BPOSDDecoder(demF).predict(dte), ote)
        g = GNNDecoder(demF, TrainConfig(), seed=0)
        g.fit(dtr, otr, dva, ova)
        lg = logical_error_rate(g.predict(dte), ote)
        gain = (lm - lg) / lm * 100 if lm > 0 else 0.0
        rows.append({"p_corr": pc, "mwpm": lm, "bposd": lb, "gnn": lg, "gain_pct": gain})
        print(f"p_corr={pc}: MWPM={lm:.4f} BP+OSD={lb:.4f} GNN={lg:.4f} gain={gain:+.1f}%", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "sweep.json")
    with open(os.path.abspath(out), "w") as f:
        json.dump({"d": D, "p": P, "rows": rows}, f, indent=2)
    print("wrote sweep.json")
