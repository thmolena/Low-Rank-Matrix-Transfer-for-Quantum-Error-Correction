"""Generate LaTeX tables and figure PDFs from results/summary.json.

Thin wrapper around :mod:`surfgnn.outputs` (the single source of truth). Every
number printed is read from summary.json; nothing is recomputed by hand. Produces:
  results/main_table.tex         headline LER comparison (Wilson CIs + McNemar)
  results/numbers.json           flat key->value map for paper placeholder fill
  figures/fig_ler_vs_distance.pdf
  figures/fig_gain_vs_distance.pdf
  figures/fig_risk_coverage.pdf
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from surfgnn import outputs  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")


if __name__ == "__main__":
    s = outputs.load(RES)
    outputs.build_all(s, RES, FIG)
    print("wrote main_table.tex, numbers.json, and figures")
