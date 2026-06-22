"""Figure generation from results artifacts (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def fig_threshold(summary: Dict, out: Path) -> Path:
    """Logical error rate vs physical error rate p, per decoder (headline)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    curves = summary.get("threshold_curves", {})
    p_grid = summary.get("p_grid", [])
    for decoder, ys in curves.items():
        ax.plot(p_grid, ys, marker="o", ms=3, label=decoder)
    # the break-even line: logical rate == physical rate
    if p_grid:
        ax.plot(p_grid, p_grid, "--", color="0.6", lw=1, label="break-even (p)")
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel("logical error rate")
    ax.set_title("Decoding under structured noise")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_risk_coverage(summary: Dict, out: Path) -> Path:
    """Risk-coverage curve for the headline decoder(s) (selective decoding)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    rc = summary.get("risk_coverage", {})
    for decoder, curve in rc.items():
        xs = [pt[0] for pt in curve]
        ys = [pt[1] for pt in curve]
        ax.plot(xs, ys, marker="o", ms=3, label=decoder)
    ax.set_xlabel("coverage (fraction of syndromes decoded)")
    ax.set_ylabel("logical error rate on committed")
    ax.set_title("Selective decoding: risk vs coverage")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # full coverage on the left, most selective on the right
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
