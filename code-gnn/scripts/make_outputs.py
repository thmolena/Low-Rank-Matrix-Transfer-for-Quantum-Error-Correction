"""Generate LaTeX tables and figure PDFs from results/summary.json.

Single source of truth: every number printed here is read from summary.json;
nothing is recomputed by hand. Produces:
  results/main_table.tex         headline LER comparison (Wilson CIs + McNemar)
  results/numbers.json           flat key->value map for paper placeholder fill
  figures/fig_ler_vs_distance.pdf
  figures/fig_gain_vs_distance.pdf
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

OKABE = {"mwpm": "#D55E00", "bposd": "#0072B2", "gnn": "#009E73"}
LABEL = {"mwpm": "MWPM", "bposd": "BP+OSD", "gnn": "GNN (ours)"}


def load() -> dict:
    with open(os.path.join(RES, "summary.json")) as f:
        return json.load(f)


def cell(summary, d, regime):
    for c in summary["cells"]:
        if c["d"] == d and c["regime"] == regime:
            return c
    raise KeyError((d, regime))


def main_table(summary) -> str:
    ds = summary["config"]["distances"]
    lines = [
        r"\begin{tabular}{llccc c}",
        r"\toprule",
        r"regime & $d$ & MWPM & BP+OSD & GNN (ours) & GNN vs MWPM \\",
        r"\midrule",
    ]
    for regime in ("independent", "correlated"):
        for d in ds:
            c = cell(summary, d, regime)
            def fmt(k):
                p = c["ler"][k]["pooled"]
                return f"{p['ler']:.4f}\\,[{p['wilson_lo']:.4f},{p['wilson_hi']:.4f}]"
            m = c["ler"]["mwpm"]["pooled"]["ler"]
            g = c["ler"]["gnn"]["pooled"]["ler"]
            gain = (m - g) / m * 100 if m > 0 else 0.0
            pval = c["paired_tests"]["gnn_vs_mwpm"]["mcnemar"]["p_value"]
            gain_s = f"{gain:+.1f}\\% (p={pval:.1e})"
            lines.append(
                f"{regime} & {d} & {fmt('mwpm')} & {fmt('bposd')} & {fmt('gnn')} & {gain_s} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def numbers(summary) -> dict:
    out = {}
    for c in summary["cells"]:
        tag = f"{c['regime']}_d{c['d']}"
        for k in ("mwpm", "bposd", "gnn"):
            out[f"ler_{k}_{tag}"] = round(c["ler"][k]["pooled"]["ler"], 4)
        m = c["ler"]["mwpm"]["pooled"]["ler"]; g = c["ler"]["gnn"]["pooled"]["ler"]
        b = c["ler"]["bposd"]["pooled"]["ler"]
        out[f"gain_gnn_vs_mwpm_{tag}"] = round((m - g) / m * 100, 1) if m else 0.0
        # fraction of MWPM->BPOSD gap recovered by GNN
        if (m - b) != 0:
            out[f"gap_recovered_{tag}"] = round((m - g) / (m - b) * 100, 1)
        out[f"mcnemar_p_gnn_vs_mwpm_{tag}"] = c["paired_tests"]["gnn_vs_mwpm"]["mcnemar"]["p_value"]
        out[f"gnn_ece_{tag}"] = round(c["gnn_ece"], 4)
    out["runtime_s"] = summary["provenance"]["runtime_s"]
    out["seeds"] = len(summary["config"]["seeds"])
    out["p"] = summary["config"]["p"]
    out["p_corr"] = summary["config"]["p_corr"]
    return out


def figures(summary) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif",
                                "font.size": 11, "axes.spines.top": False,
                                "axes.spines.right": False})
    import matplotlib.pyplot as plt
    import numpy as np
    ds = list(summary["config"]["distances"])

    # Fig 1: LER vs distance, two panels (regimes)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, regime in zip(axes, ("independent", "correlated")):
        for k in ("mwpm", "bposd", "gnn"):
            ys, los, his = [], [], []
            for d in ds:
                p = cell(summary, d, regime)["ler"][k]["pooled"]
                ys.append(p["ler"]); los.append(p["ler"] - p["wilson_lo"])
                his.append(p["wilson_hi"] - p["ler"])
            ax.errorbar(ds, ys, yerr=[los, his], marker="o", capsize=3,
                        color=OKABE[k], label=LABEL[k])
        ax.set_title(regime); ax.set_xlabel("code distance $d$")
        ax.set_xticks(ds); ax.set_yscale("log")
    axes[0].set_ylabel("logical error rate")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ler_vs_distance.pdf"))
    plt.close(fig)

    # Fig 2: relative GNN-vs-MWPM gain vs distance (correlated)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    gains = []
    for d in ds:
        c = cell(summary, d, "correlated")
        m = c["ler"]["mwpm"]["pooled"]["ler"]; g = c["ler"]["gnn"]["pooled"]["ler"]
        gains.append((m - g) / m * 100 if m else 0.0)
    ax.bar([str(d) for d in ds], gains, color=OKABE["gnn"])
    ax.set_xlabel("code distance $d$")
    ax.set_ylabel("GNN error reduction vs MWPM (\\%)")
    ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gain_vs_distance.pdf"))
    plt.close(fig)

    # Fig 3: risk--coverage for the GNN under correlated noise (selective decoding)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for d in ds:
        c = cell(summary, d, "correlated")
        rc = c.get("gnn_risk_coverage", [])
        if not rc:
            continue
        cov = [r["coverage"] for r in rc]; risk = [r["risk"] for r in rc]
        ax.plot(cov, risk, marker="o", label=f"$d={d}$")
    ax.set_xlabel("coverage (fraction committed)")
    ax.set_ylabel("committed logical error (risk)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_risk_coverage.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    s = load()
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(RES, "main_table.tex"), "w") as f:
        f.write(main_table(s))
    with open(os.path.join(RES, "numbers.json"), "w") as f:
        json.dump(numbers(s), f, indent=2)
    figures(s)
    print("wrote main_table.tex, numbers.json, and figures")
