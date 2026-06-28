"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (see Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range, with
    ``mathtext.fontset = dejavusans`` so in-figure math is also sans-serif.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red--green trap.
    Sequential encodings, where needed, use viridis (never jet/rainbow).
  * Uncertainty is shown wherever a quantitative estimate is plotted; the
    caption states n and that the interval is a 95% confidence interval. For
    the line plots (threshold, risk--coverage) the shaded band is the analytic
    Bernoulli 95% CI of the plotted rate.
  * Top/right spines removed for an uncluttered Nature-style frame.

The figures are produced purely from ``results/summary.json`` -- the single
source of truth written by the experiment runner -- so they regenerate
bit-for-bit from fixed seeds.
"""
from __future__ import annotations

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF
# backend stamps a fixed CreationDate -> byte-identical figure PDFs across runs
# (the plotted numbers are already deterministic: they are read from the
# fixed-seed results/summary.json, the single source of truth).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# --- Okabe-Ito colour-blind-safe qualitative palette ------------------------
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Column widths in inches (Nature: single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

# Human-readable, fixed decoder ordering and display names for legends/axes.
# The channel-adaptive spectral decoder leads the ordering so it takes the
# prominent first colour in every figure.
DECODER_ORDER = [
    "spectral",
    "lookup",
    "majority",
    "matching",
    "bp",
    "neural",
    "topology",
    "fused",
    "rl",
]
DECODER_LABELS = {
    "spectral": "spectral",
    "lookup": "lookup",
    "majority": "majority",
    "matching": "matching",
    "bp": "BP",
    "neural": "neural",
    "topology": "topology",
    "fused": "fused",
    "rl": "RL selector",
}


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
            "ps.fonttype": 42,
            "svg.hashsalt": "topoqec",  # deterministic element IDs across runs
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=NMI_PALETTE),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.02) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def _color_for(decoder: str) -> str:
    if decoder in DECODER_ORDER:
        return NMI_PALETTE[DECODER_ORDER.index(decoder) % len(NMI_PALETTE)]
    return NMI_PALETTE[0]


def _bernoulli_ci95(rate: np.ndarray, n: float) -> np.ndarray:
    """Analytic 95% confidence interval half-width for a Bernoulli rate."""
    rate = np.clip(np.asarray(rate, dtype=float), 0.0, 1.0)
    n = max(float(n), 1.0)
    return 1.96 * np.sqrt(rate * (1.0 - rate) / n)


def fig_threshold(summary: Dict, out: Path) -> Path:
    """Logical error rate vs physical error rate p, per decoder (headline).

    No in-plot title -- the description is in the caption. Each curve carries a
    shaded analytic 95% Bernoulli confidence band from the per-point trial count.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.0))
    curves = summary.get("threshold_curves", {})
    p_grid = np.asarray(summary.get("p_grid", []), dtype=float)
    n_trials = float(summary.get("config", {}).get("trials", 0) or 0)

    # Stable ordering for legend / colour assignment.
    ordered = [d for d in DECODER_ORDER if d in curves]
    ordered += [d for d in curves if d not in ordered]
    for decoder in ordered:
        ys = np.asarray(curves[decoder], dtype=float)
        color = _color_for(decoder)
        ax.plot(
            p_grid,
            ys,
            marker="o",
            ms=2.5,
            color=color,
            label=DECODER_LABELS.get(decoder, decoder),
        )
        if n_trials > 0:
            ci = _bernoulli_ci95(ys, n_trials)
            ax.fill_between(p_grid, ys - ci, ys + ci, color=color, alpha=0.15,
                            linewidth=0)
    # the break-even line: logical rate == physical rate
    if p_grid.size:
        ax.plot(p_grid, p_grid, "--", color="0.55", lw=1.0,
                label="break-even ($p_L=p$)")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.legend(loc="upper left", handlelength=1.4, ncol=1)
    ax.grid(True, axis="both")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_risk_coverage(summary: Dict, out: Path) -> Path:
    """Risk-coverage curve for the headline decoder(s) (selective decoding).

    No in-plot title -- the description is in the caption. The shaded band is the
    analytic 95% Bernoulli confidence interval of the committed logical error
    rate, whose effective sample size is (coverage x pooled shots).
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.0))
    rc = summary.get("risk_coverage", {})
    # Pooled shots per decoder, used to size the per-point CI band.
    decoders = summary.get("decoders", {})

    ordered = [d for d in DECODER_ORDER if d in rc]
    ordered += [d for d in rc if d not in ordered]
    for decoder in ordered:
        curve = rc[decoder]
        xs = np.asarray([pt[0] for pt in curve], dtype=float)
        ys = np.asarray([pt[1] for pt in curve], dtype=float)
        color = _color_for(decoder)
        ax.plot(xs, ys, marker="o", ms=2.5, color=color,
                label=DECODER_LABELS.get(decoder, decoder))
        n_pooled = float(
            decoders.get(decoder, {}).get("logical_error_rate", {}).get("n", 0)
            or 0
        )
        if n_pooled > 0:
            n_eff = np.maximum(xs * n_pooled, 1.0)
            ci = 1.96 * np.sqrt(np.clip(ys, 0, 1) * (1 - np.clip(ys, 0, 1)) / n_eff)
            ax.fill_between(xs, ys - ci, ys + ci, color=color, alpha=0.15,
                            linewidth=0)
    ax.set_xlabel("coverage (fraction of syndromes decoded)")
    ax.set_ylabel("logical error rate on committed shots")
    ax.legend(loc="upper left", handlelength=1.4)
    ax.grid(True, axis="both")
    ax.invert_xaxis()  # full coverage on the left, most selective on the right
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_perregime(summary: Dict, out: Path) -> Path:
    """Per-regime logical error: channel-adaptive spectral decoder vs the
    i.i.d.-optimal leader, with analytic 95% Bernoulli error bars.

    This is the figure that shows the contribution: spectral ties the leader on
    i.i.d. and read-out noise (where no data-channel advantage is possible) and
    drops the logical error sharply on the biased / burst / correlated regimes.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.0))
    br = summary.get("by_regime", {})
    n_trials = float(summary.get("config", {}).get("trials", 0) or 1)
    regimes = [r for r in ["iid", "biased", "burst", "correlated", "measurement_error"]
               if r in br]
    labels = {"iid": "i.i.d.", "biased": "biased", "burst": "burst",
              "correlated": "corr.", "measurement_error": "meas."}
    # baseline = the i.i.d.-optimal leader if present, else matching
    base_name = "lookup" if regimes and "lookup" in br[regimes[0]] else "matching"
    xs = np.arange(len(regimes))
    width = 0.38
    for off, name, color in ((-width / 2, base_name, "#9aa6b5"),
                             (width / 2, "spectral", _color_for("spectral"))):
        ys = np.array([br[r][name]["logical_error_rate"] for r in regimes])
        ci = _bernoulli_ci95(ys, n_trials)
        ax.bar(xs + off, ys, width, color=color,
               label=DECODER_LABELS.get(name, name), yerr=ci, capsize=2,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[r] for r in regimes])
    ax.set_xlabel("structured-noise regime")
    ax.set_ylabel("logical error rate")
    ax.legend(loc="upper left", handlelength=1.2)
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        zorder=5,
    )
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                        shrinkA=2, shrinkB=2),
    )


def fig_schematic(summary: Dict, out: Path) -> Path:
    """Method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline of the selective, calibrated decoding benchmark:
    structured noise corrupts a code word; the measured syndrome (on the code's
    defect graph) is fed to a single ``decode(syndrome) -> (correction,
    confidence)`` interface shared by the whole decoder family; a confidence
    threshold gates abstention; and three first-class metrics -- logical error
    rate, expected calibration error and the risk--coverage curve -- are scored
    with analytic 95% confidence intervals.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, orange, purple, grey = (
        "#D6E6F2",
        "#D6EFE3",
        "#FBE6D4",
        "#ECDCE9",
        "#ECECEC",
    )
    y = 0.40
    h = 0.34
    boxes = [
        (0.005, 0.150, "structured noise\n(iid, bias, burst,\ncorr., meas.)", blue),
        (0.190, 0.165, "syndrome on\ndefect graph\n$s_i=e_i\\oplus e_{i+1}$", green),
        (0.400, 0.175, "decoder family\n$\\mathrm{decode}(s)\\to(\\hat c,\\,\\mathrm{conf})$\n8 decoders", orange),
        (0.625, 0.150, "abstain if\n$\\mathrm{conf}<\\tau$", purple),
        (0.810, 0.180, "logical error,\nECE, risk--coverage\n(95% CIs)", grey),
    ]
    rights = []
    lefts = []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])

    # Annotate the integrity guarantees that underpin the benchmark.
    ax.text(0.5, 0.07,
            "one defect-graph interface for all decoders"
            "   $\\bullet$   audit gate: every number traces to summary.json",
            ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(0.5, 0.95,
            "calibrated, selective decoding under structured noise "
            "(automorphism-invariance integrity flag)",
            ha="center", va="center", fontsize=6.8, color="#333333")
    fig.savefig(out)
    plt.close(fig)
    return out
