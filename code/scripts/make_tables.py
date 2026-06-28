#!/usr/bin/env python
"""Generate every LaTeX table in the manuscript from results/summary.json.

Writes ``results/main_results.tex`` (the headline table body) and
``results/paper_tables.tex`` (all extended-data table bodies), and prints them.
Every table traces verbatim to the single source-of-truth artifact.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

# Fixed decoder display order (spectral first to foreground the contribution).
ORDER = ["spectral", "majority", "matching", "bp", "topology", "lookup", "neural", "fused", "rl"]
REGIMES = ["iid", "biased", "burst", "correlated", "measurement_error"]
REGIME_HDR = "Decoder & i.i.d. & biased & burst & correlated & meas.\\ error \\\\"


def _ordered(decs):
    return [d for d in ORDER if d in decs] + [d for d in decs if d not in ORDER]


def main_table(summary):
    rows = []
    for dec, m in summary["decoders"].items():
        ler = m["logical_error_rate"]
        rows.append((dec, ler["mean"], ler["ci95"], m["ece"]))
    rows.sort(key=lambda r: r[1])
    best = rows[0][0] if rows else None
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"decoder & logical err.\ rate & $\pm$95\% & ECE \\", r"\midrule"]
    for dec, mean, ci, ece in rows:
        star = r"\,$\star$" if dec == best else ""
        lines.append(f"{dec.replace('_',' ')}{star} & {mean:.4f} & {ci:.4f} & {ece:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _regime_table(summary, key, fmt="{:.5f}", bold="min"):
    """Per-(decoder, regime) table for ``key`` in by_regime; bold the column best."""
    br = summary["by_regime"]
    decs = _ordered(list(br[REGIMES[0]].keys()))
    best = {}
    for r in REGIMES:
        vals = {d: br[r][d][key] for d in decs}
        best[r] = (min if bold == "min" else max)(vals, key=vals.get)
    title = {
        "logical_error_rate": "Logical error rate by structured-noise regime",
        "ece": "Expected calibration error by structured-noise regime",
        "coverage": "Coverage at fixed abstention threshold $\\tau=0.5$"}[key]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{5}{c}{%s} \\" % title,
             r"\cmidrule(lr){2-6}", REGIME_HDR, r"\midrule"]
    for d in decs:
        cells = []
        for r in REGIMES:
            v = fmt.format(br[r][d][key])
            cells.append(f"\\textbf{{{v}}}" if best[r] == d else v)
        lines.append(f"{d.replace('_',' '):<9} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def threshold_table(summary):
    curves = summary["threshold_curves"]
    decs = _ordered(list(curves.keys()))
    p_grid = summary["p_grid"]
    head = " & ".join(["$p$"] + [d.replace("_", " ") for d in decs])
    cols = "l" + "c" * len(decs)
    lines = [r"\begin{tabular}{%s}" % cols, r"\toprule", head + r" \\", r"\midrule"]
    for i, p in enumerate(p_grid):
        row = [f"{p:.2f}"] + [f"{curves[d][i]:.6f}" for d in decs]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rc_table(summary):
    rc = summary["risk_coverage"]
    names = [n for n in ["spectral", "matching"] if n in rc]
    blocks = [rc[n] for n in names]
    maxlen = max(len(b) for b in blocks)
    cols = "c" * (2 * len(names))
    lines = [r"\begin{tabular}{%s}" % cols, r"\toprule",
             " & ".join(r"\multicolumn{2}{c}{%s}" % n for n in names) + r" \\",
             " ".join(r"\cmidrule(lr){%d-%d}" % (2 * i + 1, 2 * i + 2) for i in range(len(names))),
             " & ".join(["coverage & risk"] * len(names)) + r" \\", r"\midrule"]
    for i in range(maxlen):
        cells = []
        for b in blocks:
            if i < len(b):
                cells.append(f"{b[i][0]:.5f} & {b[i][1]:.5f}")
            else:
                cells.append(" & ")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def spectral_advantage_table(summary):
    sbr = summary["headline"].get("spectral_by_regime", {})
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Regime & spectral & lookup & abs.\ reduction & rel.\ reduction & $z$ \\",
             r"\midrule"]
    label = {"iid": "i.i.d.", "biased": "biased", "burst": "burst",
             "correlated": "correlated", "measurement_error": "meas.\\ error"}
    for r in REGIMES:
        v = sbr.get(r)
        if not v:
            continue
        lines.append(f"{label[r]} & {v['spectral_ler']:.5f} & {v['lookup_ler']:.5f} & "
                     f"{v['abs_reduction']:+.5f} & {100*v['rel_reduction']:+.1f}\\% & {v['z']:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("results")
    out.mkdir(exist_ok=True)

    main_tex = main_table(summary)
    (out / "main_results.tex").write_text(main_tex)

    tables = {
        "main": main_tex,
        "spectral_advantage": spectral_advantage_table(summary),
        "ed_ler": _regime_table(summary, "logical_error_rate", "{:.5f}", "min"),
        "ed_ece": _regime_table(summary, "ece", "{:.5f}", "min"),
        "ed_cov": _regime_table(summary, "coverage", "{:.5f}", "max"),
        "ed_threshold": threshold_table(summary),
        "ed_rc": rc_table(summary),
    }
    bundle = "\n".join(f"% ===== {k} =====\n{v}" for k, v in tables.items())
    (out / "paper_tables.tex").write_text(bundle)
    print("wrote results/main_results.tex and results/paper_tables.tex\n")
    print(bundle)


if __name__ == "__main__":
    main()
