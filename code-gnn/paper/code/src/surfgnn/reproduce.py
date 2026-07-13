"""Console entrypoint: regenerate every figure, table and paper number.

    gnndec-reproduce [preset] [--repo-root DIR] [--skip-run] [--no-paper-sync]

Presets (see :data:`PRESETS`): ``full`` (archival, hours on CPU), ``main``
(standard reproduce target, ~a couple of hours CPU), ``smoke`` (minutes, for a
quick sanity check). The default is ``main``.

Pipeline:
  1. run the multi-seed head-to-head  -> results/summary.json
  2. build table + numbers + figures  -> results/, figures/
  3. sync artifacts into the manuscript (paper/main_table.tex, paper figures,
     and fill \\PH{...} numeric placeholders + the abstract headline number).

Every number written derives from that fresh run; nothing is hand-edited.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

from .runner import RunConfig, run
from . import outputs

# --------------------------------------------------------------------------- #
# Presets. `main` is the standard reproduce target; `full` is the archival one.
# --------------------------------------------------------------------------- #
PRESETS = {
    "full": RunConfig(distances=(3, 5, 7), p=0.004, p_corr=0.010,
                      seeds=(0, 1, 2), n_train=70000, n_val=10000, n_test=12000,
                      gnn_epochs=60, gnn_hidden=96, gnn_layers=6),
    "main": RunConfig(distances=(3, 5, 7), p=0.004, p_corr=0.010,
                      seeds=(0, 1), n_train=20000, n_val=4000, n_test=6000,
                      gnn_epochs=40, gnn_hidden=96, gnn_layers=6),
    "smoke": RunConfig(distances=(3,), p=0.002, p_corr=0.01,
                       seeds=(0, 1), n_train=20000, n_val=4000, n_test=8000,
                       gnn_epochs=25, gnn_hidden=32, gnn_layers=3),
}


def find_repo_root(start: str | None = None) -> str:
    """Locate the package root (has results/summary.json or paper/main.tex).

    Works both for the historical layout (<root>/paper/main.tex,
    <root>/results/) and for the submission layout in which this package
    lives at <paper>/code, so the root is <paper>/code and the manuscript
    is one level up.
    """
    candidates = []
    if start:
        candidates.append(start)
    # installed location: src/surfgnn/reproduce.py -> repo root two levels up
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.abspath(os.path.join(here, "..", "..")))
    candidates.append(os.getcwd())
    for c in candidates:
        cur = os.path.abspath(c)
        for _ in range(6):
            if os.path.exists(os.path.join(cur, "paper", "main.tex")) or \
               os.path.exists(os.path.join(cur, "results", "summary.json")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    # last resort: cwd
    return os.path.abspath(start or os.getcwd())


# --------------------------------------------------------------------------- #
# Paper synchronization: fill numeric \PH{...} placeholders and the abstract
# headline from the freshly-computed numbers map.
# --------------------------------------------------------------------------- #
def _fmt_p(p: float) -> str:
    """Format a p-value for LaTeX, e.g. 3.8e-141 -> 3.8\\times10^{-141}."""
    s = f"{p:.1e}"                       # '3.8e-141'
    mant, exp = s.split("e")
    exp = int(exp)
    if exp == 0:
        return mant
    return f"{mant}\\times10^{{{exp}}}"


def _placeholder_values(nums: dict) -> dict:
    """Map paper \\PH{key} -> replacement string, using the numbers map."""

    def g(key, default=None):
        return nums.get(key, default)

    vals: dict[str, str] = {}
    # config / provenance
    vals["p"] = f"{g('p'):g}" if g("p") is not None else None
    vals["pcorr"] = f"{g('p_corr'):g}" if g("p_corr") is not None else None
    vals["seeds"] = str(g("seeds")) if g("seeds") is not None else None
    vals["runtime"] = f"{g('runtime_s'):.0f}" if g("runtime_s") is not None else None
    vals["hidden"] = str(g("hidden")) if g("hidden") is not None else None
    vals["layers"] = str(g("layers")) if g("layers") is not None else None

    # per-cell LERs referenced in prose
    for key_paper, key_num in [
        ("ler-gnn-indep-d3", "ler_gnn_independent_d3"),
        ("ler-mwpm-indep-d3", "ler_mwpm_independent_d3"),
    ]:
        v = g(key_num)
        if v is not None:
            vals[key_paper] = f"{v:.4f}"

    # gains vs MWPM (correlated) per distance
    for d in (3, 5, 7):
        v = g(f"gain_gnn_vs_mwpm_correlated_d{d}")
        if v is not None:
            vals[f"gain-d{d}"] = f"{v:.1f}"
    # gap recovered at d3 (correlated)
    v = g("gap_recovered_correlated_d3")
    if v is not None:
        vals["gap-recovered-d3"] = f"{v:.1f}"
    # McNemar p at d3 correlated
    v = g("mcnemar_p_gnn_vs_mwpm_correlated_d3")
    if v is not None:
        vals["mcnemar-d3"] = _fmt_p(v)
    # ECE at d3 correlated
    v = g("gnn_ece_correlated_d3")
    if v is not None:
        vals["ece-corr-d3"] = f"{v:.4f}"

    return {k: v for k, v in vals.items() if v is not None}


def sync_paper(repo_root: str, nums: dict, fig_src: str, res_dir: str) -> list[str]:
    changes: list[str] = []
    paper = os.path.join(repo_root, "paper")
    if not os.path.exists(os.path.join(paper, "main.tex")):
        # submission layout: the package sits at <paper>/code, so the
        # manuscript directory is the parent of the package root.
        cand = os.path.abspath(os.path.join(repo_root, ".."))
        if os.path.exists(os.path.join(cand, "main.tex")):
            paper = cand
    if not os.path.exists(os.path.join(paper, "main.tex")):
        return changes

    # 1) copy table
    src_tab = os.path.join(res_dir, "main_table.tex")
    if os.path.exists(src_tab):
        shutil.copyfile(src_tab, os.path.join(paper, "main_table.tex"))
        changes.append("paper/main_table.tex <- results/main_table.tex")

    # 2) copy figures into paper/figures (graphicspath also lists ../figures)
    pfig = os.path.join(paper, "figures")
    os.makedirs(pfig, exist_ok=True)
    for name in ("fig_ler_vs_distance.pdf", "fig_gain_vs_distance.pdf",
                 "fig_risk_coverage.pdf"):
        s = os.path.join(fig_src, name)
        if os.path.exists(s):
            shutil.copyfile(s, os.path.join(pfig, name))
    changes.append("paper/figures/*.pdf <- figures/*.pdf")

    # 3) fill numeric placeholders in main.tex
    main_tex = os.path.join(paper, "main.tex")
    if os.path.exists(main_tex):
        with open(main_tex) as f:
            text = f.read()
        vals = _placeholder_values(nums)
        for key, val in vals.items():
            pat = r"\\PH\{" + re.escape(key) + r"\}"
            new, n = re.subn(pat, val.replace("\\", "\\\\"), text)
            if n:
                text = new
                changes.append(f"main.tex \\PH{{{key}}} -> {val}")

        # 4) abstract headline number (hardcoded gain + McNemar p at d3 corr)
        gain_d3 = nums.get("gain_gnn_vs_mwpm_correlated_d3")
        p_d3 = nums.get("mcnemar_p_gnn_vs_mwpm_correlated_d3")
        if gain_d3 is not None:
            new, n = re.subn(
                r"cuts the logical error\s*\n?\s*rate by \$\\res\{[0-9.]+\}\\%\$ at distance \$3\$",
                f"cuts the logical error rate by $\\\\res{{{gain_d3:.1f}}}\\\\%$ at distance $3$",
                text)
            if n:
                text = new
                changes.append(f"abstract gain -> {gain_d3:.1f}")
        if p_d3 is not None:
            new, n = re.subn(
                r"paired McNemar\s*\n?\s*\$p=\\res\{[^}]*\}\$",
                f"paired McNemar $p=\\\\res{{{_fmt_p(p_d3)}}}$",
                text)
            if n:
                text = new
                changes.append(f"abstract McNemar p -> {_fmt_p(p_d3)}")

        with open(main_tex, "w") as f:
            f.write(text)

    return changes


def reproduce(preset: str = "main", repo_root: str | None = None,
              skip_run: bool = False, paper_sync: bool = True) -> dict:
    root = find_repo_root(repo_root)
    res_dir = os.path.join(root, "results")
    fig_dir = os.path.join(root, "figures")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    summary_path = os.path.join(res_dir, "summary.json")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; choose from {list(PRESETS)}")

    print(f"[reproduce] repo_root={root}")
    print(f"[reproduce] preset={preset}  skip_run={skip_run}")

    if not skip_run:
        run(PRESETS[preset], summary_path)
    elif not os.path.exists(summary_path):
        raise SystemExit("--skip-run given but results/summary.json is missing")

    summary = outputs.load(res_dir)
    nums = outputs.build_all(summary, res_dir, fig_dir)
    print(f"[reproduce] wrote results/main_table.tex, results/numbers.json, figures/*.pdf")

    if paper_sync:
        changes = sync_paper(root, nums, fig_dir, res_dir)
        print(f"[reproduce] synced manuscript ({len(changes)} edits):")
        for c in changes:
            print("   ", c)

    print("[reproduce] done.")
    return nums


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate all gnndec figures, tables and paper numbers.")
    ap.add_argument("preset", nargs="?", default="main", choices=list(PRESETS),
                    help="run configuration (default: main)")
    ap.add_argument("--repo-root", default=None, help="path to the code-gnn checkout")
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse existing results/summary.json; only rebuild outputs")
    ap.add_argument("--no-paper-sync", action="store_true",
                    help="do not copy artifacts / fill placeholders into paper/")
    args = ap.parse_args(argv)
    reproduce(preset=args.preset, repo_root=args.repo_root,
              skip_run=args.skip_run, paper_sync=not args.no_paper_sync)
    return 0


if __name__ == "__main__":
    sys.exit(main())
