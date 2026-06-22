#!/usr/bin/env python
"""Generate LaTeX tables from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    tables = Path("results")
    tables.mkdir(exist_ok=True)

    rows = []
    for dec, m in summary["decoders"].items():
        ler = m["logical_error_rate"]
        rows.append((dec, ler["mean"], ler["ci95"], m["ece"]))
    # sort by logical error rate (best first)
    rows.sort(key=lambda r: r[1])

    best = rows[0][0] if rows else None

    # LaTeX
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"decoder & logical err.\ rate & $\pm$95\% & ECE \\",
             r"\midrule"]
    for dec, mean, ci, ece in rows:
        star = r"\,$\star$" if dec == best else ""
        lines.append(f"{dec.replace('_',' ')}{star} & {mean:.4f} & {ci:.4f} & {ece:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(lines) + "\n")

    print("wrote results/main_results.tex")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
