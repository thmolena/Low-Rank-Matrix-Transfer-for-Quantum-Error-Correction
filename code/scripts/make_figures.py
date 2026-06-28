#!/usr/bin/env python
"""Generate all figures from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from topoqec import plotting


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plotting.fig_schematic(summary, out / "fig_schematic.pdf")
    plotting.fig_perregime(summary, out / "fig_perregime.pdf")
    plotting.fig_threshold(summary, out / "fig_threshold.pdf")
    plotting.fig_risk_coverage(summary, out / "fig_risk_coverage.pdf")
    print(f"wrote {out}/fig_schematic.pdf, {out}/fig_perregime.pdf, "
          f"{out}/fig_threshold.pdf, {out}/fig_risk_coverage.pdf")


if __name__ == "__main__":
    main()
