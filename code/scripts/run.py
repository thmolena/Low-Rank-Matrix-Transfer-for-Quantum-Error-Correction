#!/usr/bin/env python
"""Run the QEC decoding experiment and write results/summary.json."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from topoqec.config import Config
from topoqec.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = run(cfg, Path(args.out))
    h = summary["headline"]
    print(f"[{cfg.name}] code={cfg.code} d={cfg.distance}  "
          f"best={h['best_decoder']} LER={h['best_logical_error_rate']:.4f}  "
          f"fused LER={h['fused_logical_error_rate']:.4f}  "
          f"fused-vs-majority={h['fused_vs_majority_improvement']}x  "
          f"fused ECE={h['fused_ece']:.4f}  "
          f"automorphism_invariant={summary['automorphism_invariant']}  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
