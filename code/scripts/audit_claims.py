#!/usr/bin/env python
"""Readiness-gate audit: traceable numbers, required phrase, no forbidden claims.

Exits non-zero if any check fails, so it can gate CI / `make audit`.
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from topoqec import audit


def main() -> int:
    problems = []

    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        print("FAIL: results/summary.json missing -- run `make demo` first")
        return 1
    summary = json.loads(summary_path.read_text())

    # The headline must exist and be non-empty (traceable to this run).
    headline = summary.get("headline", {})
    if not headline:
        problems.append("summary has an empty 'headline' block")
    for key in ("best_decoder", "spectral_vs_lookup_improvement", "spectral_ece",
                "spectral_by_regime", "n_decoders"):
        if key not in headline:
            problems.append(f"headline macro {key!r} not generated")

    # The central performance claim must be *evidenced*: the spectral decoder's
    # advantage over the i.i.d.-optimal lookup leader must be statistically
    # significant (z > 3) in at least one structured regime, and not significant
    # under i.i.d. noise, where theory forbids any data-channel advantage.
    sbr = headline.get("spectral_by_regime", {})
    if sbr:
        structured = [r for r in sbr if r != "iid"]
        if not any(sbr[r].get("z", 0) > 3.0 for r in structured):
            problems.append("spectral advantage not significant (z<=3) in any structured regime")
        if abs(sbr.get("iid", {}).get("z", 0)) > 4.0:
            problems.append("spectral claims an i.i.d. advantage (|z|>4) that theory forbids")

    # Integrity flag: the cyclic-relabeling automorphism must hold.
    if not summary.get("automorphism_invariant", False):
        problems.append("automorphism invariance failed (integrity flag false)")

    # README must carry the mandated phrase and avoid forbidden claims.
    readme = Path("../README.md")
    if readme.exists():
        text = readme.read_text()
        problems += audit.audit_reproducibility_phrase(text)
        problems += audit.audit_forbidden_claims(text)

    if problems:
        print("AUDIT FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("AUDIT PASSED: all numbers traceable, phrase present, no forbidden claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
