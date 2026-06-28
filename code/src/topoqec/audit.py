"""Claim auditing: no fabricated numbers, no unevidenced hype.

Implements the readiness gate:
  * every claimed number must trace to a value in the generated summary;
  * the mandated reproducibility sentence must be present;
  * unevidenced *hype* superlatives are rejected.

The manuscript makes one substantive performance claim -- that the channel-
adaptive spectral decoder achieves a statistically significant reduction in
logical error under structured noise -- but that claim is *evidenced*: it is
backed by per-regime two-sample z-scores written to ``summary.json`` and by the
provable sub-optimality of the i.i.d. coset leader on a structured channel. The
gate therefore forbids only vague, unquantified superlatives (the phrases below),
not specific, statistically supported statements.
"""
from __future__ import annotations

import re
from typing import Dict, List

REQUIRED_PHRASE = (
    "All experiments are reproducible on commodity hardware; "
    "runtime and memory are reported for each benchmark."
)

# Vague, unquantified hype only. Specific, evidenced claims ("statistically
# significant reduction of X under regime Y, z = Z") are permitted.
FORBIDDEN_PATTERNS = [
    r"state[- ]of[- ]the[- ]art",
    r"\bworld[- ]?class\b",
    r"\bbest[- ]ever\b",
    r"\bbeats all\b",
    r"\boutperforms everyone\b",
    r"\bunconditionally optimal\b",
]


def audit_reproducibility_phrase(text: str) -> List[str]:
    return [] if REQUIRED_PHRASE in text else [
        "missing required reproducibility phrase"
    ]


def audit_forbidden_claims(text: str) -> List[str]:
    problems = []
    for pat in FORBIDDEN_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            problems.append(f"forbidden claim {m.group(0)!r}")
    return problems


def audit_numbers(claimed: Dict[str, float], summary: Dict[str, float],
                  tol: float = 1e-6) -> List[str]:
    """Each claimed macro must equal a generated summary value."""
    problems = []
    for key, val in claimed.items():
        if key not in summary:
            problems.append(f"claimed number {key!r} has no generated artifact")
        elif abs(summary[key] - val) > tol:
            problems.append(
                f"claimed {key}={val} != generated {summary[key]}"
            )
    return problems
