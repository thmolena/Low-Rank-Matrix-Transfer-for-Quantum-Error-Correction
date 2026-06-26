"""Honest statistics for head-to-head decoder comparison on identical shots.

- Wilson score interval for each logical error rate (better than Wald near 0/1).
- Paired McNemar exact test for whether two decoders differ on the SAME shots.
- Paired bootstrap CI for the LER difference.
All comparisons are paired: every decoder sees the identical detector samples,
so discordant-pair tests are valid and far more powerful than unpaired ones.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (phat, lo, hi) Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return phat, max(0.0, center - half), min(1.0, center + half)


def mcnemar(fail_a: np.ndarray, fail_b: np.ndarray) -> dict:
    """Paired McNemar test on per-shot failure indicators (0/1) for A vs B.

    b = A correct, B wrong; c = A wrong, B correct.  Uses the exact binomial
    test on the discordant pairs (robust for small counts).
    """
    fa = fail_a.astype(bool); fb = fail_b.astype(bool)
    b = int(np.sum(~fa & fb))   # A right, B wrong
    c = int(np.sum(fa & ~fb))   # A wrong, B right
    nd = b + c
    if nd == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    # two-sided exact binomial test, H0: discordant pairs split 50/50
    p = float(sps.binomtest(min(b, c), nd, 0.5, alternative="two-sided").pvalue)
    return {"b": b, "c": c, "p_value": p}


def paired_bootstrap_diff(fail_a: np.ndarray, fail_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap CI for LER(A) - LER(B) over identical shots."""
    rng = np.random.default_rng(seed)
    n = len(fail_a)
    diffs = np.empty(n_boot)
    fa = fail_a.astype(np.float64); fb = fail_b.astype(np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fa[idx].mean() - fb[idx].mean()
    return {
        "diff": float(fa.mean() - fb.mean()),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }


def risk_coverage(conf: np.ndarray, fail: np.ndarray, n_points: int = 11) -> list:
    """Risk--coverage operating points by sweeping a confidence threshold.

    Returns a list of {coverage, risk} dicts in descending coverage order;
    `risk` is the logical error rate on the committed (high-confidence) shots.
    """
    conf = np.asarray(conf); fail = np.asarray(fail).astype(np.float64)
    order = np.argsort(-conf)  # most confident first
    fsorted = fail[order]
    n = len(conf)
    out = []
    for frac in np.linspace(1.0, 0.1, n_points):
        k = max(1, int(round(frac * n)))
        out.append({"coverage": round(k / n, 4), "risk": round(float(fsorted[:k].mean()), 5)})
    return out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE between confidence (max class prob) and realized accuracy."""
    conf = np.asarray(conf); correct = np.asarray(correct).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
