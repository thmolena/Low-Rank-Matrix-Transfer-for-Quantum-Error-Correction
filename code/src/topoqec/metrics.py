"""Decoding metrics: logical error rate, calibration (ECE), risk-coverage.

A decoder may *abstain* (decline to commit a correction when its confidence is
below a threshold). Two quantities then matter:

  * **logical error rate** on the trials it *did* decode (lower is better);
  * **coverage** -- the fraction of trials it chose to decode.

Sweeping the abstention threshold traces a **risk-coverage curve**: a good,
well-calibrated decoder pushes its errors onto the trials it abstains from, so
risk falls monotonically as coverage shrinks. Calibration is summarised by the
**expected calibration error** (ECE) between stated confidence and realised
accuracy.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def logical_error_rate(failures: Sequence[int]) -> float:
    """Mean of per-trial logical-failure indicators (0/1)."""
    arr = np.asarray(failures, dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def summarize(values: List[float]) -> Dict[str, float]:
    """Mean / std / 95% CI / n -- identical convention to the rest of the suite."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "ci95": round(ci95, 6), "n": int(arr.size)}


def expected_calibration_error(
    confidences: Sequence[float], corrects: Sequence[int], n_bins: int = 10
) -> float:
    """ECE: average |confidence - accuracy| over equal-width confidence bins.

    ``corrects[i] == 1`` iff trial ``i`` was decoded *without* a logical error.
    Returns a value in ``[0, 1]`` (0 = perfectly calibrated).
    """
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(corrects, dtype=float)
    if conf.size == 0:
        return 0.0
    conf = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        # last bin is closed on the right so confidence == 1 is counted
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        m = int(in_bin.sum())
        if m == 0:
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(acc[in_bin].mean())
        ece += (m / n) * abs(bin_conf - bin_acc)
    return float(ece)


def risk_coverage_curve(
    confidences: Sequence[float], failures: Sequence[int], n_points: int = 21
) -> List[List[float]]:
    """Sweep an abstention threshold; report ``[coverage, risk]`` pairs.

    At threshold ``t`` the decoder commits only on trials with confidence ``>= t``.
    *Coverage* is the committed fraction; *risk* is the logical error rate on the
    committed trials. The thresholds run from the min to the max observed
    confidence, so coverage spans (almost) the full ``[0, 1]`` range.
    """
    conf = np.asarray(confidences, dtype=float)
    fail = np.asarray(failures, dtype=float)
    if conf.size == 0:
        return []
    thresholds = np.linspace(conf.min(), conf.max(), n_points)
    curve: List[List[float]] = []
    for t in thresholds:
        keep = conf >= t
        cov = float(keep.mean())
        if keep.sum() == 0:
            risk = 0.0
        else:
            risk = float(fail[keep].mean())
        curve.append([round(cov, 6), round(risk, 6)])
    # sort by descending coverage so the curve reads left (full) -> right (selective)
    curve.sort(key=lambda pr: -pr[0])
    return curve


def is_monotone_risk_coverage(curve: List[List[float]], tol: float = 1e-6) -> bool:
    """True if risk is (weakly) non-increasing as coverage shrinks.

    The curve is stored in descending-coverage order, so a well-calibrated
    decoder has risk that does not increase as we read left-to-right.
    """
    risks = [r for _, r in curve]
    return all(risks[i + 1] <= risks[i] + tol for i in range(len(risks) - 1))
