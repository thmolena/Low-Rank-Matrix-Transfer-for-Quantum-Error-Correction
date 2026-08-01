"""Simple quantity-aware baselines for the authenticated QEC table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BITS = 16
TABLE_SIZE = 1 << BITS
RHO_GRID = tuple([0.0, *np.logspace(-5, -1, 13).tolist()])


def class_balanced_weights(labels: np.ndarray, quantities: np.ndarray) -> np.ndarray:
    """Return weights assigning total mass one half to each binary class."""

    y = np.asarray(labels, dtype=int).reshape(-1)
    q = np.asarray(quantities, dtype=float).reshape(-1)
    if len(y) != len(q):
        raise ValueError("label and quantity lengths differ")
    if np.any(q < 0.0) or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("invalid binary labels or quantities")
    weights = q.copy()
    for label in (0, 1):
        mask = y == label
        total = float(weights[mask].sum())
        if total <= 0.0:
            raise ValueError("both classes must have positive quantity")
        weights[mask] *= 0.5 / total
    return weights


def binary_metrics(
    labels: np.ndarray,
    quantities: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Compute class-balanced log loss, Brier score, and balanced accuracy."""

    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    weights = class_balanced_weights(y.astype(int), quantities)
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return {
        "log_loss": float(
            np.sum(weights * (-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))
        ),
        "brier": float(np.sum(weights * (p - y) ** 2)),
        "balanced_accuracy": float(np.sum(weights * ((p >= 0.5) == y))),
    }


def hypercube_kernel_apply(values: np.ndarray, rho: float, *, bits: int = BITS) -> np.ndarray:
    """Apply the product kernel ``rho**Hamming(s,t)`` on the binary cube.

    The kernel is the Kronecker power of ``[[1, rho], [rho, 1]]``.  This
    routine applies one two-point transform per bit and never constructs the
    square ``2**bits`` kernel.
    """

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    array = np.asarray(values, dtype=float).reshape(-1).copy()
    expected = 1 << bits
    if len(array) != expected:
        raise ValueError(f"expected {expected} hypercube values")
    for bit in range(bits):
        step = 1 << bit
        blocks = array.reshape(-1, 2 * step)
        left = blocks[:, :step].copy()
        right = blocks[:, step:].copy()
        blocks[:, :step] = left + rho * right
        blocks[:, step:] = right + rho * left
    return array


def hamming_prediction_table(
    patterns: np.ndarray,
    labels: np.ndarray,
    quantities: np.ndarray,
    fit_mask: np.ndarray,
    rho: float,
    *,
    bits: int = BITS,
) -> np.ndarray:
    """Fit an equal-class-prior Hamming product-kernel probability table."""

    pattern = np.asarray(patterns, dtype=int).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    q = np.asarray(quantities, dtype=float).reshape(-1)
    mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if not (len(pattern) == len(y) == len(q) == len(mask)):
        raise ValueError("input lengths differ")
    size = 1 << bits
    scores = []
    for label in (0, 1):
        selected = mask & (y == label)
        counts = np.bincount(
            pattern[selected], weights=q[selected], minlength=size
        ).astype(float)
        total = float(counts.sum())
        if total <= 0.0:
            raise ValueError("both fitted classes require positive quantity")
        scores.append(hypercube_kernel_apply(counts / total, rho, bits=bits))
    denominator = scores[0] + scores[1]
    return np.divide(
        scores[1],
        denominator,
        out=np.full(size, 0.5, dtype=float),
        where=denominator > 0.0,
    )


@dataclass(frozen=True)
class HammingSelection:
    window: int
    rho: float
    mean_development_log_loss: float
    trace: list[dict[str, float | int]]
    prediction_table: np.ndarray


def tune_hamming_transfer(
    patterns: np.ndarray,
    labels: np.ndarray,
    quantities: np.ndarray,
    rate_indices: np.ndarray,
    *,
    development_rates: tuple[int, ...] = (2, 3, 4),
    window_grid: tuple[int, ...] = (1, 2, 3, 4),
    rho_grid: tuple[float, ...] = RHO_GRID,
) -> HammingSelection:
    """Select history width and Hamming bandwidth by rolling rate holdout.

    Each development rate is predicted using only lower-rate cohorts.  After
    selection, the table is refitted on the selected number of cohorts ending
    at the last development rate; the two higher test rates remain untouched.
    """

    rate = np.asarray(rate_indices, dtype=int).reshape(-1)
    y = np.asarray(labels, dtype=int).reshape(-1)
    q = np.asarray(quantities, dtype=float).reshape(-1)
    trace: list[dict[str, float | int]] = []
    best: tuple[float, int, float] | None = None
    for window in window_grid:
        for rho in rho_grid:
            losses = []
            for target in development_rates:
                start = max(0, target - window)
                fit_mask = (rate >= start) & (rate < target)
                table = hamming_prediction_table(
                    patterns, y, q, fit_mask, float(rho)
                )
                target_mask = rate == target
                losses.append(
                    binary_metrics(y[target_mask], q[target_mask], table[np.asarray(patterns)[target_mask]])[
                        "log_loss"
                    ]
                )
            mean_loss = float(np.mean(losses))
            trace.append(
                {
                    "window": int(window),
                    "rho": float(rho),
                    "mean_development_log_loss": mean_loss,
                }
            )
            candidate = (mean_loss, int(window), float(rho))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("empty Hamming selection grid")
    _, selected_window, selected_rho = best
    final_rate = max(development_rates)
    fit_mask = (rate >= final_rate + 1 - selected_window) & (rate <= final_rate)
    table = hamming_prediction_table(
        patterns, y, q, fit_mask, selected_rho
    )
    return HammingSelection(
        window=selected_window,
        rho=selected_rho,
        mean_development_log_loss=float(best[0]),
        trace=trace,
        prediction_table=table,
    )
