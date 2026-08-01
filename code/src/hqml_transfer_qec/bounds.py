"""Deterministic finite-batch bounds used by the QEC audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationAudit:
    row_bounds: np.ndarray
    actual_row_errors: np.ndarray
    covered: bool


def normalized_prediction_error_bound(
    exact_numerator: np.ndarray,
    exact_denominator: np.ndarray,
    approximate_numerator: np.ndarray,
    approximate_denominator: np.ndarray,
) -> NormalizationAudit:
    """Bound rowwise errors in a normalized kernel prediction.

    For nonzero ``d`` and ``d_tilde``, the identity obtained by adding and
    subtracting ``n_tilde / d`` gives

    ``||n/d - n_tilde/d_tilde|| <= ||n-n_tilde||/|d|``
    ``+ ||n_tilde|| |d-d_tilde|/(|d d_tilde|)``.

    The result is a deterministic approximation bound, not a statistical
    generalization guarantee.
    """

    numerator = np.asarray(exact_numerator, dtype=float)
    approximate = np.asarray(approximate_numerator, dtype=float)
    denominator = np.asarray(exact_denominator, dtype=float).reshape(-1)
    approximate_denominator = np.asarray(
        approximate_denominator, dtype=float
    ).reshape(-1)
    if numerator.ndim == 1:
        numerator = numerator[:, None]
    if approximate.ndim == 1:
        approximate = approximate[:, None]
    if numerator.shape != approximate.shape:
        raise ValueError("numerator shapes differ")
    if len(denominator) != len(numerator) or len(approximate_denominator) != len(
        numerator
    ):
        raise ValueError("denominator shapes differ")

    absolute_denominator = np.abs(denominator)
    absolute_approximate_denominator = np.abs(approximate_denominator)
    valid = (absolute_denominator > 0.0) & (
        absolute_approximate_denominator > 0.0
    )
    row_bounds = np.full(len(numerator), np.inf, dtype=float)
    row_bounds[valid] = (
        np.linalg.norm(numerator[valid] - approximate[valid], axis=1)
        / absolute_denominator[valid]
        + np.linalg.norm(approximate[valid], axis=1)
        * np.abs(denominator[valid] - approximate_denominator[valid])
        / (absolute_denominator[valid] * absolute_approximate_denominator[valid])
    )
    actual = np.full(len(numerator), np.inf, dtype=float)
    actual[valid] = np.linalg.norm(
        numerator[valid] / denominator[valid, None]
        - approximate[valid] / approximate_denominator[valid, None],
        axis=1,
    )
    return NormalizationAudit(
        row_bounds=row_bounds,
        actual_row_errors=actual,
        covered=bool(np.all(actual <= row_bounds * (1.0 + 1e-12))),
    )


@dataclass(frozen=True)
class TwoRankAudit:
    contraction_ratio: float
    error_bound: float
    condition_observed: bool
    covered: bool


def two_rank_contraction_audit(
    low_rank_prediction: np.ndarray,
    high_rank_prediction: np.ndarray,
    reference_prediction: np.ndarray,
) -> TwoRankAudit:
    """Evaluate the conditional two-rank inequality on a finite batch."""

    low = np.asarray(low_rank_prediction, dtype=float)
    high = np.asarray(high_rank_prediction, dtype=float)
    reference = np.asarray(reference_prediction, dtype=float)
    low_error = float(np.linalg.norm(low - reference))
    high_error = float(np.linalg.norm(high - reference))
    if low_error == 0.0:
        ratio = 0.0 if high_error == 0.0 else np.inf
    else:
        ratio = high_error / low_error
    condition = bool(ratio < 1.0)
    bound = (
        ratio / (1.0 - ratio) * float(np.linalg.norm(high - low))
        if condition
        else np.inf
    )
    return TwoRankAudit(
        contraction_ratio=float(ratio),
        error_bound=float(bound),
        condition_observed=condition,
        covered=bool(condition and high_error <= bound * (1.0 + 1e-12)),
    )


@dataclass(frozen=True)
class LogLossAudit:
    actual_difference: float
    bound: float
    covered: bool


def clipped_log_loss_perturbation_bound(
    labels: np.ndarray,
    first_prediction: np.ndarray,
    second_prediction: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> LogLossAudit:
    """Bound weighted binary-log-loss change after probability clipping.

    On ``[epsilon, 1-epsilon]`` the derivative magnitude of binary log loss
    is at most ``1/epsilon``.  The mean-value theorem therefore yields a
    weighted Lipschitz bound.  The weights are normalized internally.
    """

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must lie in (0, 1/2)")
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.clip(np.asarray(first_prediction, dtype=float).reshape(-1), epsilon, 1.0 - epsilon)
    q = np.clip(np.asarray(second_prediction, dtype=float).reshape(-1), epsilon, 1.0 - epsilon)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if not (len(y) == len(p) == len(q) == len(w)):
        raise ValueError("input lengths differ")
    w = w / max(float(w.sum()), 1e-300)
    loss_p = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    loss_q = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    actual = float(abs(np.sum(w * (loss_p - loss_q))))
    bound = float(np.sum(w * np.abs(p - q)) / epsilon)
    return LogLossAudit(
        actual_difference=actual,
        bound=bound,
        covered=bool(actual <= bound * (1.0 + 1e-12)),
    )
