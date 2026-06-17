"""Structured-noise error models.

All samplers take a numpy ``Generator`` and a number of data bits, and return a
0/1 ``int8`` error vector (which physical bits were flipped). The regimes:

  * ``iid``          -- each bit flips independently with probability ``p``.
  * ``biased``       -- two interleaved sublattices flip at ``p`` and ``bias*p``
                        (a coarse Z/X bias surrogate for a single error type).
  * ``burst``        -- with probability ``p`` a contiguous run of length
                        ``~burst_len`` flips (correlated spatial damage).
  * ``correlated``   -- a seed flip recruits its lattice neighbours with
                        probability ``corr`` (nearest-neighbour coupling).
  * ``measurement_error`` -- iid data noise PLUS syndrome read-out noise: the
                        measured syndrome is flipped per-bit with probability
                        ``q``. Returned as ``(error, syndrome_flips)``.

The structured regimes are exactly where a topology-aware decoder is expected to
help over a memoryless majority vote, so they are the headline comparison.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

REGIMES = ["iid", "biased", "burst", "correlated", "measurement_error"]


def sample_iid(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < p).astype(np.int8)


def sample_biased(n: int, p: float, rng: np.random.Generator, bias: float = 3.0) -> np.ndarray:
    """Even bits flip at ``min(1, bias*p)``, odd bits at ``p`` (or vice versa)."""
    err = np.zeros(n, dtype=np.int8)
    ph = min(1.0, bias * p)
    err[0::2] = (rng.random(len(err[0::2])) < ph).astype(np.int8)
    err[1::2] = (rng.random(len(err[1::2])) < p).astype(np.int8)
    return err


def sample_burst(n: int, p: float, rng: np.random.Generator, burst_len: int = 3) -> np.ndarray:
    """Spatially-correlated bursts: contiguous runs of flips."""
    err = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        if rng.random() < p:
            length = 1 + rng.poisson(max(0.0, burst_len - 1))
            err[i:i + length] = 1
            i += length
        else:
            i += 1
    return err


def sample_correlated(n: int, p: float, rng: np.random.Generator, corr: float = 0.5) -> np.ndarray:
    """Nearest-neighbour recruitment: a seed flip drags its neighbours in."""
    err = (rng.random(n) < p).astype(np.int8)
    for _ in range(2):  # two recruitment sweeps
        seeds = np.where(err == 1)[0]
        for s in seeds:
            for nb in (s - 1, s + 1):
                if 0 <= nb < n and rng.random() < corr:
                    err[nb] = 1
    return err


def sample_measurement_error(
    n: int, p: float, rng: np.random.Generator, q: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(data_error, syndrome_flip_mask_for_n-1_checks)``.

    The data error is iid at rate ``p``; the syndrome read-out is corrupted at
    rate ``q`` (the caller XORs the flip mask into the measured syndrome).
    """
    err = sample_iid(n, p, rng)
    synd_flips = (rng.random(max(0, n - 1)) < q).astype(np.int8)
    return err, synd_flips


def sample(
    regime: str,
    n: int,
    p: float,
    rng: np.random.Generator,
    *,
    bias: float = 3.0,
    burst_len: int = 3,
    corr: float = 0.5,
    q: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Dispatch to a regime. Returns ``(error, syndrome_flips_or_None)``."""
    if regime == "iid":
        return sample_iid(n, p, rng), None
    if regime == "biased":
        return sample_biased(n, p, rng, bias=bias), None
    if regime == "burst":
        return sample_burst(n, p, rng, burst_len=burst_len), None
    if regime == "correlated":
        return sample_correlated(n, p, rng, corr=corr), None
    if regime == "measurement_error":
        e, sf = sample_measurement_error(n, p, rng, q=q)
        return e, sf
    raise ValueError(f"unknown noise regime {regime!r}")
