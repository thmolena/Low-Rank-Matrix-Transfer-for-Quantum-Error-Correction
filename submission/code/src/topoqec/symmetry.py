"""Code automorphisms and a decoder-invariance check.

The repetition code is symmetric under cyclic relabeling of its data bits
(viewed as a ring): rotating every error pattern by a fixed offset and rotating
the syndrome to match leaves the logical error rate unchanged. A correct decoder
must therefore be (statistically) invariant under that automorphism. The runner
exposes this as an integrity flag.
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np

from .codes import RepetitionCode


def cyclic_relabel(error: np.ndarray, shift: int) -> np.ndarray:
    """Rotate a data-bit error pattern by ``shift`` positions (ring symmetry)."""
    return np.roll(np.asarray(error, dtype=np.int8), shift)


def logical_error_rate_under_relabeling(
    code: RepetitionCode,
    decode: Callable[[np.ndarray], "tuple"],
    errors: List[np.ndarray],
    shift: int,
) -> float:
    """Logical error rate when every error is cyclically relabeled by ``shift``.

    For each relabeled error we recompute the syndrome, decode it, and measure
    the logical error. The repetition-code logical observable (global parity) is
    invariant under rotation, so a faithful decoder yields the same rate as on
    the unrotated errors (up to boundary effects, which we count honestly).
    """
    fails = []
    for e in errors:
        er = cyclic_relabel(e, shift)
        s = code.syndrome(er)
        corr, _conf = decode(s)
        fails.append(code.logical_error(er, corr))
    return float(np.mean(fails)) if fails else 0.0


def is_automorphism_invariant(
    code: RepetitionCode,
    decode: Callable[[np.ndarray], "tuple"],
    errors: List[np.ndarray],
    shifts: List[int],
    tol: float = 0.05,
) -> bool:
    """True if the logical error rate is stable across cyclic relabelings.

    Compares the rate at each requested ``shift`` to the unshifted baseline and
    requires every deviation to stay within ``tol`` (a statistical tolerance for
    finite Monte-Carlo samples).
    """
    base = logical_error_rate_under_relabeling(code, decode, errors, 0)
    for sh in shifts:
        rate = logical_error_rate_under_relabeling(code, decode, errors, sh)
        if abs(rate - base) > tol:
            return False
    return True
