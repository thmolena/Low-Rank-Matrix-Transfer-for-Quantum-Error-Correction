"""Minimal GF(2) linear algebra used to build and validate the CSS code.

Everything here operates on 0/1 integer numpy arrays and works modulo 2.
The routines are small and deliberately readable rather than fast: they run
once, at code-construction time, on matrices no larger than (d^2) columns.
"""
from __future__ import annotations

import numpy as np


def row_reduce(mat: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return (reduced row echelon form over GF(2), list of pivot columns)."""
    m = (np.asarray(mat) % 2).astype(np.uint8).copy()
    n_rows, n_cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        # find a pivot in column c at or below row r
        piv = None
        for i in range(r, n_rows):
            if m[i, c]:
                piv = i
                break
        if piv is None:
            continue
        m[[r, piv]] = m[[piv, r]]
        # eliminate this column from every other row
        for i in range(n_rows):
            if i != r and m[i, c]:
                m[i, :] ^= m[r, :]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return m, pivots


def rank(mat: np.ndarray) -> int:
    _, pivots = row_reduce(mat)
    return len(pivots)


def nullspace(mat: np.ndarray) -> np.ndarray:
    """Basis (as rows) of the right null space of `mat` over GF(2)."""
    m = (np.asarray(mat) % 2).astype(np.uint8)
    n_cols = m.shape[1]
    rref, pivots = row_reduce(m)
    pivot_set = set(pivots)
    free = [c for c in range(n_cols) if c not in pivot_set]
    basis = []
    for f in free:
        vec = np.zeros(n_cols, dtype=np.uint8)
        vec[f] = 1
        for r, pc in enumerate(pivots):
            if rref[r, f]:
                vec[pc] = 1
        basis.append(vec)
    return np.array(basis, dtype=np.uint8) if basis else np.zeros((0, n_cols), np.uint8)


def in_rowspace(vec: np.ndarray, mat: np.ndarray) -> bool:
    """True iff `vec` lies in the GF(2) row space of `mat`."""
    stacked = np.vstack([np.asarray(mat) % 2, (np.asarray(vec) % 2)[None, :]])
    return rank(stacked) == rank(mat)
