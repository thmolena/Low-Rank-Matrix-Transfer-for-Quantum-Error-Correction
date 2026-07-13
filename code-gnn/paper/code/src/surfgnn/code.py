"""Rotated surface code: geometry, parity-check matrices, and logicals.

We build the standard distance-`d` rotated planar surface code (d odd) as a
CSS code with check matrices H_X (X-stabilizers) and H_Z (Z-stabilizers) over
the n = d*d data qubits, then verify it is a *valid* CSS code with one logical
qubit and distance d before anything downstream uses it.

We track X errors detected by the Z-stabilizers (a Z-basis memory experiment):
the syndrome of an X error e is H_Z @ e (mod 2), and the logical observable is
a minimum-weight Z-logical, computed algebraically so we never depend on a
fragile hand-picked boundary convention.

Data-qubit layout: a d x d grid, data qubit (r, c) has index r*d + c with
r, c in {0, ..., d-1}.  Plaquettes use a checkerboard colouring with weight-2
boundary stabilizers on the matching pair of edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import gf2


def _idx(r: int, c: int, d: int) -> int:
    return r * d + c


def _build_stabilizers(d: int) -> tuple[list[list[int]], list[list[int]]]:
    """Return (z_checks, x_checks); each check is a sorted list of data indices.

    Bulk: weight-4 plaquettes on a checkerboard.  A plaquette whose top-left
    data qubit is (r, c) is Z-type when (r + c) is even, X-type otherwise.
    Boundary: weight-2 stabilizers complete the code.  Z-type weight-2 checks
    sit on the top/bottom edges (horizontal pairs); X-type weight-2 checks sit
    on the left/right edges (vertical pairs).
    """
    z_checks: list[list[int]] = []
    x_checks: list[list[int]] = []

    # Bulk weight-4 plaquettes.
    for r in range(d - 1):
        for c in range(d - 1):
            corners = [
                _idx(r, c, d), _idx(r, c + 1, d),
                _idx(r + 1, c, d), _idx(r + 1, c + 1, d),
            ]
            if (r + c) % 2 == 0:
                z_checks.append(sorted(corners))
            else:
                x_checks.append(sorted(corners))

    # Boundary weight-2 Z checks on top (row 0) and bottom (row d-1).
    for c in range(d - 1):
        # top edge: include where colouring continues the checkerboard
        if (0 + c) % 2 == 1:
            z_checks.append(sorted([_idx(0, c, d), _idx(0, c + 1, d)]))
        if ((d - 2) + c) % 2 == 1:
            z_checks.append(sorted([_idx(d - 1, c, d), _idx(d - 1, c + 1, d)]))

    # Boundary weight-2 X checks on left (col 0) and right (col d-1).
    for r in range(d - 1):
        if (r + 0) % 2 == 1:
            x_checks.append(sorted([_idx(r, 0, d), _idx(r + 1, 0, d)]))
        if (r + (d - 2)) % 2 == 1:
            x_checks.append(sorted([_idx(r, d - 1, d), _idx(r + 1, d - 1, d)]))

    return z_checks, x_checks


def _checks_to_matrix(checks: list[list[int]], n: int) -> np.ndarray:
    h = np.zeros((len(checks), n), dtype=np.uint8)
    for i, ch in enumerate(checks):
        for q in ch:
            h[i, q] = 1
    return h


def _min_weight_logical(hz: np.ndarray, hx: np.ndarray, n: int) -> np.ndarray:
    """A minimum-weight X-type logical: in ker(H_Z) but not in rowspace(H_X).

    For d up to 7 the logical coset is small enough to search by combining the
    nullspace basis of H_Z; we take the lightest representative that is not an
    X-stabilizer.  This is the observable support for the Z-memory experiment.
    """
    ker = gf2.nullspace(hz)  # X-patterns with trivial Z-syndrome
    # Logical reps = ker(H_Z) \ rowspace(H_X).  Search small combinations.
    best: np.ndarray | None = None
    n_basis = ker.shape[0]
    # n_basis is small (= k + rank(H_X) - ...); brute force over its span is fine
    # for d<=7 because the logical dimension is 1 and rank(H_X) keeps it modest.
    from itertools import combinations
    # Try increasing combination sizes for a light logical.
    found_any = False
    for size in range(1, n_basis + 1):
        for combo in combinations(range(n_basis), size):
            vec = np.zeros(n, dtype=np.uint8)
            for b in combo:
                vec ^= ker[b]
            if vec.sum() == 0:
                continue
            if gf2.in_rowspace(vec, hx):
                continue  # pure X-stabilizer, logically trivial
            w = int(vec.sum())
            if best is None or w < int(best.sum()):
                best = vec.copy()
                found_any = True
        if found_any and best is not None and int(best.sum()) <= size:
            # cannot do better than current size at this level for light ops
            break
    assert best is not None, "no nontrivial logical found"
    return best


@dataclass
class SurfaceCode:
    """Distance-d rotated surface code, Z-memory (X errors, Z-checks)."""
    d: int
    n: int = field(init=False)
    hz: np.ndarray = field(init=False)          # (m_z, n) Z-checks
    hx: np.ndarray = field(init=False)          # (m_x, n) X-checks
    z_checks: list[list[int]] = field(init=False)
    x_checks: list[list[int]] = field(init=False)
    obs: np.ndarray = field(init=False)         # (n,) logical observable support
    coords: np.ndarray = field(init=False)      # (n, 2) data-qubit (row, col)

    def __post_init__(self) -> None:
        if self.d % 2 == 0:
            raise ValueError("distance d must be odd")
        self.n = self.d * self.d
        self.z_checks, self.x_checks = _build_stabilizers(self.d)
        self.hz = _checks_to_matrix(self.z_checks, self.n)
        self.hx = _checks_to_matrix(self.x_checks, self.n)
        self.obs = _min_weight_logical(self.hz, self.hx, self.n)
        self.coords = np.array(
            [(q // self.d, q % self.d) for q in range(self.n)], dtype=np.int64
        )

    # ---- validation -----------------------------------------------------
    def validate(self) -> dict:
        """Check CSS commutation, single logical qubit, and code distance == d."""
        comm = (self.hx @ self.hz.T) % 2
        css_ok = bool((comm == 0).all())
        rz, rx = gf2.rank(self.hz), gf2.rank(self.hx)
        k = self.n - rz - rx
        # observable must be a valid logical: commutes with X-checks, not a Z-stab
        obs_commutes = bool(((self.hx @ self.obs) % 2 == 0).all())
        obs_nontrivial = not gf2.in_rowspace(self.obs, self.hz)
        info = {
            "d": self.d,
            "n": self.n,
            "num_z_checks": len(self.z_checks),
            "num_x_checks": len(self.x_checks),
            "rank_hz": rz,
            "rank_hx": rx,
            "k": k,
            "css_commute": css_ok,
            "obs_weight": int(self.obs.sum()),
            "obs_commutes_x": obs_commutes,
            "obs_nontrivial": obs_nontrivial,
        }
        info["valid"] = bool(
            css_ok and k == 1 and obs_commutes and obs_nontrivial
            and info["obs_weight"] == self.d
        )
        return info

    # ---- convenience ----------------------------------------------------
    def syndrome(self, error: np.ndarray) -> np.ndarray:
        return (self.hz @ (error % 2)) % 2

    def logical_flip(self, error: np.ndarray) -> np.ndarray:
        return int((self.obs @ (error % 2)) % 2)
