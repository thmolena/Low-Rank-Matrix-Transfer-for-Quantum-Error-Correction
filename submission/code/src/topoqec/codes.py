"""Stabilizer codes used as decoding benchmarks.

Two small, fully explicit codes back every experiment:

  * :class:`RepetitionCode` -- the bit-flip repetition code on ``d`` data bits
    with ``d-1`` weight-2 parity checks ``Z_i Z_{i+1}``. It corrects up to
    ``(d-1)//2`` bit flips and has a single logical operator (global parity).
    Everything here is over GF(2): errors and corrections are 0/1 numpy arrays,
    addition is XOR.

  * :class:`SurfaceLikeCode` -- a ``d x d`` toy lattice whose plaquette checks
    fire on the boundary of an error chain. Defects are matched pairwise by
    Manhattan distance (a minimum-weight-matching proxy for the surface-code
    decoding problem). It is deliberately small and self-contained; it is *not*
    a full CSS surface code, but it exposes the same defect-geometry structure
    that the topology-aware decoder exploits.

The Tanner graph (data bits <-> checks) is exposed via :meth:`tanner_graph`, so
the message-passing and matching decoders can operate on the code topology
directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- #
# Repetition code
# --------------------------------------------------------------------------- #
@dataclass
class RepetitionCode:
    """Distance-``d`` bit-flip repetition code.

    Data bits ``0..d-1``; check ``i`` is the weight-2 parity ``Z_i Z_{i+1}`` for
    ``i in 0..d-2``. A syndrome bit is 1 iff its two data bits disagree, i.e. an
    error chain crossed that link.
    """

    distance: int

    def __post_init__(self) -> None:
        if self.distance < 1:
            raise ValueError("distance must be >= 1")
        self.n_data = self.distance
        self.n_checks = self.distance - 1

    # -- parity / syndrome ------------------------------------------------- #
    def syndrome(self, error: np.ndarray) -> np.ndarray:
        """Defect pattern: ``s_i = e_i XOR e_{i+1}`` for each link ``i``."""
        e = np.asarray(error, dtype=np.int8).ravel()
        if e.size != self.n_data:
            raise ValueError(f"error length {e.size} != n_data {self.n_data}")
        return (e[:-1] ^ e[1:]).astype(np.int8)

    def logical_value(self, bits: np.ndarray) -> int:
        """The single logical observable = global parity of the data bits."""
        return int(np.asarray(bits, dtype=np.int8).sum() % 2)

    def logical_error(self, error: np.ndarray, correction: np.ndarray) -> int:
        """1 if ``error XOR correction`` is a non-trivial logical operator.

        After correction the residual ``r = e ^ c`` must be a *stabilizer* (here:
        all-zeros or all-ones, the two trivial cosets) to be harmless. A residual
        of odd parity flips the logical bit -> a logical error.
        """
        e = np.asarray(error, dtype=np.int8).ravel()
        c = np.asarray(correction, dtype=np.int8).ravel()
        residual = e ^ c
        # A correction that does not even reproduce the syndrome is invalid;
        # such a residual is not a stabilizer and counts as a logical failure.
        if not np.array_equal(self.syndrome(residual), np.zeros(self.n_checks, np.int8)):
            return 1
        return self.logical_value(residual)

    # -- topology ---------------------------------------------------------- #
    def tanner_graph(self) -> nx.Graph:
        """Bipartite data<->check graph; node attr ``kind`` in {data, check}."""
        g = nx.Graph()
        for i in range(self.n_data):
            g.add_node(("data", i), kind="data")
        for j in range(self.n_checks):
            g.add_node(("check", j), kind="check")
            g.add_edge(("check", j), ("data", j))
            g.add_edge(("check", j), ("data", j + 1))
        return g

    def all_errors(self) -> np.ndarray:
        """Enumerate all ``2**d`` error patterns (used by the lookup decoder)."""
        n = self.n_data
        table = np.zeros((1 << n, n), dtype=np.int8)
        for m in range(1 << n):
            for b in range(n):
                table[m, b] = (m >> b) & 1
        return table


# --------------------------------------------------------------------------- #
# Surface-like toy code
# --------------------------------------------------------------------------- #
@dataclass
class SurfaceLikeCode:
    """A ``d x d`` lattice of data bits with plaquette defects.

    Data bits sit on a ``d x d`` grid. Each interior plaquette (a 2x2 cell, of
    which there are ``(d-1) x (d-1)``) checks the parity of its four corners; a
    defect (lit plaquette) marks the boundary of an error region. Defects are
    decoded by pairing them with a minimum-weight matching under Manhattan
    distance, mirroring the surface-code MWPM decoder.

    This is a topology-faithful *proxy*, not a stabilizer-exact surface code:
    the logical observable is the parity of the left lattice column, and a
    correction fails when the residual carries an odd number of crossings of a
    horizontal logical cut.
    """

    distance: int

    def __post_init__(self) -> None:
        if self.distance < 2:
            raise ValueError("surface-like distance must be >= 2")
        d = self.distance
        self.n_data = d * d
        self.plaquettes: List[Tuple[int, int, int, int]] = []
        self.plaq_pos: List[Tuple[float, float]] = []
        for r in range(d - 1):
            for c in range(d - 1):
                a = r * d + c
                b = r * d + c + 1
                e = (r + 1) * d + c
                f = (r + 1) * d + c + 1
                self.plaquettes.append((a, b, e, f))
                self.plaq_pos.append((r + 0.5, c + 0.5))
        self.n_checks = len(self.plaquettes)

    def syndrome(self, error: np.ndarray) -> np.ndarray:
        e = np.asarray(error, dtype=np.int8).ravel()
        if e.size != self.n_data:
            raise ValueError(f"error length {e.size} != n_data {self.n_data}")
        out = np.zeros(self.n_checks, dtype=np.int8)
        for k, (a, b, c, dd) in enumerate(self.plaquettes):
            out[k] = (e[a] ^ e[b] ^ e[c] ^ e[dd])
        return out

    def defect_positions(self, syndrome: np.ndarray) -> List[Tuple[int, Tuple[float, float]]]:
        """Lit-plaquette indices paired with their lattice (row, col) centres."""
        s = np.asarray(syndrome, dtype=np.int8).ravel()
        return [(k, self.plaq_pos[k]) for k in range(self.n_checks) if s[k]]

    def logical_value(self, bits: np.ndarray) -> int:
        """Logical observable: parity of the left column of the lattice."""
        b = np.asarray(bits, dtype=np.int8).ravel().reshape(self.distance, self.distance)
        return int(b[:, 0].sum() % 2)

    def logical_error(self, error: np.ndarray, correction: np.ndarray) -> int:
        e = np.asarray(error, dtype=np.int8).ravel()
        c = np.asarray(correction, dtype=np.int8).ravel()
        residual = e ^ c
        if not np.array_equal(self.syndrome(residual), np.zeros(self.n_checks, np.int8)):
            return 1
        return self.logical_value(residual)

    def tanner_graph(self) -> nx.Graph:
        g = nx.Graph()
        for i in range(self.n_data):
            g.add_node(("data", i), kind="data")
        for k, corners in enumerate(self.plaquettes):
            g.add_node(("check", k), kind="check")
            for v in corners:
                g.add_edge(("check", k), ("data", v))
        return g


def make_code(kind: str, distance: int):
    """Factory: ``"repetition"`` or ``"surface"``."""
    if kind == "repetition":
        return RepetitionCode(distance)
    if kind == "surface":
        return SurfaceLikeCode(distance)
    raise ValueError(f"unknown code kind {kind!r}")
