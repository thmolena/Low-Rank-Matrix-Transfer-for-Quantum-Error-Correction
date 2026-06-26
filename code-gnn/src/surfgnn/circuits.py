"""Stim circuits for the rotated surface-code Z-memory experiment, with an
explicit *correlated*-noise knob whose faults propagate to BOTH the detectors
and the logical observable (so the measured logical error rate is the true
memory-experiment LER, not a post-hoc artifact).

Design
------
- The base is Stim's standard circuit-level-noise rotated surface code, which
  on its own gives a correct threshold and a graphlike detector error model
  (DEM) that minimum-weight perfect matching handles near-optimally.
- The *correlated* regime adds, at the start of every round, a layer of
  two-qubit XX errors on neighbouring data qubits.  Each such fault flips up to
  four Z-detectors jointly.  Under Stim's `decompose_errors=True`, matching sees
  these only as a pair of independent graphlike edges and therefore cannot use
  the joint structure -- this is the precise sense in which matching, built on
  an independent-error model, is suboptimal under genuine correlations.
- BP+OSD is given the full (non-decomposed) DEM and can exploit the hyperedges;
  the neural decoder learns the joint structure directly from samples.  All
  decoders are evaluated on identical shots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


def _data_qubits(circuit: stim.Circuit) -> list[int]:
    """Data qubits = targets of the final (non-reset) measurement `M`."""
    flat = circuit.flattened()
    data: list[int] = []
    for inst in flat:
        if inst.name == "M":
            data = [t.value for t in inst.targets_copy()]
    return data


def _coords(circuit: stim.Circuit) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for inst in circuit.flattened():
        if inst.name == "QUBIT_COORDS":
            xy = tuple(inst.gate_args_copy())
            for t in inst.targets_copy():
                out[t.value] = (float(xy[0]), float(xy[1]))
    return out


def neighbour_pairs(data: list[int], coords: dict[int, tuple[float, float]]) -> list[tuple[int, int]]:
    """Adjacent data-qubit pairs (Stim coords differ by 2 along one axis)."""
    pairs: list[tuple[int, int]] = []
    for a in range(len(data)):
        for b in range(a + 1, len(data)):
            qa, qb = data[a], data[b]
            (xa, ya), (xb, yb) = coords[qa], coords[qb]
            if abs(xa - xb) + abs(ya - yb) == 2.0:
                pairs.append((qa, qb))
    return pairs


def _inject_correlated(circuit: stim.Circuit, pairs: list[tuple[int, int]],
                       p_corr: float) -> stim.Circuit:
    """Insert correlated XX faults on neighbour pairs at the start of each round.

    A round boundary is the initial reset `R` and every ancilla measure-reset
    `MR`; injecting data XX errors there means they evolve through syndrome
    extraction and accumulate onto the final data measurement, so they flip
    detectors and the observable consistently.
    """
    if p_corr <= 0.0:
        return circuit
    flat = circuit.flattened()
    out = stim.Circuit()

    def append_corr(c: stim.Circuit) -> None:
        for (i, j) in pairs:
            c.append("CORRELATED_ERROR", [stim.target_x(i), stim.target_x(j)], p_corr)

    for inst in flat:
        out.append(inst)
        if inst.name in ("R", "MR"):
            append_corr(out)
    return out


@dataclass
class Benchmark:
    d: int
    rounds: int
    p: float
    p_corr: float
    circuit: stim.Circuit
    data: list[int]
    pairs: list[tuple[int, int]]

    def detector_sampler(self):
        return self.circuit.compile_detector_sampler()

    def dem(self, decompose: bool) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(
            decompose_errors=decompose,
            allow_gauge_detectors=False,
            approximate_disjoint_errors=True,
        )


def build_benchmark(d: int, rounds: int | None = None, p: float = 0.005,
                    p_corr: float = 0.0) -> Benchmark:
    """Rotated surface-code Z-memory benchmark; `p_corr>0` adds correlated XX."""
    if rounds is None:
        rounds = d
    base = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    data = _data_qubits(base)
    coords = _coords(base)
    pairs = neighbour_pairs(data, coords)
    circuit = _inject_correlated(base, pairs, p_corr)
    return Benchmark(d=d, rounds=rounds, p=p, p_corr=p_corr,
                     circuit=circuit, data=data, pairs=pairs)


def sample(benchmark: Benchmark, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (detectors [shots, n_det] bool, observables [shots, n_obs] bool)."""
    sampler = benchmark.circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return det, obs
