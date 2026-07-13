"""Classical baselines: minimum-weight perfect matching (PyMatching) and
BP+OSD (ldpc), both decoding from the Stim detector error model (DEM).

MWPM consumes the *decomposed* DEM (graphlike edges -- its native, and the
realistic, independent-error view).  BP+OSD consumes the *full* DEM including
the correlated hyperedges, so it is the strong, correctly-specified baseline.
Both decode the same detector samples; we score the logical observable.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import stim
import pymatching
from ldpc import BpOsdDecoder


def dem_to_matrices(dem: stim.DetectorErrorModel):
    """Return (H [n_det x n_err], O [n_obs x n_err], probs [n_err]) as CSR."""
    n_det = dem.num_detectors
    n_obs = dem.num_observables
    rows_h, cols_h = [], []
    rows_o, cols_o = [], []
    probs: list[float] = []
    col = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obss = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obss.append(t.val)
        for di in dets:
            rows_h.append(di); cols_h.append(col)
        for oi in obss:
            rows_o.append(oi); cols_o.append(col)
        probs.append(p)
        col += 1
    n_err = col
    H = sp.csr_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)),
                      shape=(n_det, n_err))
    O = sp.csr_matrix((np.ones(len(rows_o), np.uint8), (rows_o, cols_o)),
                      shape=(n_obs, n_err))
    return H, O, np.array(probs, dtype=np.float64)


class MWPMDecoder:
    """Minimum-weight perfect matching on the decomposed DEM."""

    def __init__(self, dem_decomposed: stim.DetectorErrorModel):
        self.m = pymatching.Matching.from_detector_error_model(dem_decomposed)

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        """detectors [shots, n_det] -> predicted observables [shots, n_obs]."""
        return self.m.decode_batch(detectors.astype(np.uint8))


class BPOSDDecoder:
    """BP + ordered-statistics decoding on the full (correlated) DEM."""

    def __init__(self, dem_full: stim.DetectorErrorModel, max_iter: int = 30,
                 osd_order: int = 6):
        self.H, self.O, self.probs = dem_to_matrices(dem_full)
        self.dec = BpOsdDecoder(
            self.H,
            error_channel=list(self.probs),
            max_iter=max_iter,
            bp_method="ms",
            osd_method="osd_cs",
            osd_order=osd_order,
        )

    def predict(self, detectors: np.ndarray) -> np.ndarray:
        shots = detectors.shape[0]
        n_obs = self.O.shape[0]
        out = np.zeros((shots, n_obs), dtype=np.uint8)
        for i in range(shots):
            e = self.dec.decode(detectors[i].astype(np.uint8))
            out[i] = (self.O @ e) % 2
        return out


def logical_error_rate(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.any(pred != obs, axis=1)))
