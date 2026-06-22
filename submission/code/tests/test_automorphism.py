"""Cyclic-relabeling automorphism leaves the logical error rate invariant."""
import numpy as np

from topoqec import noise
from topoqec.codes import RepetitionCode
from topoqec.decoders import MatchingDecoder
from topoqec.symmetry import (
    cyclic_relabel,
    is_automorphism_invariant,
    logical_error_rate_under_relabeling,
)


def test_cyclic_relabel_roundtrip():
    err = np.array([1, 0, 0, 1, 0], np.int8)
    assert np.array_equal(cyclic_relabel(cyclic_relabel(err, 2), -2), err)


def test_logical_error_rate_invariant_under_relabeling():
    code = RepetitionCode(7)
    dec = MatchingDecoder(code)
    rng = np.random.default_rng(11)
    errors = [noise.sample_iid(code.n_data, 0.12, rng) for _ in range(800)]
    base = logical_error_rate_under_relabeling(code, dec.decode, errors, 0)
    for shift in (1, 2, 3):
        rate = logical_error_rate_under_relabeling(code, dec.decode, errors, shift)
        assert abs(rate - base) < 0.06, (shift, rate, base)


def test_is_automorphism_invariant_flag():
    code = RepetitionCode(9)
    dec = MatchingDecoder(code)
    rng = np.random.default_rng(5)
    errors = [noise.sample_iid(code.n_data, 0.10, rng) for _ in range(600)]
    assert is_automorphism_invariant(
        code, dec.decode, errors, shifts=[1, 2, code.n_data // 2], tol=0.07
    )
