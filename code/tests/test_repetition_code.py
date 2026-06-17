"""Repetition-code identities: syndromes, single-flip correction, distance bound."""
import numpy as np

from topoqec.codes import RepetitionCode, SurfaceLikeCode
from topoqec.decoders import LookupDecoder


def test_syndrome_known_patterns():
    code = RepetitionCode(5)
    # no error -> no defects
    assert np.array_equal(code.syndrome(np.zeros(5, np.int8)), np.zeros(4, np.int8))
    # a single flip in the middle lights its two adjacent checks
    err = np.array([0, 0, 1, 0, 0], np.int8)
    assert np.array_equal(code.syndrome(err), np.array([0, 1, 1, 0], np.int8))
    # a flip at the boundary lights one check
    err0 = np.array([1, 0, 0, 0, 0], np.int8)
    assert np.array_equal(code.syndrome(err0), np.array([1, 0, 0, 0], np.int8))


def test_single_middle_flip_is_corrected():
    code = RepetitionCode(5)
    dec = LookupDecoder(code)
    err = np.array([0, 0, 1, 0, 0], np.int8)
    corr, _ = dec.decode(code.syndrome(err))
    assert code.logical_error(err, corr) == 0
    # the recovered residual is a stabilizer (all-zeros here)
    assert np.array_equal((err ^ corr), np.zeros(5, np.int8))


def test_logical_error_only_above_distance_threshold():
    code = RepetitionCode(5)  # corrects up to (d-1)//2 = 2 flips
    dec = LookupDecoder(code)
    rng = np.random.default_rng(0)
    # weight-1 and weight-2 errors must always be corrected
    for weight in (1, 2):
        for _ in range(40):
            idx = rng.choice(5, size=weight, replace=False)
            err = np.zeros(5, np.int8)
            err[idx] = 1
            corr, _ = dec.decode(code.syndrome(err))
            assert code.logical_error(err, corr) == 0, (weight, idx)
    # a majority of bits flipped (weight 3 of 5) is beyond the distance and
    # the minimum-weight leader produces a logical error
    err = np.array([1, 1, 1, 0, 0], np.int8)
    corr, _ = dec.decode(code.syndrome(err))
    assert code.logical_error(err, corr) == 1


def test_lookup_is_minimum_weight():
    code = RepetitionCode(4)
    dec = LookupDecoder(code)
    # every leader must reproduce the syndrome it is indexed by
    for m in range(1 << 4):
        err = np.array([(m >> b) & 1 for b in range(4)], np.int8)
        s = code.syndrome(err)
        corr, _ = dec.decode(s)
        assert np.array_equal(code.syndrome(corr), s)


def test_surface_like_code_basic():
    code = SurfaceLikeCode(3)
    assert code.n_data == 9
    assert code.n_checks == 4  # (3-1)*(3-1)
    # no error -> no lit plaquettes
    assert np.array_equal(code.syndrome(np.zeros(9, np.int8)), np.zeros(4, np.int8))
    # a single interior-corner flip lights the plaquettes that touch it
    err = np.zeros(9, np.int8)
    err[4] = 1  # centre cell touches all four plaquettes on a 3x3 lattice
    assert int(code.syndrome(err).sum()) == 4
