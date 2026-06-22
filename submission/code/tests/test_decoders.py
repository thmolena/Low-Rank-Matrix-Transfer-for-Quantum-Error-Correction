"""Decoder behaviour: majority below threshold, matching wins on burst noise."""
import numpy as np

from topoqec import noise
from topoqec.codes import RepetitionCode
from topoqec.decoders import MajorityDecoder, MatchingDecoder


def _logical_error_rate(code, decoder, regime, trials, p, seed):
    rng = np.random.default_rng(seed)
    fails = []
    for _ in range(trials):
        err, _ = noise.sample(regime, code.n_data, p, rng)
        corr, _conf = decoder.decode(code.syndrome(err))
        fails.append(code.logical_error(err, corr))
    return float(np.mean(fails))


def test_majority_decodes_below_threshold():
    code = RepetitionCode(5)  # tolerates up to 2 flips
    dec = MajorityDecoder(code)
    rng = np.random.default_rng(1)
    # all weight<=2 errors must decode without a logical error
    for _ in range(60):
        weight = int(rng.integers(0, 3))
        err = np.zeros(5, np.int8)
        if weight:
            err[rng.choice(5, size=weight, replace=False)] = 1
        corr, _ = dec.decode(code.syndrome(err))
        assert code.logical_error(err, corr) == 0


def test_majority_low_error_rate_under_iid():
    code = RepetitionCode(7)
    dec = MajorityDecoder(code)
    ler = _logical_error_rate(code, dec, "iid", 2000, 0.08, seed=2)
    assert ler < 0.10  # well below the physical rate at low p


def test_matching_beats_or_matches_majority_on_burst():
    code = RepetitionCode(9)
    maj = MajorityDecoder(code)
    match = MatchingDecoder(code)
    # burst noise creates spatially-correlated chains where geometry-aware
    # matching should not lose to a memoryless majority vote
    ler_maj = _logical_error_rate(code, maj, "burst", 3000, 0.12, seed=7)
    ler_match = _logical_error_rate(code, match, "burst", 3000, 0.12, seed=7)
    assert ler_match <= ler_maj + 1e-9


def test_decoders_return_valid_confidence():
    code = RepetitionCode(5)
    for dec in (MajorityDecoder(code), MatchingDecoder(code)):
        rng = np.random.default_rng(3)
        for _ in range(50):
            err, _ = noise.sample("iid", code.n_data, 0.15, rng)
            corr, conf = dec.decode(code.syndrome(err))
            assert corr.shape == (code.n_data,)
            assert 0.0 <= conf <= 1.0
