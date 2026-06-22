"""Metric sanity: LER in [0,1], ECE in [0,1], risk-coverage monotone."""
import numpy as np

from topoqec import metrics


def test_logical_error_rate_bounds():
    rng = np.random.default_rng(0)
    for _ in range(20):
        fails = (rng.random(100) < 0.3).astype(int)
        ler = metrics.logical_error_rate(fails)
        assert 0.0 <= ler <= 1.0
    assert metrics.logical_error_rate([]) == 0.0
    assert metrics.logical_error_rate([1, 1, 1]) == 1.0
    assert metrics.logical_error_rate([0, 0, 0]) == 0.0


def test_ece_in_unit_interval():
    rng = np.random.default_rng(1)
    # perfectly calibrated: confidence == accuracy in expectation
    conf = rng.random(2000)
    corrects = (rng.random(2000) < conf).astype(int)
    ece = metrics.expected_calibration_error(conf, corrects, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.1  # near-calibrated by construction
    # overconfident: always claim 1.0 but only 50% correct -> ECE ~ 0.5
    conf2 = np.ones(1000)
    corr2 = (rng.random(1000) < 0.5).astype(int)
    ece2 = metrics.expected_calibration_error(conf2, corr2)
    assert 0.0 <= ece2 <= 1.0
    assert ece2 > 0.3


def test_risk_coverage_monotone_and_bounded():
    rng = np.random.default_rng(2)
    n = 2000
    conf = rng.random(n)
    # well-ordered: low-confidence trials fail more often
    fail = (rng.random(n) > conf).astype(int)
    curve = metrics.risk_coverage_curve(conf, fail, n_points=21)
    assert curve
    for cov, risk in curve:
        assert 0.0 <= cov <= 1.0
        assert 0.0 <= risk <= 1.0
    assert metrics.is_monotone_risk_coverage(curve)


def test_risk_coverage_empty():
    assert metrics.risk_coverage_curve([], []) == []
