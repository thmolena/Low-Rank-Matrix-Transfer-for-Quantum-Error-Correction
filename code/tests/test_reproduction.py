from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hqml_transfer_qec import experiment
from hqml_transfer_qec.baselines import (
    binary_metrics,
    class_balanced_weights,
    hypercube_kernel_apply,
)
from hqml_transfer_qec.bounds import (
    clipped_log_loss_perturbation_bound,
    normalized_prediction_error_bound,
    two_rank_contraction_audit,
)
from hqml_transfer_qec.matrix_free import select_rank_counterfactual_blend


@pytest.fixture(scope="session")
def reproduced() -> dict:
    return experiment.reproduce()


def test_authenticated_dataset_contract(reproduced: dict) -> None:
    assert reproduced["dataset_sha256"] == experiment.DATA_SHA256
    assert reproduced["rows"] == 75_598
    assert reproduced["shots"] == 70_000_000
    assert reproduced["noise_rates"] == 7
    assert reproduced["dataset_origin"].startswith("Stim simulation")


def test_locked_scientific_results_replay(reproduced: dict) -> None:
    # verify ignores host/platform timings but checks every scientific number.
    experiment.verify(reproduced)


def test_locked_results_are_byte_identical() -> None:
    results = Path(experiment.PACKAGE_ROOT) / "results"
    assert (results / "results.json").read_bytes() == (
        results / "locked_results.json"
    ).read_bytes()


def test_hypercube_transform_equals_dense_kernel() -> None:
    bits = 4
    rho = 0.23
    values = np.arange(1 << bits, dtype=float) / 7.0
    indices = np.arange(1 << bits, dtype=int)
    xor = np.bitwise_xor(indices[:, None], indices[None, :])
    distances = np.frompyfunc(int.bit_count, 1, 1)(xor).astype(int)
    dense = rho**distances
    np.testing.assert_allclose(
        hypercube_kernel_apply(values, rho, bits=bits), dense @ values, atol=1e-13
    )


def test_class_balancing_preserves_shot_multiplicity() -> None:
    labels = np.array([0, 0, 1, 1])
    quantities = np.array([1.0, 9.0, 2.0, 8.0])
    weights = class_balanced_weights(labels, quantities)
    np.testing.assert_allclose([weights[labels == c].sum() for c in (0, 1)], 0.5)
    assert weights[1] / weights[0] == pytest.approx(9.0)
    assert weights[3] / weights[2] == pytest.approx(4.0)


def test_balanced_metrics_are_quantity_weighted() -> None:
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([1.0, 9.0, 1.0, 9.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
    )
    assert metrics["balanced_accuracy"] == pytest.approx(0.55)


def test_normalization_bound_covers_random_perturbations() -> None:
    rng = np.random.default_rng(12)
    numerator = rng.normal(size=(31, 3))
    denominator = rng.uniform(0.5, 2.0, size=31)
    approximate_numerator = numerator + 1e-3 * rng.normal(size=numerator.shape)
    approximate_denominator = denominator + 1e-3 * rng.normal(size=31)
    audit = normalized_prediction_error_bound(
        numerator, denominator, approximate_numerator, approximate_denominator
    )
    assert audit.covered
    assert np.all(audit.actual_row_errors <= audit.row_bounds * (1 + 1e-12))


def test_two_rank_bound_is_explicitly_conditional() -> None:
    reference = np.zeros(2)
    passing = two_rank_contraction_audit(
        np.array([2.0, 0.0]), np.array([0.5, 0.0]), reference
    )
    failing = two_rank_contraction_audit(
        np.array([0.5, 0.0]), np.array([2.0, 0.0]), reference
    )
    assert passing.condition_observed and passing.covered
    assert not failing.condition_observed and not failing.covered
    assert np.isinf(failing.error_bound)


def test_clipped_log_loss_lipschitz_bound() -> None:
    audit = clipped_log_loss_perturbation_bound(
        np.array([0, 1, 1]),
        np.array([0.1, 0.8, 0.9]),
        np.array([0.11, 0.78, 0.88]),
        np.array([1.0, 2.0, 3.0]),
        epsilon=1e-3,
    )
    assert audit.covered
    assert audit.actual_difference <= audit.bound


def test_failed_operator_diagnostic_forces_zero_blend() -> None:
    result = select_rank_counterfactual_blend(
        np.array([0.0, 1.0]),
        np.array([0.2, 0.8]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        sampled_operator_error=0.2,
        error_tolerance=0.01,
    )
    assert result["alpha"] == 0.0
    assert all(
        row["alpha"] == 0.0 or not row["admissible"] for row in result["trace"]
    )


def test_standard_hamming_baseline_falsifies_hierarchy_claim(reproduced: dict) -> None:
    aggregate = reproduced["aggregate_test"]
    assert aggregate["hamming_kernel"]["log_loss"] < aggregate[
        "repaired_hierarchy_blend"
    ]["log_loss"]
    assert aggregate["hamming_kernel"]["brier"] < aggregate[
        "repaired_hierarchy_blend"
    ]["brier"]


def test_rank_sweep_reports_failed_condition(reproduced: dict) -> None:
    rank_two = reproduced["rank_frontier"][0]
    assert rank_two["rank"] == 2
    assert not rank_two["two_rank_condition_observed"]
    assert not rank_two["two_rank_bound_observed_coverage"]


def test_runtime_audit_records_matched_repeated_measurements(reproduced: dict) -> None:
    rows = reproduced["runtime_audit"]
    assert [row["records"] for row in rows] == [256, 512, 1024, 2048]
    for row in rows:
        assert row["repeats"] == 7
        assert row["right_hand_sides"] == 4
        assert row["dense_apply_seconds"] > 0.0
        assert row["hierarchy_apply_seconds"] > 0.0
        assert row["full_relative_error"] > 0.0


def test_hierarchy_scope_is_transductive_and_label_safe(reproduced: dict) -> None:
    assert reproduced["hierarchy_is_transductive"]
    assert not reproduced["test_labels_used_in_hierarchy"]
    assert reproduced["global_dense_matrix_materialized"] is False


def test_required_quantitative_figures_exist() -> None:
    figures = Path(experiment.PACKAGE_ROOT) / "results" / "figures"
    expected = {
        "fig_method_audit.pdf",
        "fig_rank_audit.pdf",
        "fig_rate_audit.pdf",
        "fig_runtime_audit.pdf",
    }
    assert {path.name for path in figures.glob("*.pdf")} == expected
