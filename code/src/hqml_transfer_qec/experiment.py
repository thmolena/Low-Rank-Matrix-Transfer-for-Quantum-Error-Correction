"""Reproduce the negative audit of hierarchical QEC rate transfer."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import platform
from pathlib import Path
import shutil
import zipfile

import numpy as np

from .baselines import (
    BITS,
    TABLE_SIZE,
    binary_metrics,
    class_balanced_weights,
    hamming_prediction_table,
    tune_hamming_transfer,
)
from .bounds import clipped_log_loss_perturbation_bound
from .matrix_free import (
    dense_hierarchy_audit,
    fair_dense_runtime_audit,
    hierarchical_prior_bundle,
    select_rank_counterfactual_blend,
)


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "results").is_dir() else SOURCE_ROOT
DATA_SHA256 = "bdfce36a71f04295ac78fb372d9c2e381801c05e3be119f919750ef59026d072"
SEED = 1707
REGULARIZATION_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True)
class PublicData:
    features: np.ndarray
    patterns: np.ndarray
    labels: np.ndarray
    quantities: np.ndarray
    rate_indices: np.ndarray
    physical_rates: np.ndarray
    rate_values: np.ndarray
    source_names: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _syndrome_array(value: str) -> np.ndarray:
    array = np.asarray(ast.literal_eval(value), dtype=int)
    if array.shape != (4, 4) or not np.all((array == 0) | (array == 1)):
        raise ValueError("syndrome must be a binary 4-by-4 array")
    return array


def syndrome_features(value: str) -> np.ndarray:
    array = _syndrome_array(value).astype(float)
    flat = array.ravel()
    temporal = np.abs(np.diff(array, axis=0)).ravel()
    round_parity = np.mod(array.sum(axis=1), 2)
    spatial_parity = np.mod(array.sum(axis=0), 2)
    return np.concatenate(
        (flat, temporal, round_parity, spatial_parity, [flat.sum()])
    )


def syndrome_pattern(value: str) -> int:
    flat = _syndrome_array(value).reshape(-1)
    return int(sum(int(bit) << index for index, bit in enumerate(flat)))


def data_path() -> Path:
    path = PACKAGE_ROOT / "data" / "source" / "syndromes_dataset.zip"
    if not path.exists():
        raise FileNotFoundError(f"packaged QEC data missing: {path}")
    return path


def load_public_data() -> PublicData:
    path = data_path()
    if sha256(path) != DATA_SHA256:
        raise ValueError("QEC dataset SHA-256 mismatch")
    features: list[np.ndarray] = []
    patterns: list[int] = []
    labels: list[int] = []
    quantities: list[int] = []
    rate_indices: list[int] = []
    physical_rates: list[float] = []
    with zipfile.ZipFile(path) as archive:
        names = tuple(sorted(name for name in archive.namelist() if name.endswith(".csv")))
        if len(names) != 7:
            raise ValueError("expected seven physical-rate CSV files")
        for rate_index, name in enumerate(names):
            physical_rate = float(name.split("_pfr-", 1)[1].split("_", 1)[0])
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                if reader.fieldnames != ["labels", "syndromes", "quantity"]:
                    raise ValueError(f"unexpected columns in {name}")
                for row in reader:
                    label = int(row["labels"])
                    quantity = int(row["quantity"])
                    if label not in (0, 1) or quantity <= 0:
                        raise ValueError("invalid label or quantity")
                    features.append(syndrome_features(row["syndromes"]))
                    patterns.append(syndrome_pattern(row["syndromes"]))
                    labels.append(label)
                    quantities.append(quantity)
                    rate_indices.append(rate_index)
                    physical_rates.append(physical_rate)
    result = PublicData(
        features=np.vstack(features),
        patterns=np.asarray(patterns, dtype=int),
        labels=np.asarray(labels, dtype=int),
        quantities=np.asarray(quantities, dtype=float),
        rate_indices=np.asarray(rate_indices, dtype=int),
        physical_rates=np.asarray(physical_rates, dtype=float),
        rate_values=np.unique(np.asarray(physical_rates, dtype=float)),
        source_names=names,
    )
    if int(result.quantities.sum()) != 70_000_000:
        raise ValueError("unexpected total shot multiplicity")
    return result


def full_balanced_weights(data: PublicData, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    weights = np.zeros(len(data.labels), dtype=float)
    weights[selected] = class_balanced_weights(
        data.labels[selected], data.quantities[selected]
    )
    return weights


def logistic_fit(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> np.ndarray:
    beta = np.zeros(design.shape[1])
    eye = np.eye(design.shape[1])
    eye[0, 0] = 0.0
    for _ in range(60):
        score = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-score))
        gradient = (
            design.T @ (weights * (probability - labels))
            + regularization * (eye @ beta)
        )
        curvature = weights * probability * (1.0 - probability)
        hessian = (
            (design.T * curvature) @ design
            + regularization * eye
            + 1e-10 * np.eye(design.shape[1])
        )
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.linalg.norm(step) < 1e-10:
            break
    return beta


def logistic_predictions(
    data: PublicData,
    fit_mask: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(fit_mask, dtype=bool)
    mean = np.average(
        data.features[mask], axis=0, weights=data.quantities[mask]
    )
    variance = np.average(
        (data.features[mask] - mean) ** 2,
        axis=0,
        weights=data.quantities[mask],
    )
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    standardized = (data.features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    weights = class_balanced_weights(
        data.labels[mask], data.quantities[mask]
    )
    beta = logistic_fit(
        design[mask], data.labels[mask], weights, regularization
    )
    prediction = 1.0 / (
        1.0 + np.exp(-np.clip(design @ beta, -30.0, 30.0))
    )
    return prediction, standardized


def tune_logistic(data: PublicData) -> tuple[float, list[dict[str, float]]]:
    trace: list[dict[str, float]] = []
    best: tuple[float, float] | None = None
    for regularization in REGULARIZATION_GRID:
        losses = []
        for target_rate in (2, 3, 4):
            fit_mask = data.rate_indices < target_rate
            prediction, _ = logistic_predictions(
                data, fit_mask, regularization
            )
            target_mask = data.rate_indices == target_rate
            losses.append(
                binary_metrics(
                    data.labels[target_mask],
                    data.quantities[target_mask],
                    prediction[target_mask],
                )["log_loss"]
            )
        mean_loss = float(np.mean(losses))
        trace.append(
            {
                "regularization": float(regularization),
                "mean_development_log_loss": mean_loss,
            }
        )
        candidate = (mean_loss, float(regularization))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("empty logistic grid")
    return best[1], trace


def method_metrics(
    data: PublicData,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(mask, dtype=bool)
    return binary_metrics(
        data.labels[selected], data.quantities[selected], prediction[selected]
    )


def _stratified_audit_indices(data: PublicData, per_cell: int = 8) -> np.ndarray:
    indices: list[int] = []
    for rate in range(len(data.rate_values)):
        for label in (0, 1):
            available = np.flatnonzero(
                (data.rate_indices == rate) & (data.labels == label)
            )
            count = min(per_cell, len(available))
            positions = np.linspace(0, len(available) - 1, count).round().astype(int)
            indices.extend(available[positions].tolist())
    return np.asarray(sorted(set(indices)), dtype=int)


def _rank_frontier(
    data: PublicData,
    hierarchy_features: np.ndarray,
) -> tuple[list[dict[str, float | int | bool]], dict[str, float | bool]]:
    selected = _stratified_audit_indices(data)
    subset_rates = data.rate_indices[selected]
    known = subset_rates <= 4
    supervision = np.zeros(len(selected), dtype=float)
    supervision[known] = class_balanced_weights(
        data.labels[selected][known], data.quantities[selected][known]
    )
    rows: list[dict[str, float | int | bool]] = []
    bound_summary: dict[str, float | bool] = {}
    for rank in (2, 3, 4, 6, 8):
        audit = dense_hierarchy_audit(
            hierarchy_features[selected],
            supervision,
            data.labels[selected],
            SEED + rank,
            bandwidth=0.10,
            leaf_size=8,
            low_rank=max(1, rank // 2),
            high_rank=rank,
            maximum_records=128,
        )
        dense_prediction = np.asarray(audit.pop("dense_prediction"))[:, 0]
        audit.pop("low_prediction")
        high_prediction = np.asarray(audit.pop("high_prediction"))[:, 0]
        audit.pop("normalization_row_bounds")
        relative_prediction = float(
            np.linalg.norm(high_prediction - dense_prediction)
            / max(float(np.linalg.norm(dense_prediction)), 1e-15)
        )
        row = {
            "rank": rank,
            "sampled_relative_error": float(
                audit["hmat_sampled_row_relative_error"]
            ),
            "full_operator_frobenius_relative_error": float(
                audit["full_operator_frobenius_relative_error"]
            ),
            "full_operator_spectral_relative_error": float(
                audit["full_operator_spectral_relative_error"]
            ),
            "prediction_relative_error": relative_prediction,
            "storage_fraction": float(audit["hmat_storage_fraction"]),
            "counted_work_fraction": float(
                1.0 / audit["hmat_matmat_work_ratio"]
            ),
            "normalization_bound_observed_coverage": bool(
                audit["normalization_bound_observed_coverage"]
            ),
            "observed_two_rank_contraction_ratio": float(
                audit["observed_two_rank_contraction_ratio"]
            ),
            "two_rank_condition_observed": bool(
                audit["two_rank_condition_observed"]
            ),
            "two_rank_bound_observed_coverage": bool(
                audit["two_rank_bound_observed_coverage"]
            ),
        }
        rows.append(row)
        if rank == 8:
            audit_weights = class_balanced_weights(
                data.labels[selected], data.quantities[selected]
            )
            log_bound = clipped_log_loss_perturbation_bound(
                data.labels[selected],
                dense_prediction,
                high_prediction,
                audit_weights,
                epsilon=1e-8,
            )
            bound_summary = {
                "dense_audit_records": int(len(selected)),
                "normalization_bound_observed_coverage": bool(
                    audit["normalization_bound_observed_coverage"]
                ),
                "two_rank_condition_observed": bool(
                    audit["two_rank_condition_observed"]
                ),
                "two_rank_bound_observed_coverage": bool(
                    audit["two_rank_bound_observed_coverage"]
                ),
                "clipped_log_loss_actual_difference": log_bound.actual_difference,
                "clipped_log_loss_bound": log_bound.bound,
                "clipped_log_loss_bound_observed_coverage": log_bound.covered,
            }
    return rows, bound_summary


def _runtime_audit(hierarchy_features: np.ndarray) -> list[dict]:
    rows = []
    for size in (256, 512, 1024, 2048):
        selected = np.linspace(
            0, len(hierarchy_features) - 1, size
        ).round().astype(int)
        rows.append(
            fair_dense_runtime_audit(
                hierarchy_features[selected],
                SEED + size,
                bandwidth=0.10,
                leaf_size=32,
                rank=8,
                right_hand_sides=4,
                repeats=7,
            )
        )
    return rows


def reproduce() -> dict:
    data = load_public_data()
    fit_mask = data.rate_indices <= 3
    development_mask = data.rate_indices == 4
    train_mask = data.rate_indices <= 4
    test_mask = data.rate_indices >= 5

    regularization, logistic_trace = tune_logistic(data)
    development_logistic, _ = logistic_predictions(
        data, fit_mask, regularization
    )
    logistic, standardized = logistic_predictions(
        data, train_mask, regularization
    )

    hamming = tune_hamming_transfer(
        data.patterns,
        data.labels,
        data.quantities,
        data.rate_indices,
    )
    hamming_prediction = hamming.prediction_table[data.patterns]
    zero_trace = [row for row in hamming.trace if row["rho"] == 0.0]
    exact_window = min(
        zero_trace,
        key=lambda row: (row["mean_development_log_loss"], row["window"]),
    )["window"]
    exact_fit = (
        (data.rate_indices >= 5 - int(exact_window))
        & (data.rate_indices <= 4)
    )
    exact_table = hamming_prediction_table(
        data.patterns,
        data.labels,
        data.quantities,
        exact_fit,
        0.0,
    )
    exact_prediction = exact_table[data.patterns]

    hierarchy_features = np.column_stack(
        (standardized, data.rate_indices / max(float(data.rate_indices.max()), 1.0))
    )
    fit_supervision = full_balanced_weights(data, fit_mask)
    train_supervision = full_balanced_weights(data, train_mask)
    bundle = hierarchical_prior_bundle(
        hierarchy_features,
        [
            fit_supervision,
            train_supervision,
            fit_mask.astype(float),
            train_mask.astype(float),
        ],
        data.labels,
        SEED,
        bandwidth=0.10,
        leaf_size=64,
        low_rank=2,
        high_rank=8,
    )
    repaired_selection = select_rank_counterfactual_blend(
        data.labels[development_mask],
        development_logistic[development_mask],
        np.clip(bundle["low"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        np.clip(bundle["high"][0][development_mask, 0], 1e-8, 1.0 - 1e-8),
        weights=class_balanced_weights(
            data.labels[development_mask], data.quantities[development_mask]
        ),
        loss="log",
        sampled_operator_error=float(
            bundle["metrics"]["hmat_sampled_row_relative_error"]
        ),
        storage_fraction=float(bundle["metrics"]["hmat_storage_fraction"]),
    )
    repaired_alpha = float(repaired_selection["alpha"])
    repaired_prior = np.clip(bundle["high"][1][:, 0], 1e-8, 1.0 - 1e-8)
    repaired_blend = (
        (1.0 - repaired_alpha) * logistic + repaired_alpha * repaired_prior
    )

    unweighted_selection = select_rank_counterfactual_blend(
        data.labels[development_mask],
        development_logistic[development_mask],
        np.clip(bundle["low"][2][development_mask, 0], 1e-8, 1.0 - 1e-8),
        np.clip(bundle["high"][2][development_mask, 0], 1e-8, 1.0 - 1e-8),
        weights=class_balanced_weights(
            data.labels[development_mask], data.quantities[development_mask]
        ),
        loss="log",
        sampled_operator_error=float(
            bundle["metrics"]["hmat_sampled_row_relative_error"]
        ),
        storage_fraction=float(bundle["metrics"]["hmat_storage_fraction"]),
    )
    unweighted_alpha = float(unweighted_selection["alpha"])
    unweighted_blend = (
        (1.0 - unweighted_alpha) * logistic
        + unweighted_alpha
        * np.clip(bundle["high"][3][:, 0], 1e-8, 1.0 - 1e-8)
    )

    methods = {
        "constant": np.full(len(data.labels), 0.5),
        "logistic": logistic,
        "exact_lookup": exact_prediction,
        "hamming_kernel": hamming_prediction,
        "repaired_hierarchy_blend": repaired_blend,
    }
    aggregate = {
        name: method_metrics(data, prediction, test_mask)
        for name, prediction in methods.items()
    }
    per_rate = []
    for rate_index in (5, 6):
        mask = data.rate_indices == rate_index
        row: dict[str, float | int] = {
            "rate_index": rate_index,
            "physical_fault_rate": float(data.rate_values[rate_index]),
            "rows": int(mask.sum()),
            "shots": int(data.quantities[mask].sum()),
        }
        for name, prediction in methods.items():
            metrics = method_metrics(data, prediction, mask)
            for metric, value in metrics.items():
                row[f"{name}_{metric}"] = value
        per_rate.append(row)

    unweighted_metrics = method_metrics(data, unweighted_blend, test_mask)
    rank_frontier, bound_summary = _rank_frontier(data, hierarchy_features)
    runtime_audit = _runtime_audit(hierarchy_features)

    training_patterns = np.unique(data.patterns[train_mask])
    seen_test = np.isin(data.patterns[test_mask], training_patterns)
    test_quantities = data.quantities[test_mask]
    rate_summary = []
    for index, rate in enumerate(data.rate_values):
        mask = data.rate_indices == index
        positive = mask & (data.labels == 1)
        rate_summary.append(
            {
                "rate_index": index,
                "physical_fault_rate": float(rate),
                "rows": int(mask.sum()),
                "unique_syndromes": int(np.unique(data.patterns[mask]).size),
                "shots": int(data.quantities[mask].sum()),
                "positive_shots": int(data.quantities[positive].sum()),
                "positive_shot_fraction": float(
                    data.quantities[positive].sum() / data.quantities[mask].sum()
                ),
            }
        )

    return {
        "schema_version": 2,
        "package": "hqml_transfer_qec",
        "manuscript_key": "transfer_qec_negative_audit",
        "seed": SEED,
        "task": "rate-held-out distance-three syndrome classification audit",
        "dataset": "Zenodo distance-three simulated surface-code syndrome counts",
        "dataset_url": "https://doi.org/10.5281/zenodo.11166209",
        "dataset_sha256": DATA_SHA256,
        "dataset_origin": "Stim simulation; not quantum-hardware measurements",
        "source_files": list(data.source_names),
        "rows": int(len(data.labels)),
        "shots": int(data.quantities.sum()),
        "noise_rates": int(len(data.rate_values)),
        "rate_summary": rate_summary,
        "unique_syndromes": int(np.unique(data.patterns).size),
        "syndrome_bits": BITS,
        "feature_dimension": int(data.features.shape[1]),
        "train_shots": int(data.quantities[train_mask].sum()),
        "test_shots": int(data.quantities[test_mask].sum()),
        "test_rows": int(test_mask.sum()),
        "test_row_seen_syndrome_fraction": float(np.mean(seen_test)),
        "test_shot_seen_syndrome_fraction": float(
            np.sum(test_quantities[seen_test]) / np.sum(test_quantities)
        ),
        "metric": "class-balanced binary log loss, Brier score, and balanced accuracy",
        "logistic_regularization": regularization,
        "logistic_selection_trace": logistic_trace,
        "hamming_selected_window": int(hamming.window),
        "hamming_selected_rho": float(hamming.rho),
        "hamming_development_log_loss": float(
            hamming.mean_development_log_loss
        ),
        "hamming_selection_trace": hamming.trace,
        "hamming_prediction_table_entries": TABLE_SIZE,
        "hamming_prediction_table_bytes_float64": TABLE_SIZE * 8,
        "hamming_two_class_output_updates": 2 * BITS * TABLE_SIZE,
        "exact_lookup_selected_window": int(exact_window),
        "aggregate_test": aggregate,
        "per_rate_test": per_rate,
        "repaired_hierarchy_selected_alpha": repaired_alpha,
        "repaired_hierarchy_selection_trace": repaired_selection["trace"],
        "row_unweighted_selected_alpha": unweighted_alpha,
        "row_unweighted_test": unweighted_metrics,
        "hierarchy_is_transductive": True,
        "test_labels_used_in_hierarchy": False,
        "rank_frontier": rank_frontier,
        "runtime_audit": runtime_audit,
        "runtime_platform": platform.platform(),
        **bound_summary,
        **bundle["metrics"],
    }


def _rounded_value(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 12)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _rounded_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded_value(item) for item in value]
    return value


def rounded(result: dict) -> dict:
    return _rounded_value(result)


def write_outputs(result: dict) -> None:
    result = rounded(result)
    outputs = [PACKAGE_ROOT / "results"]
    if SOURCE_ROOT != PACKAGE_ROOT:
        outputs.append(SOURCE_ROOT / "results")
    canonical = json.dumps(result, indent=2, sort_keys=True) + "\n"
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(canonical)
    (PACKAGE_ROOT / "results" / "locked_results.json").write_text(canonical)

    aggregate = result["aggregate_test"]
    rank_two = next(row for row in result["rank_frontier"] if row["rank"] == 2)
    rank_four = next(row for row in result["rank_frontier"] if row["rank"] == 4)
    rank_eight = next(row for row in result["rank_frontier"] if row["rank"] == 8)
    runtime_largest = result["runtime_audit"][-1]
    commands = {
        "AuditRows": f'{result["rows"]:,}',
        "AuditShots": f'{result["shots"]:,}',
        "UniqueSyndromes": f'{result["unique_syndromes"]:,}',
        "TestShots": f'{result["test_shots"]:,}',
        "LogisticLoss": f'{aggregate["logistic"]["log_loss"]:.4f}',
        "HierarchyLoss": f'{aggregate["repaired_hierarchy_blend"]["log_loss"]:.4f}',
        "HammingLoss": f'{aggregate["hamming_kernel"]["log_loss"]:.4f}',
        "HammingBrier": f'{aggregate["hamming_kernel"]["brier"]:.4f}',
        "HammingAccuracy": f'{aggregate["hamming_kernel"]["balanced_accuracy"]:.4f}',
        "ConstantLoss": f'{aggregate["constant"]["log_loss"]:.4f}',
        "ConstantBrier": f'{aggregate["constant"]["brier"]:.4f}',
        "ConstantAccuracy": f'{aggregate["constant"]["balanced_accuracy"]:.4f}',
        "LogisticBrier": f'{aggregate["logistic"]["brier"]:.4f}',
        "LogisticAccuracy": f'{aggregate["logistic"]["balanced_accuracy"]:.4f}',
        "LookupLoss": f'{aggregate["exact_lookup"]["log_loss"]:.4f}',
        "LookupBrier": f'{aggregate["exact_lookup"]["brier"]:.4f}',
        "LookupAccuracy": f'{aggregate["exact_lookup"]["balanced_accuracy"]:.4f}',
        "HierarchyBrier": f'{aggregate["repaired_hierarchy_blend"]["brier"]:.4f}',
        "HierarchyAccuracy": f'{aggregate["repaired_hierarchy_blend"]["balanced_accuracy"]:.4f}',
        "LegacyUnweightedLoss": f'{result["row_unweighted_test"]["log_loss"]:.4f}',
        "HierarchyAlpha": f'{result["repaired_hierarchy_selected_alpha"]:.3f}',
        "HammingRho": f'{result["hamming_selected_rho"]:.3e}',
        "HammingWindow": str(result["hamming_selected_window"]),
        "RankEightSampled": f'{rank_eight["sampled_relative_error"]:.3e}',
        "RankEightFull": f'{rank_eight["full_operator_frobenius_relative_error"]:.3e}',
        "RankEightStorage": f'{100 * result["hmat_storage_fraction"]:.3f}',
        "RankEightWork": f'{100 / result["hmat_matmat_work_ratio"]:.3f}',
        "SeenShotFraction": f'{100 * result["test_shot_seen_syndrome_fraction"]:.2f}',
        "RankTwoFull": f'{rank_two["full_operator_frobenius_relative_error"]:.3e}',
        "RankTwoPrediction": f'{rank_two["prediction_relative_error"]:.3e}',
        "RankTwoStorage": f'{rank_two["storage_fraction"]:.3f}',
        "RankTwoWork": f'{rank_two["counted_work_fraction"]:.3f}',
        "RankFourFull": f'{rank_four["full_operator_frobenius_relative_error"]:.3e}',
        "RankFourPrediction": f'{rank_four["prediction_relative_error"]:.3e}',
        "RankFourStorage": f'{rank_four["storage_fraction"]:.3f}',
        "RankFourWork": f'{rank_four["counted_work_fraction"]:.3f}',
        "RankEightFull": f'{rank_eight["full_operator_frobenius_relative_error"]:.3e}',
        "RankEightPrediction": f'{rank_eight["prediction_relative_error"]:.3e}',
        "RankEightAuditStorage": f'{rank_eight["storage_fraction"]:.3f}',
        "RankEightAuditWork": f'{rank_eight["counted_work_fraction"]:.3f}',
        "LargestDenseMillis": f'{1000 * runtime_largest["dense_apply_seconds"]:.3f}',
        "LargestHierarchyMillis": f'{1000 * runtime_largest["hierarchy_apply_seconds"]:.3f}',
        "LargestRuntimeError": f'{runtime_largest["full_relative_error"]:.3e}',
        "DenseAuditRows": str(result["dense_audit_records"]),
    }
    lines = [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in commands.items()]
    for output in outputs:
        (output / "results.tex").write_text("\n".join(lines) + "\n")

    from .figures import write_figures

    write_figures(result, outputs[0] / "figures")
    for output in outputs[1:]:
        destination = output / "figures"
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.pdf"):
            stale.unlink()
        for figure in (outputs[0] / "figures").glob("*.pdf"):
            shutil.copy2(figure, destination / figure.name)


def verify(result: dict) -> None:
    path = PACKAGE_ROOT / "results" / "results.json"
    if not path.exists():
        raise SystemExit("No locked results.json; run with --write first")
    expected = json.loads(path.read_text())
    actual = rounded(result)

    def compare(value, old, path_name: str) -> None:
        key = path_name.rsplit(".", 1)[-1]
        if "seconds" in key or key.endswith("platform"):
            return
        if key in {"clipped_log_loss_actual_difference", "clipped_log_loss_bound"}:
            # The global 1/epsilon Lipschitz diagnostic magnifies harmless
            # BLAS-level perturbations in the low-rank prediction.  Its
            # coverage flag remains locked and the bound inequality is tested
            # directly; the platform-sensitive magnitudes are not byte locks.
            return
        if isinstance(value, dict):
            if not isinstance(old, dict) or value.keys() != old.keys():
                raise SystemExit(f"Result key mismatch at {path_name}")
            for child, item in value.items():
                compare(item, old[child], f"{path_name}.{child}")
            return
        if isinstance(value, list):
            if not isinstance(old, list) or len(value) != len(old):
                raise SystemExit(f"Result list mismatch at {path_name}")
            for index, item in enumerate(value):
                compare(item, old[index], f"{path_name}[{index}]")
            return
        if isinstance(value, float):
            # Reductions and factorizations can differ by a few ulps across
            # BLAS implementations without changing any reported conclusion.
            if not math.isclose(value, old, rel_tol=1e-8, abs_tol=5e-9):
                raise SystemExit(
                    f"Mismatch at {path_name}: expected {old}, got {value}"
                )
            return
        if value != old:
            raise SystemExit(
                f"Mismatch at {path_name}: expected {old}, got {value}"
            )

    compare(actual, expected, "results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = reproduce()
    if args.write:
        write_outputs(result)
    if args.verify:
        verify(result)
    print(json.dumps(rounded(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
