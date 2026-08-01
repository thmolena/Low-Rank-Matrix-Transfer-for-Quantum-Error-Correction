"""Generate the four quantitative figures for the QEC negative audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT if (PROJECT_ROOT / "results").is_dir() else SOURCE_ROOT
STAMP = datetime(2026, 7, 31, tzinfo=timezone.utc)


METHODS = (
    ("constant", "constant", "#777777"),
    ("logistic", "tuned logistic", "#4c78a8"),
    ("exact_lookup", "exact lookup", "#f58518"),
    ("hamming_kernel", "Hamming kernel", "#54a24b"),
    ("repaired_hierarchy_blend", "hierarchy blend", "#e45756"),
)


def _finish(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        path,
        bbox_inches="tight",
        metadata={"Creator": "hqml-transfer-qec", "CreationDate": STAMP, "ModDate": STAMP},
    )
    plt.close()


def write_figures(result: dict, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("*.pdf"):
        stale.unlink()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
        }
    )
    paths: list[Path] = []

    per_rate = result["per_rate_test"]
    rates = np.asarray([row["physical_fault_rate"] for row in per_rate])
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))
    markers = ("o", "s", "^", "D", "v")
    for (key, label, color), marker in zip(METHODS, markers):
        axes[0].plot(
            rates,
            [row[f"{key}_log_loss"] for row in per_rate],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1].plot(
            rates,
            [row[f"{key}_brier"] for row in per_rate],
            marker=marker,
            color=color,
            label=label,
        )
    for axis, ylabel in zip(axes, ("class-balanced log loss", "class-balanced Brier score")):
        axis.set_xlabel("held-out physical fault rate")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    path = output / "fig_rate_audit.pdf"
    _finish(path)
    paths.append(path)

    aggregate = result["aggregate_test"]
    labels = [label for _, label, _ in METHODS]
    colors = [color for _, _, color in METHODS]
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))
    for axis, metric, ylabel in (
        (axes[0], "log_loss", "two-rate log loss"),
        (axes[1], "brier", "two-rate Brier score"),
    ):
        values = [aggregate[key][metric] for key, _, _ in METHODS]
        bars = axis.bar(np.arange(len(values)), values, color=colors)
        axis.set_xticks(np.arange(len(values)), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.bar_label(bars, fmt="%.3f", fontsize=7)
        axis.grid(axis="y", alpha=0.2)
    path = output / "fig_method_audit.pdf"
    _finish(path)
    paths.append(path)

    frontier = result["rank_frontier"]
    ranks = np.asarray([row["rank"] for row in frontier])
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))
    axes[0].plot(
        ranks,
        [row["sampled_relative_error"] for row in frontier],
        "o-",
        label="sampled action",
    )
    axes[0].plot(
        ranks,
        [row["full_operator_frobenius_relative_error"] for row in frontier],
        "s-",
        label="full Frobenius",
    )
    axes[0].plot(
        ranks,
        [row["prediction_relative_error"] for row in frontier],
        "^-",
        label="normalized prediction",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("hierarchy rank")
    axes[0].set_ylabel("relative error")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(
        ranks,
        [row["storage_fraction"] for row in frontier],
        "o-",
        label="storage/dense",
    )
    axes[1].plot(
        ranks,
        [row["counted_work_fraction"] for row in frontier],
        "s-",
        label="counted work/dense",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("hierarchy rank")
    axes[1].set_ylabel("resource fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    path = output / "fig_rank_audit.pdf"
    _finish(path)
    paths.append(path)

    runtime = result["runtime_audit"]
    sizes = np.asarray([row["records"] for row in runtime])
    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.45))
    axes[0].plot(
        sizes,
        [1000 * row["hierarchy_apply_seconds"] for row in runtime],
        "o-",
        label="hierarchy apply",
    )
    axes[0].plot(
        sizes,
        [1000 * row["dense_apply_seconds"] for row in runtime],
        "s-",
        label="optimized dense apply",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("authenticated-data rows")
    axes[0].set_ylabel("median apply time (ms)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(
        sizes,
        [row["full_relative_error"] for row in runtime],
        "o-",
        label="full output error",
    )
    axes[1].plot(
        sizes,
        [row["hmat_storage_fraction"] for row in runtime],
        "s-",
        label="storage/dense",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("authenticated-data rows")
    axes[1].set_ylabel("error or fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    path = output / "fig_runtime_audit.pdf"
    _finish(path)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT / "results" / "figures",
    )
    args = parser.parse_args()
    result = json.loads((PACKAGE_ROOT / "results" / "results.json").read_text())
    for path in write_figures(result, args.output):
        print(path)


if __name__ == "__main__":
    main()
