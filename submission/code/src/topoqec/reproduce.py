"""Installed reproduction command for the topology-aware QEC decoding artifact."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _code_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def _sync_submission() -> None:
    code = _code_dir()
    submission = code.parent
    (submission / "figures").mkdir(exist_ok=True)
    (submission / "tables").mkdir(exist_ok=True)
    for path in (code / "figures").glob("*"):
        if path.suffix.lower() in {".pdf", ".png"}:
            shutil.copy2(path, submission / "figures" / path.name)
    for name in ("main_results.tex",):
        src = code / "results" / name
        if src.exists():
            shutil.copy2(src, submission / "tables" / name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="experiment config (default: the reported-scale configs/full.yaml, ~5 min on one CPU)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the same pipeline at reduced scale (configs/smoke.yaml, "
        "distance 5, 1,500 trials/regime; ~30 s). Overwrites the package's "
        "results/ and figures/ with quick-scale outputs but does NOT sync "
        "them into the manuscript's submission/figures and submission/tables "
        "-- rerun without --quick to regenerate and sync the reported-scale "
        "artifacts.",
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)
    config = "configs/smoke.yaml" if args.quick else args.config
    if not args.skip_run:
        _run("scripts/run.py", "--config", config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")
    if not args.quick:
        _sync_submission()


if __name__ == "__main__":
    main()
