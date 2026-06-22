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
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.skip_run:
        _run("scripts/run.py", "--config", args.config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")
    _sync_submission()


if __name__ == "__main__":
    main()
