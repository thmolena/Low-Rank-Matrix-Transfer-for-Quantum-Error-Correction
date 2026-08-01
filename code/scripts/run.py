#!/usr/bin/env python3
"""Backward-compatible alias for the registered source-tree replay."""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401
from hqml_transfer_qec.experiment import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--verify")
    main()
