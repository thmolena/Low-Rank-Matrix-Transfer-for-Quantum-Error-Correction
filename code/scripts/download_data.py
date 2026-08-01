#!/usr/bin/env python3
"""Verify the redistributed Zenodo archive; no download is required."""
from __future__ import annotations

import _bootstrap  # noqa: F401
from hqml_transfer_qec.experiment import DATA_SHA256, data_path, sha256


if __name__ == "__main__":
    path = data_path()
    actual = sha256(path)
    if actual != DATA_SHA256:
        raise SystemExit(f"SHA-256 mismatch: {actual}")
    print(f"authenticated: {path}\nsha256: {actual}")
