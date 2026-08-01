#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"
python scripts/download_data.py
python scripts/reproduce.py --verify
python -m pytest -q tests
python scripts/validate_release.py
