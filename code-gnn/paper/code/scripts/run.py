"""Run the full head-to-head and write results/summary.json.

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python scripts/run.py [full|smoke]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from surfgnn.runner import RunConfig, run  # noqa: E402

PRESETS = {
    # Highest-fidelity preset used to produce the archival headline numbers.
    # CPU-only this can take many hours (dense per-shot message passing and the
    # per-shot BP+OSD loop both grow steeply with distance); prefer "main".
    "full": RunConfig(distances=(3, 5, 7), p=0.004, p_corr=0.010,
                      seeds=(0, 1, 2), n_train=70000, n_val=10000, n_test=12000,
                      gnn_epochs=60, gnn_hidden=96, gnn_layers=6),
    # Standard reproduce target: same three distances and both noise regimes as
    # "full", with reduced shot counts / epochs so it finishes CPU-only in a
    # couple of hours while keeping all confidence intervals meaningful.
    "main": RunConfig(distances=(3, 5, 7), p=0.004, p_corr=0.010,
                      seeds=(0, 1), n_train=20000, n_val=4000, n_test=6000,
                      gnn_epochs=40, gnn_hidden=96, gnn_layers=6),
    "smoke": RunConfig(distances=(3,), p=0.002, p_corr=0.01,
                       seeds=(0, 1), n_train=20000, n_val=4000, n_test=8000,
                       gnn_epochs=25, gnn_hidden=32, gnn_layers=3),
}

if __name__ == "__main__":
    preset = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    cfg = PRESETS[preset]
    out = os.path.join(os.path.dirname(__file__), "..", "results", "summary.json")
    run(cfg, os.path.abspath(out))
