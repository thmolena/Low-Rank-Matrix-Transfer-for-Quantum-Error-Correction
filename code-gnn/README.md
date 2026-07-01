# Topological Graph Neural Decoder for Correlated Quantum Errors

A learned surface-code decoder that **matches minimum-weight perfect matching
(MWPM) under independent noise and beats it under correlated circuit-level
noise**, evaluated head-to-head on real [Stim](https://github.com/quantumlib/Stim)
circuits against MWPM ([PyMatching](https://github.com/oscarhiggott/PyMatching))
and BP+OSD ([ldpc](https://github.com/quantumgizmos/ldpc)) with paired statistics.

Distributed as the pip package **`specops-gnndec`** (import name `surfgnn`),
part of the *spectral-truncation operators* (SpecOps) research program.

## Why this works
MWPM scores a correction by summing independent per-edge weights of a *graphlike*
error model. Correlated faults (crosstalk, correlated two-qubit gate errors) flip
several detectors *jointly* — a hyperedge that a sum of independent edge weights
cannot represent. We prove MWPM is maximum-likelihood **iff** the fault
distribution factorizes over graphlike edges (`paper/theory_notes.tex`), and a
graph neural network on the *detector graph* learns the joint structure from
samples alone.

## Install

```bash
pip install .
# or, for development (editable):
pip install -e .
```

Heavy dependencies (`stim`, `pymatching`, `ldpc`, `torch`, `numpy`, `scipy`,
`matplotlib`) are declared in `pyproject.toml` and pulled in automatically.
CPU-only; no GPU required.

## Reproduce everything (figures, tables, paper numbers)

The package ships a console entrypoint that runs the full head-to-head, rebuilds
every figure and table, and syncs them into the manuscript:

```bash
gnndec-reproduce            # default preset: "main" (d = 3,5,7; ~a couple of hours CPU)
gnndec-reproduce smoke      # quick sanity check (d = 3 only; minutes)
gnndec-reproduce full       # archival high-fidelity preset (hours on CPU)

gnndec-reproduce --skip-run # reuse results/summary.json; only rebuild outputs/paper
gnndec-reproduce main --no-paper-sync   # don't touch paper/
```

This writes:
- `results/summary.json` — the single source of truth (all raw numbers + provenance),
- `results/main_table.tex`, `results/numbers.json`,
- `figures/fig_ler_vs_distance.pdf`, `fig_gain_vs_distance.pdf`, `fig_risk_coverage.pdf`,
- and copies the table + figures into `paper/` and fills the manuscript's
  `\PH{...}` numeric placeholders + abstract headline number from the fresh run.

Equivalent low-level invocation:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE          # conda+pip OpenMP clash on macOS
PYTHONPATH=src python scripts/run.py main # writes results/summary.json
PYTHONPATH=src python scripts/make_outputs.py
```

All randomness derives from recorded seeds; provenance (platform, library
versions, runtime) is written into `summary.json`.

## Layout
- `src/surfgnn/circuits.py` — rotated surface-code Z-memory in Stim; correlated
  XX injection that propagates to detectors **and** the observable.
- `src/surfgnn/baselines.py` — MWPM (decomposed DEM) and BP+OSD (full DEM).
- `src/surfgnn/gnn.py` — message-passing decoder on the detector graph.
- `src/surfgnn/stats.py` — Wilson interval, paired McNemar, paired bootstrap, ECE.
- `src/surfgnn/runner.py` — multi-seed head-to-head → `results/summary.json`.
- `src/surfgnn/outputs.py` — tables + numbers + figures from `summary.json`.
- `src/surfgnn/reproduce.py` — `gnndec-reproduce` entrypoint (run + build + paper sync).
- `scripts/run.py`, `scripts/make_outputs.py` — thin CLI wrappers.
- `paper/` — manuscript (staging) and theory notes.

## Extend / tweak

Everything is driven by one dataclass, `surfgnn.runner.RunConfig`, and one
training dataclass, `surfgnn.gnn.TrainConfig`. Presets live in
`surfgnn.reproduce.PRESETS` (and mirror `scripts/run.py`).

### Experiment knobs — `RunConfig` (`src/surfgnn/runner.py`)
| field        | meaning                                               | default        |
|--------------|-------------------------------------------------------|----------------|
| `distances`  | code distances to sweep (tuple)                       | `(3, 5, 7)`    |
| `p`          | circuit-level depolarizing rate (independent regime)  | `0.002`        |
| `p_corr`     | correlated XX fault rate on neighbour pairs           | `0.01`         |
| `seeds`      | independent seeds (train/sample variance → CIs)       | `(0,1,2,3,4)`  |
| `n_train`    | training shots per seed                               | `40000`        |
| `n_val`      | validation shots per seed (threshold selection)       | `8000`         |
| `n_test`     | held-out test shots per seed (scored, paired)         | `20000`        |
| `gnn_epochs` | max training epochs (early stopping on val LER)       | `60`           |
| `gnn_hidden` | GNN hidden width                                      | `64`           |
| `gnn_layers` | number of message-passing layers                      | `4`            |

Both noise regimes (`independent`, `correlated`) are always run per distance;
the correlated regime sets `p_corr` while independent forces it to `0`.

### Model knobs — `TrainConfig` (`src/surfgnn/gnn.py`)
`epochs`, `batch`, `lr`, `hidden`, `layers`, `patience`, `weight_decay`. The
decision threshold is auto-calibrated on validation LER each epoch.

### Run a custom experiment from Python
```python
from surfgnn.runner import RunConfig, run
cfg = RunConfig(distances=(3, 5), p=0.003, p_corr=0.02,
                seeds=(0, 1, 2), n_train=30000, gnn_layers=8)
run(cfg, "results/summary.json")

from surfgnn import outputs
s = outputs.load("results")
outputs.build_all(s, "results", "figures")   # tables + numbers + figures
```

### Add a new preset
Edit `PRESETS` in `src/surfgnn/reproduce.py` (and `scripts/run.py`), then:
```bash
gnndec-reproduce mypreset
```

### Add a new decoder / baseline
Implement a class with a `predict(detectors) -> observables` method (see
`surfgnn.baselines.MWPMDecoder`), add it to the `decoders` list and the
prediction dict in `surfgnn.runner.run_cell`; it will automatically flow into
the pooled LERs, Wilson intervals and paired tests.

### Add a new correlated-noise model
`surfgnn.circuits._inject_correlated` inserts the correlated faults; swap in your
own fault layer there (or change `neighbour_pairs`) to study different
correlation structures. The DEM, MWPM/BP+OSD/GNN and statistics adapt
automatically.

### Plug into your own project
```python
import stim
from surfgnn.circuits import build_benchmark, sample
from surfgnn.gnn import GNNDecoder, TrainConfig

bm = build_benchmark(d=5, p=0.004, p_corr=0.01)
dem = bm.dem(decompose=False)
dtr, otr = sample(bm, 20000, seed=1)
dva, ova = sample(bm, 4000, seed=2)

gnn = GNNDecoder(dem, TrainConfig(epochs=40, hidden=96, layers=6), seed=0)
gnn.fit(dtr, otr, dva, ova)

dte, _ = sample(bm, 8000, seed=3)
preds = gnn.predict(dte)          # 0/1 logical-flip predictions
proba = gnn.predict_proba(dte)    # calibrated confidences for selective decoding
```
Any Stim `DetectorErrorModel` works: pass your own DEM to `GNNDecoder`, or feed
your own detector/observable arrays to `.fit()` / `.predict()`.

## Cite this work

If you use this package or its results, please cite:

```bibtex
@article{huynh2026gnndec,
  title   = {Learning to decode correlated quantum errors: a topological graph
             neural decoder that surpasses minimum-weight matching under
             structured circuit-level noise},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (SpecOps) program},
}
```

See `CITATION.cff` for machine-readable metadata.
