# topoqec — code artifact

CPU reference implementation for **Selective and Calibrated Decoding for Quantum Error
Correction: A Topology-Aware Decoder-Family Benchmark under Structured Noise**. Codes are
explicit stabilizer constructions (a bit-flip repetition code and a surface-like toy
lattice); decoders run on the code's Tanner / defect graph through a single
`decode(syndrome) → (correction, confidence)` interface, built on numpy, networkx, and
scikit-learn. An optional PyTorch graph decoder can be dropped in; the core artifact carries
a lightweight dependency set.

The benchmark reports logical error rate (with analytic 95% intervals), expected calibration
error, and the selective risk–coverage trade-off jointly across five structured-noise
regimes. At reported scale the eight decoders are statistically tied on raw logical error
rate (spread 0.0024, below one interval width ≈ 0.0032), while their calibration spans an
eight-fold range; the artifact reports exactly that and an audit gate rejects
absolute-superiority phrasing.

## Installation

From PyPI:

```bash
pip install topoqec
```

From this source tree (with the reproducibility pipeline):

```bash
conda env create -f environment.yml && conda activate topoqec
pip install -e .            # install topoqec from pyproject.toml (core deps)
# pip install -e .[gpu]     # optional accelerated backend (torch)
# pip install -e .[dev]     # pytest + build + twine
```

The scripts also prepend `src/` to `sys.path` via a bootstrap, so `export PYTHONPATH=src`
works without an editable install.

## Reproduce

The installed console entry point deterministically regenerates the tables and figures from a
config (master seed 0):

```bash
topoqec-reproduce --config configs/full.yaml      # reported-scale tables + figures
topoqec-reproduce --config configs/smoke.yaml     # laptop-scale demo
topoqec-reproduce --config configs/full.yaml --skip-run   # regenerate tables/figures from an existing summary.json
```

Equivalently, via the Makefile / scripts:

```bash
make test        # tests: code identities, decoder ordering, automorphism, metric bounds
make demo        # smoke config (~6 s) -> results/summary.json
make tables      # results/main_results.{tex,md}
make figures     # figures/fig_threshold.pdf, figures/fig_risk_coverage.pdf
make audit       # readiness gate: traceable numbers, forbidden phrasing rejected
make full-run    # reported-scale config (minutes)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```

> macOS: the Makefile and scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can both ship an OpenMP runtime. This setting leaves the results unchanged.

## Figures and tables regenerated

`topoqec-reproduce` (and `make tables figures`) regenerate the following from
`results/summary.json`:

| Artifact | Source script | Contents |
|---|---|---|
| `results/main_results.tex` | `scripts/make_tables.py` | LaTeX table: per-decoder logical error rate, ±95% CI, ECE |
| `results/main_results.md` | `scripts/make_tables.py` | Markdown mirror of the same table |
| `figures/fig_threshold.pdf` | `scripts/make_figures.py` | Logical-vs-physical error rate (threshold) curves for the core decoders |
| `figures/fig_risk_coverage.pdf` | `scripts/make_figures.py` | Risk–coverage curve from sweeping the abstention threshold |

The entry point also mirrors the regenerated tables and figures into the sibling
`submission/` tree when one is present.

## Determinism

Every run is seeded from the config's `seed` field (master seed 0 for both `smoke.yaml` and
`full.yaml`); `src/topoqec/seed.py` derives the per-component generators and records run
provenance into `results/summary.json`. The provenance block carries the platform, the
dependency versions, the wall-clock runtime, and the peak memory of the run, so a reviewer
reproduces identical numbers by re-running the pipeline at the same scale. To reduce
nondeterminism from threaded BLAS, the entry point and scripts set `OMP_NUM_THREADS=1` and
`KMP_DUPLICATE_LIB_OK=TRUE`. The audit gate (`scripts/audit_claims.py`) exits non-zero if any
reported number fails to trace to `summary.json` or if forbidden absolute-superiority phrasing
is detected.

## Layout

```
src/topoqec/
  codes.py        repetition code + surface-like lattice; syndromes, logical operators, Tanner graph
  noise.py        structured-noise samplers: iid / biased / burst / correlated / measurement-error
  decoders.py     lookup · majority · matching (MWPM) · BP · neural · topology · fused · RL selector
  metrics.py      logical error rate · expected calibration error (ECE) · risk-coverage curve
  symmetry.py     cyclic-relabeling automorphism invariance check (integrity flag)
  runner.py       Monte-Carlo over regimes x decoders -> results/summary.json
  reproduce.py    topoqec-reproduce console entry point (tables + figures from a config)
  config.py       config loading (smoke.yaml demo · full.yaml reported)
  seed.py         deterministic seeding + run provenance
  plotting.py     threshold (logical-vs-physical) + risk-coverage figures
  audit.py        forbidden-claims + traceable-number checks
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     repetition-code identities · decoder ordering · automorphism invariance · metric bounds
```

## What is computed

`results/summary.json` holds, per decoder: the logical error rate with 95% confidence
intervals and the expected calibration error; a per-(noise regime, decoder) breakdown;
threshold curves (logical vs physical error rate) for every core decoder; a risk–coverage
curve for the fused decoder; and the `automorphism_invariant` integrity flag. It is the
single source of truth for every table, figure, and macro.

## Pinned dependencies

Declared in `pyproject.toml` and `requirements.txt`:

| Package | Constraint | Reported-run version |
|---|---|---|
| numpy | `>=1.24` | 2.4.3 |
| scipy | `>=1.10` | 1.17.1 |
| networkx | `>=3.0` | 3.6.1 |
| scikit-learn | `>=1.2` | 1.8.0 |
| matplotlib | `>=3.6` | 3.10.8 |
| pyyaml | `>=6.0` | 6.0.3 |

Supported Python: 3.9–3.13 (the reported run used Python 3.13.12). The exact versions in the
right column are recorded in the `provenance` block of `results/summary.json`; pinning to them
reproduces the reported numbers byte-for-byte under master seed 0. Optional extras: `gpu`
(`torch>=2.0`) and `dev` (`pytest>=7.0`, `build>=1.0`, `twine>=5.0`).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
