# topoqec — code artifact

CPU reference implementation for **Topology-Aware Graph RL Decoding for Quantum Error
Correction under Structured Noise**. Codes are explicit stabilizer constructions (a
bit-flip repetition code and a surface-like toy lattice); decoders run on the code's
Tanner / defect graph with numpy, networkx and scikit-learn (an optional PyTorch graph
decoder can be dropped in). No heavyweight quantum dependency.

## Install
```bash
conda env create -f environment.yml && conda activate topoqec
pip install -e .          # install topoqec from pyproject.toml (core deps +
                          # the topoqec-reproduce entry point)
# pip install -e .[gpu]   # optional accelerated backend (torch)
```
The scripts also prepend `src/` to `sys.path` via a bootstrap, so `export
PYTHONPATH=src` works without an editable install if you prefer.

## Reproduce
```bash
make test        # tests: code identities, decoder ordering, automorphism, metric bounds
make demo        # smoke config (~1-3 s) -> results/summary.json
make tables      # results/main_results.tex
make figures     # figures/fig_threshold.pdf, figures/fig_risk_coverage.pdf
make audit       # readiness gate
make full-run    # reported-scale config (minutes)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```
> macOS: the Makefile/scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can both ship an OpenMP runtime. No effect on results.

## Layout
```
src/topoqec/
  codes.py        repetition code + surface-like lattice; syndromes, logical operators, Tanner graph
  noise.py        structured-noise samplers: iid / biased / burst / correlated / measurement-error
  decoders.py     lookup · majority · matching (MWPM) · BP · neural · topology · fused · RL selector
  metrics.py      logical error rate · expected calibration error (ECE) · risk-coverage curve
  symmetry.py     cyclic-relabeling automorphism invariance check (integrity flag)
  runner.py       Monte-Carlo over regimes x decoders -> results/summary.json
  config.py       smoke.yaml (demo) · full.yaml (reported)
  plotting.py     threshold (logical-vs-physical) + risk-coverage figures
  audit.py        forbidden-claims + traceable-number checks
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     repetition-code identities · decoder ordering · automorphism invariance · metric bounds
```

## What is computed
`results/summary.json` holds, per decoder: the logical error rate with 95% CIs and the
expected calibration error; a per-(noise regime, decoder) breakdown; threshold curves
(logical vs physical error rate) for every decoder; a risk-coverage curve for the fused
decoder; and the `automorphism_invariant` integrity flag. It is the single source of truth
for every table, figure and macro.

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
A from-basics explanation of the stabilizer, graph, and decoder objects is in `src/topoqec/foundations.py`.
