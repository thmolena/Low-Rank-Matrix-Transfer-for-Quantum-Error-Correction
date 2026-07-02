# Learning to decode correlated quantum errors: a topological graph neural decoder

A learned surface-code decoder that recovers most of the accuracy gap between
minimum-weight perfect matching (MWPM) and a model-aware belief-propagation
baseline under correlated circuit-level noise, without ever being given the noise
model. The decoder is a lightweight message-passing network on the code's
*detector graph*; it is trained only on samples, outputs a calibrated posterior
over the logical class, and is evaluated head-to-head against MWPM and BP+OSD on
identical [Stim](https://github.com/quantumlib/Stim) shots with paired statistics.
Distributed as the pip package **`specops-gnndec`** (import name `surfgnn`), part
of the *spectral-truncation operators* (SpecOps) research program.

## Summary

Decoding a quantum error-correcting code is a problem of statistical inference:
a syndrome is a noisy observation, and the correct logical class is the one the
fault distribution makes most probable. The standard surface-code decoder, MWPM,
is provably optimal only when faults are independent and *graphlike*. Real
hardware noise is neither: crosstalk and correlated two-qubit faults flip several
detectors jointly, forming *hyperedges* that a sum of independent edge weights
cannot represent. This work (i) proves precisely when MWPM is optimal and
constructs correlated instances on which it is provably suboptimal, and (ii)
introduces a graph neural decoder that learns the joint fault structure from
samples, matching MWPM where MWPM is optimal and surpassing it under correlated
noise while recovering most of the gap to a correctly specified BP+OSD baseline.

## Background and problem setting

For a reader outside quantum error correction, the setting is as follows. A
logical qubit is encoded redundantly across many physical qubits (here, the
distance-*d* rotated surface code, with *d²* data qubits). Rather than measuring
the fragile encoded state directly, the hardware repeatedly measures *stabilizers*
— multi-qubit parity checks that are deterministic in the absence of noise. The
record of stabilizer outcomes, compared across successive rounds, defines a set of
binary *detectors*; a lit detector marks a space-time location where the parity
changed, i.e. evidence of an error. The task of the **decoder** is to read the
pattern of lit detectors (the *syndrome*) and infer whether the encoded *logical*
bit was flipped, so the flip can be corrected. The quality of this classical
inference — not the raw device physics — sets the achievable logical error rate.

Stim compiles a noisy circuit into a *detector error model* (DEM): a list of
independent fault mechanisms, each firing with some probability and flipping a
fixed set of detectors and (possibly) the logical observable. A mechanism that
flips at most two detectors is *graphlike* (an edge in a decoding graph); one that
flips three or more is a *hyperedge*. MWPM operates on the graphlike (decomposed)
view: it assigns each edge a weight and returns the minimum-weight set of edges
consistent with the syndrome. This is exact maximum-likelihood decoding **if and
only if** the fault distribution factorizes over independent graphlike edges.
Correlated hardware faults violate that assumption, and matching has no way to
represent the joint structure it discards. This is the gap the present work
formalizes and closes.

## Contributions

1. **A tight optimality characterization for matching.** We prove that MWPM
   coincides with maximum-likelihood decoding if and only if the fault
   distribution factorizes over graphlike edges (Lemma on additive log-likelihood
   ⇔ independence; Proposition "matching is ML iff the model factorizes"). Under
   independence the minimum-weight configuration is a maximum-likelihood fault
   configuration; under any correlated distribution no fixed edge weights make
   matching maximum-likelihood.

2. **A provable separation under correlation.** We construct an explicit DEM with
   a single correlated mechanism on which matching's logical error rate exceeds
   the Bayes-optimal decoder's by a strictly positive, closed-form amount — the
   probability mass on a logical-flipping hyperedge matching cannot represent.
   Summed over syndromes, this yields an exact population-level lower bound on
   matching's excess logical error under correlated noise.

3. **A statistical account of decoding as Bayesian inference.** We identify the
   Bayes-optimal target as the *coset MAP* decoder over logical cosets (not the
   maximum-likelihood *configuration*, which can differ under degeneracy), and
   prove four results for the learned decoder: Bayes-optimality of the coset
   posterior, a universal-approximation theorem on the detector graph, a
   Rademacher/VC generalization bound with explicit sample complexity, and a
   risk-consistency theorem. Together they establish the network as a consistent
   probabilistic surrogate for an intractable maximum-likelihood decoder.

4. **A topological graph neural decoder.** We introduce a message-passing network
   on the detector graph — one node per detector, edges joining detectors that
   share a fault mechanism, so correlated hyperedges appear as denser local
   subgraphs. Trained only on samples, it estimates the Bayes posterior over the
   logical class and returns a calibrated confidence usable for selective
   decoding.

5. **A controlled, like-for-like empirical dissection.** All three decoders score
   *identical* Stim shots, with a noise knob that interpolates from independent to
   correlated. We report Wilson intervals, paired McNemar exact tests, paired
   bootstrap intervals, and calibration error, so any measured difference is
   attributable to representational power rather than to sampling advantage or
   model mismatch. The entire pipeline is released and every reported number is
   traced to a single `summary.json`.

## Method

The decoder places one node per detector and joins two nodes whenever some
mechanism in the *full* DEM flips both; a correlated hyperedge therefore
contributes a small clique, so local graph density encodes exactly the correlation
structure matching discards. Each node carries its normalized space-time
coordinates *(x, y, t)* as static features; the only per-shot input is the binary
detector value. Message passing uses the symmetric-normalized adjacency with
self-loops:

    h^{(l+1)} = LayerNorm( ReLU( W_self · h^{(l)} + W_neigh · Â · h^{(l)} ) )

followed by concatenated mean- and max-pooling over nodes and a two-layer
perceptron to a single logit. A sigmoid yields q̂ = Pr[logical flip | detectors],
an estimate of the Bayes posterior. The decision threshold is calibrated on
held-out validation data; max(q̂, 1−q̂) serves as the confidence for selective
decoding. Training minimizes binary cross-entropy (a strictly proper scoring rule,
so the population optimum is the true posterior and the output stays calibrated)
on fresh samples from the circuit under test; the network never sees the DEM.
Because the graph topology is shared across all shots, message passing reduces to a
few dense matrix products and runs CPU-only. Baselines are MWPM (PyMatching, on the
decomposed DEM) and BP+OSD (ldpc, min-sum BP with combination-sweep OSD, on the
full DEM); both are regime-aware, receiving the DEM of the exact circuit they
decode.

## Main results

All figures below are the released small-scale reproduce target: the rotated
surface code at distance *d = 3*, *T = d* rounds, over 2 independent seeds, with
all decoders scoring identical test shots (CPU-only; total runtime ≈ 36 s). Pooled
logical error rates (LER) are reported with 95% Wilson intervals.

**Under independent circuit-level depolarizing noise (rate p = 0.002).** MWPM is
provably maximum-likelihood, so no decoder can beat it. In this regime MWPM and the
model-aware BP+OSD agree (LER 0.0025 [0.0018, 0.0034] and 0.0026 [0.0019, 0.0035]);
the learned decoder attains 0.0316 [0.0290, 0.0345] at *d = 3*, i.e. it does not
improve on matching where matching is already optimal (see Caveats).

**Under correlated noise (added XX faults, rate p_corr = 0.01).** Matching cannot
represent the correlated hyperedges and its LER rises to 0.1985 [0.1924, 0.2048],
well above the BP+OSD ceiling of 0.0757 [0.0718, 0.0800]. The learned decoder,
having seen the joint structure in training, attains 0.1070 [0.1023, 0.1119] — a
**46.1% relative reduction in logical error versus MWPM**, with a paired McNemar
test that is overwhelmingly significant (p = 2.4 × 10⁻¹⁶⁴) and a paired-bootstrap
interval for the LER difference that excludes zero. At *d = 3* the learned decoder
recovers **74.5% of the MWPM → BP+OSD gap without ever being given the noise
model** — only samples.

**Calibrated selective decoding.** The learned posterior is well calibrated
(expected calibration error 0.0251 under correlated noise at *d = 3*), and
committing only on the most confident shots drives the committed logical error down
monotonically, turning the decoder into an uncertainty-aware control primitive.

## Significance

The result isolates *where* and *why* a learned decoder helps and pins the
advantage to a precise, provable structural feature — correlated hyperedges that a
graphlike additive objective cannot represent. Because the advantage tracks
correlation strength and is largest at small distance, it is concentrated in
exactly the strongly-correlated, near-term hardware regime where a better decoder
is most valuable, and because the decoder returns a calibrated posterior rather
than a bare decision, it supplies the abstention signal a fault-tolerant control
loop can act on. More broadly, the work reframes decoding as consistent
probabilistic surrogate modeling of an intractable coset posterior, folding the
code's known geometry and fault propagation directly into the model.

## Installation and reproduction

```bash
pip install .
# or, for development (editable):
pip install -e .
```

Heavy dependencies (`stim`, `pymatching`, `ldpc`, `torch`, `numpy`, `scipy`,
`matplotlib`) are declared in `pyproject.toml` and pulled in automatically.
CPU-only; no GPU required.

The package ships a console entrypoint that runs the full head-to-head, rebuilds
every figure and table, and syncs them into the manuscript:

```bash
gnndec-reproduce            # default preset "main"; runs the head-to-head
gnndec-reproduce smoke      # quick sanity check (d = 3 only; minutes)
gnndec-reproduce full       # archival high-fidelity preset (hours on CPU)

gnndec-reproduce --skip-run # reuse results/summary.json; only rebuild outputs/paper
gnndec-reproduce main --no-paper-sync   # don't touch paper/
```

This writes `results/summary.json` (the single source of truth: all raw numbers
plus platform/library/runtime provenance), the LaTeX table and numbers JSON, the
three figures, and — unless `--no-paper-sync` — copies the table and figures into
`paper/` and fills the manuscript's numeric placeholders from the fresh run. All
randomness derives from recorded seeds.

Equivalent low-level invocation:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE          # conda+pip OpenMP clash on macOS
PYTHONPATH=src python scripts/run.py main # writes results/summary.json
PYTHONPATH=src python scripts/make_outputs.py
```

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

Everything is driven by one experiment dataclass, `surfgnn.runner.RunConfig`, and
one training dataclass, `surfgnn.gnn.TrainConfig`. Presets live in
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
