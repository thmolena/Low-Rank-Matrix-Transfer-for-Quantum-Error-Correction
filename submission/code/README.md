# specops-transfer-decoder

Reproducibility package for *"Transfer-Operator Decoding of Structured Noise in
Quantum Error Correction"* (Molena Huynh, North Carolina State University).

This artifact implements a channel-adaptive **spectral decoder** for quantum
error correction: it represents the physical error channel by a truncated
transfer operator (an inhomogeneous first-order Markov, i.e. classical
matrix-product, probabilistic surrogate on the code's defect graph) estimated
from a small calibration sample, and selects the maximum-a-posteriori (MAP)
logical coset with a posterior confidence that is calibrated by construction.
It is Bayes-optimal for the structured channel, provably reduces to the
minimum-weight coset leader under independent and identically distributed
(i.i.d.) noise, and is evaluated inside a single reproducible benchmark that
scores logical error, selective decoding (risk-coverage), and calibration on a
common footing against eight structure-agnostic baseline decoders.

## Background and problem setting

Quantum error-correcting (QEC) codes protect logical information by encoding it
redundantly across many physical qubits and repeatedly measuring stabilizer
operators. The stabilizer outcomes form a **syndrome**: a binary record of which
parity checks are violated. A **decoder** maps a syndrome to a correction. The
correction succeeds when the residual error (true error XOR correction) acts
trivially on the encoded information; otherwise the code suffers a **logical
error**.

Decoding is, formally, a problem of statistical inference. Because the syndrome
is a deterministic function of the error, Bayes' rule reduces to restricting the
error distribution to the set of errors consistent with the observed syndrome
(the **coset**) and choosing the most probable **logical class**. The optimal
rule is therefore the coset-MAP decoder for whatever error distribution the
device actually realises.

The difficulty is that the dominant decoders do not use the true distribution.
Minimum-weight perfect matching, belief propagation, and learned (neural /
reinforcement-learning) decoders all approximate the **minimum-weight coset
leader**. That leader is the maximum-likelihood recovery *only* for i.i.d.
(depolarizing or bit-flip) noise, where error probability decreases monotonically
in Hamming weight. Real hardware noise is not i.i.d.: it is **structured** -
dominated by one Pauli type (bias), spatially bursty, correlated between
neighbours, and accompanied by faulty syndrome read-out. Wherever structure
makes a heavier coset more probable than a lighter one, the minimum-weight leader
is provably sub-optimal. Two deployment-relevant properties are also usually
absent from decoder benchmarks: the ability to **abstain** on unreliable
syndromes (selective decoding), which is useful only if the gating confidence is
**calibrated**.

On a distance-`d` repetition code (the primary benchmark here), the syndrome bit
`s_i = e_i XOR e_{i+1}` fires where an error chain crosses link `i`, so defects
live on a one-dimensional line and each syndrome labels a two-element coset of
opposite logical class. This geometry makes eight otherwise dissimilar decoders
collapse to the *same* decision rule, which is why they tie on raw logical error.

## Contributions

1. **A channel-adaptive spectral decoder.** A coset-MAP decoder whose error model
   is a truncated transfer operator on the defect graph, estimated from a small
   calibration sample of the operating channel. It is the first decoder in the
   nine-member family to leave the accuracy tie, beating the i.i.d.-optimal
   leader with statistical significance under biased, bursty, and correlated
   noise (`z = 10.5, 16.6, 11.6`), and tying it honestly where the theory
   forbids a single-shot advantage.

2. **A theory of the structured-channel gap.** A proof that the coset leader is
   sub-optimal on any spatially non-uniform or correlated channel; that the
   transfer-operator (plug-in) decoder realises the channel-matched MAP optimum
   and is consistent in the calibration-sample size; and that its reported
   confidence is the exact model posterior, hence calibrated by construction. A
   spectral-gap analysis relates correlation decay to a threshold-type,
   distance-dependent error bound.

3. **A regime-resolved, honest evaluation.** All nine decoders share one
   `decode(syndrome) -> (correction, confidence)` interface across five
   structured-noise regimes, scored with analytic 95% Bernoulli confidence
   intervals and per-regime two-sample `z`-scores, with selective decoding
   (risk-coverage curves) and calibration (expected calibration error) as
   first-class metrics.

4. **An audited, installable artifact.** Every table and figure derives from one
   seeded source-of-truth file (`results/summary.json`); an audit gate rejects
   any reported number not traceable to that artifact, requires the central
   spectral-versus-leader claim to be statistically evidenced, and rejects
   unevidenced superlatives in continuous integration.

## Method

For a distance-`d` repetition code, each syndrome `s` labels the coset
`{c_0, c_1}` with `c_1 = c_0 XOR 1`, the two representatives carrying opposite
logical class. The spectral decoder models the error law by the transfer
operator

```text
P_theta(e) = pi_0(e_0) * prod_{i=0}^{d-2} T_i(e_i, e_{i+1}),
```

an initial law `pi_0` on {0,1} and per-site stochastic 2x2 transition matrices
`T_i`. Its log-likelihood is a sum of local terms read along the chain, so both
coset probabilities are obtained by a single `O(d)` forward sweep. The decoder
returns the coset representative of larger model log-likelihood and reports the
confidence `sigma(|log P_theta(c_0) - log P_theta(c_1)|)`, which is exactly the
model posterior of the chosen coset.

The operator is estimated by plug-in: Laplace-smoothed, row-normalised
transition counts from a labelled calibration sample of the operating channel
(the same data the neural baseline trains on). When the true channel factorises
as `P_theta`, maximising the model log-likelihood over the coset is identical to
the channel-matched MAP rule, so the decoder is Bayes-optimal; under i.i.d. noise
the operator becomes weight-monotone and the decoder reduces to the
minimum-weight leader, claiming no advantage where none exists. On the two-
dimensional surface-like code the operator degrades gracefully to a per-bit field
model. The eight baselines - `lookup` (optimal minimum-weight leader),
`majority`, `matching`, `bp`, `neural`, `topology`, `fused`, and an `rl`
contextual-bandit selector - are all structure-agnostic: each targets the i.i.d.
leader.

## Main results

Reported-scale configuration: repetition code, distance 9, 20,000 trials per
(regime, decoder), 100,000 pooled shots per decoder, 12,000 calibration/training
shots, evaluation physical error rate `p = 0.12`, master seed 0.

- **The tie, broken.** The eight structure-agnostic decoders span 0.1508-0.1542
  pooled logical error with 95% confidence interval half-width ~0.0022; their
  0.0033 best-to-worst spread is below 1.5 interval widths, so they are
  statistically indistinguishable. The spectral decoder reaches **0.1291**,
  lower than the best baseline by ~10 interval widths.

- **The win is structured-noise-specific.** Against the i.i.d.-optimal leader the
  spectral decoder cuts logical error by **44.0%** under bias (`z = 10.5`),
  **33.0%** under bursts (`z = 16.6`), and **21.8%** under neighbour
  correlations (`z = 11.6`). It ties the leader under i.i.d. noise (`z = 2.0`,
  below the audited `z > 3` threshold) and read-out noise (`z = 0.3`), where a
  corrupted syndrome carries no recoverable single-shot structure.

- **Calibration by construction.** The spectral decoder's expected calibration
  error is near zero where its model matches the channel (0.00069 i.i.d., 0.00159
  biased, 0.02538 burst) - orders of magnitude below the baselines - and large
  only under read-out noise (0.32359), the predicted model-misspecification
  failure. Across the agnostic family, calibration spans a factor of nearly six
  (0.0446-0.2624) and is uncorrelated with decoder sophistication.

- **Selective decoding.** Abstention yields a large, monotone reduction in
  committed logical error: `matching`'s committed error falls from 0.1525 at full
  coverage to 0.0108 at coverage 0.248 (a more than fourteenfold reduction).

## Significance

The result reframes how decoders should be evaluated and built:
channel-awareness, not algorithmic sophistication within the i.i.d.-optimal
class, is the lever that separates decoders under realistic noise - and it can be
made provable, tractable, and self-calibrating. A device whose noise can be
characterised obtains, from a small calibration sample, a decoder that is both
more accurate and trustworthy in its abstentions exactly where its accuracy gains
are real, with the boundary of that advantage made explicit by per-regime
significance tests.

**Scope.** The evidence is a statistically significant advantage under
*simulated* structured noise on toy codes (a one-dimensional repetition code and
a surface-*like* toy lattice, not a stabilizer-exact CSS surface code at scale).
The advantage requires a calibration sample of the operating channel; the
transfer operator is first-order (nearest-neighbour); results are at a single
distance and physical error rate.

## Installation and reproduction

Requires Python >= 3.9.

```bash
pip install specops-transfer-decoder     # from PyPI
# or, from this source tree:
pip install .                            # add -e for an editable install
```

Regenerate every figure, table, and reported number from fixed seeds:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE         # conda + pip OpenMP runtimes on macOS
transfer-decoder-reproduce               # full reproduce entry point -> figures + tables + summary.json
```

Equivalent lower-level entry points (run from `submission/code`):

```bash
python scripts/run.py                    # regenerate results/summary.json
python scripts/make_figures.py           # regenerate figures from results
python scripts/make_tables.py            # regenerate tables from results
python scripts/audit_claims.py           # audit gate: every number must trace to summary.json
bash   scripts/reproduce_all.sh full     # reported scale (d=9, 20,000 trials/regime)
bash   scripts/reproduce_all.sh          # fast smoke config for CI (d=5)
```

The manuscript's numbers are wired to `results/summary.json` via `\sync{}`
macros, so a fresh run updates the paper automatically. The reported run
completes end-to-end in ~277 s at ~206 MB peak memory on a single CPU.

## Extend and tweak

Every knob is exposed so the decoder can be adapted to other codes and noise
models. The configuration lives in `configs/` (`smoke.yaml`, `full.yaml`) and the
CLI flags of `scripts/run.py`; the tunable parameters are:

- **Code and distance** - repetition or surface-like code; code distance `d`.
- **Noise regimes** - `iid`, `biased` (bias factor), `burst` (burst length),
  `correlated` (recruitment probability), `measurement_error` (per-check
  read-out flip rate). Add your own by supplying a channel sampler in
  `src/topoqec/noise.py`.
- **Transfer-operator model** - Markov order / spectral-truncation rank, Laplace
  smoothing, and calibration-sample size.
- **Evaluation** - trials per regime, physical-error grid, abstention threshold,
  and the master seed.

To fit the decoder to your own data, point the estimator at a labelled
calibration sample of your operating channel (the per-site transition counts are
estimated directly from it), then import the decoder:

```python
from topoqec.decoders import SpectralDecoder   # channel-adaptive coset-MAP decoder
```

## Cite this work

```bibtex
@article{huynh2026topoqec,
  title   = {Transfer-Operator Decoding of Structured Noise in Quantum Error Correction},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {Manuscript in preparation},
  url     = {https://thmolena.github.io/Topological-Graph-for-Decoding-in-QEC/submission/main.pdf}
}
```

License: MIT.
