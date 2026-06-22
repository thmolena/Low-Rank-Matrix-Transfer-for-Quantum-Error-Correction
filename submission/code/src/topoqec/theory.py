"""From-principles guide for the selective, calibrated QEC decoder benchmark.

Quantum error correction
------------------------
A repetition code stores one logical bit redundantly. A physical bit-flip error
is a binary vector e. The measured syndrome records adjacent parity changes:

    s_i = e_i xor e_{i+1}.

A decoder receives the syndrome and returns a correction plus a confidence. It
succeeds when the residual error has trivial logical parity. codes.py implements
the repetition code, a surface-like toy lattice and the logical-failure checks.

Structured noise
----------------
noise.py implements seeded iid, biased, burst, correlated and measurement-error
models. These regimes test whether decoders work beyond the oversimplified iid
case. The benchmark reports per-regime results so noise structure is not hidden
inside a single average.

Decoder family
--------------
decoders.py puts lookup, majority, matching, belief propagation, neural coset
selection, topology recalibration, confidence fusion and a contextual-bandit
selector behind one interface:

    decode(syndrome) -> (correction, confidence).

This common interface makes logical error, calibration and abstention comparable
across the whole family.

Machine learning
----------------
The neural decoder is a logistic-regression coset selector. The RL decoder is a
contextual bandit that chooses which candidate decoder to trust in each syndrome
context. Both are intentionally small, reproducible estimators rather than opaque
deep models.

Metrics
-------
metrics.py computes logical error rate with Bernoulli 95 percent confidence
intervals, expected calibration error (ECE), and risk-coverage curves. Selective
decoding commits only when confidence exceeds a threshold. Calibration matters
because a confidence threshold is useful only when stated confidence matches
empirical correctness.

Integrity and reproduction
--------------------------
runner.py writes results/summary.json as the single source of truth. symmetry.py
checks automorphism invariance and audit.py verifies that claims trace to the
generated artifact. From code/:

    export PYTHONPATH=src
    make test
    bash scripts/reproduce_all.sh full
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
