"""From-scratch foundations for the selective QEC decoding artifact."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Repetition-code computation",
        "A bit-flip repetition code stores one logical bit redundantly.  An "
        "error vector over GF(2) produces a syndrome by XORing adjacent data "
        "bits.  codes.py implements syndrome, logical value and logical-error "
        "checks, making decoder success an exact finite computation.",
    ),
    (
        "Structured noise",
        "noise.py samples iid, biased, burst, correlated and measurement-error "
        "regimes.  The benchmark asks whether decoder confidence and abstention "
        "remain useful when the noise distribution has spatial or readout "
        "structure rather than independent flips.",
    ),
    (
        "Unified decoder interface",
        "Every decoder returns (correction, confidence).  decoders.py includes "
        "lookup, majority, matching, belief propagation, neural coset selection, "
        "topology-calibrated confidence, fused decoding and a contextual-bandit "
        "selector.  The common interface makes all metrics comparable.",
    ),
    (
        "Logical error statistics",
        "metrics.py treats each shot as a Bernoulli failure indicator.  The "
        "reported logical error is a sample mean with analytic 95 percent "
        "confidence intervals, so decoders whose intervals overlap are reported "
        "as tied rather than ranked by noise.",
    ),
    (
        "Selective decoding",
        "Risk-coverage curves sweep a confidence threshold.  Coverage is the "
        "fraction of shots committed; risk is the logical error on committed "
        "shots.  Calibration determines whether abstaining on low-confidence "
        "shots actually reduces committed risk.",
    ),
    (
        "Calibration",
        "Expected calibration error bins predictions by confidence and compares "
        "mean confidence to empirical accuracy.  This separates accurate "
        "decoding from honest confidence, which is central to the manuscript's "
        "claim.",
    ),
    (
        "Integrity checks",
        "symmetry.py checks cyclic automorphism invariance and audit.py verifies "
        "that manuscript numbers trace back to summary.json.  The scripts treat "
        "summary.json as the single source of truth for figures and tables.",
    ),
    (
        "Reproduction path",
        "Run python scripts/run.py --config configs/full.yaml --out results, "
        "then scripts/make_tables.py, scripts/make_figures.py and "
        "scripts/audit_claims.py.  The topoqec-reproduce entry point performs "
        "the same steps and syncs the submission artifacts.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
