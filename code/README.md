# `hqml-transfer-qec` reproduction package

This package reproduces the negative audit reported in `main.tex`. It parses
the authenticated public archive, preserves every shot multiplicity, performs
rolling lower-rate model selection, evaluates the untouched two-rate test,
runs full and sampled operator audits, and regenerates four quantitative PDF
figures.

## Install and verify

From the repository root:

```bash
python -m venv /tmp/transfer-qec-audit
/tmp/transfer-qec-audit/bin/python -m pip install ./code
/tmp/transfer-qec-audit/bin/transfer-qec-reproduce --verify
/tmp/transfer-qec-audit/bin/python -m pytest -q code/tests
/tmp/transfer-qec-audit/bin/python code/scripts/validate_release.py
```

Editable use is also supported:

```bash
python -m pip install -e ./code
transfer-qec-reproduce --verify
transfer-qec-figures --output /tmp/transfer-qec-figures
```

The replay takes roughly tens of seconds on a modern laptop CPU. Scientific
values are locked recursively; wall times and the platform string are recorded
but excluded from cross-host equality.

## What is tested

- the dataset hash, seven file contract, 75,598 rows, and 70,000,000 shots;
- exact equality of the staged Hamming transform and a small dense Kronecker
  matrix;
- class-balancing with shot multiplicities;
- normalization, conditional two-rank, and clipped-log-loss bounds;
- hard rejection of nonzero blending after a failed operator diagnostic;
- the rank-two contraction failure in the full dense audit;
- the empirical finding that the Hamming baseline beats the hierarchy;
- the matched seven-repeat timing crossover, without a platform-independent
  speed claim;
- locked-result replay and presence of exactly four figures.

## Scope

The binary table does not provide the detector graph, circuit, recovery target,
or logical-observable history required to run MWPM, belief matching, or an
end-to-end decoder fairly. The package therefore compares only methods that can
be evaluated on the released columns. It makes no threshold, hardware-latency,
or inductive-generalization claim.

The source archive remains governed by its upstream CC BY 4.0 terms; the code is
MIT licensed.
