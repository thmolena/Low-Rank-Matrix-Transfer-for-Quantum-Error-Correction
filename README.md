# Fast Hamming Product-Kernel Transfer for Surface-Code Syndrome Probabilities

This repository evaluates an exact binary-cube product kernel on the complete
public distance-three surface-code archive at Zenodo DOI
`10.5281/zenodo.11166209`. The source contains 75,598 rows representing 70
million Stim-simulated shots; the highest two fault rates remain held out.

The kernel obtains confirmation log loss 0.2871, compared with 0.5358 for the
hierarchical blend, 0.6333 for logistic regression, and 0.6931 for a constant.
Exact lookup retains the best balanced accuracy, a metric tradeoff reported in
the manuscript.

```bash
python -m pip install ./code
transfer-qec-reproduce --verify
python -m pytest -q code/tests
python code/scripts/validate_release.py
```

The product kernel is applied by $b$ stages of pair updates on the binary cube,
using $O(b2^b)$ arithmetic and $O(2^b)$ storage. The source is simulation, not
hardware data.
