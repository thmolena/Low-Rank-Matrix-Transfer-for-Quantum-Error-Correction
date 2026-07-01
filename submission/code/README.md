# specops-transfer-decoder

Reproducibility package for **"Transfer-Operator Decoding of Structured Noise in
Quantum Error Correction"** (Molena Huynh). Part of the single-author
*spectral-truncation-operators* program.

It implements a channel-adaptive **spectral decoder** that models the error
channel as a truncated transfer operator (an inhomogeneous first-order Markov /
classical matrix-product model on the code's defect graph), estimated from a
small calibration sample, and selects the maximum-a-posteriori coset with a
calibrated posterior confidence.

## Install

```bash
pip install .
# or, editable for development:
pip install -e .
```

Requires Python >= 3.9. Dependencies are pinned in `pyproject.toml` /
`requirements.txt`.

## Reproduce every figure and table

```bash
transfer-decoder-reproduce            # runs the full pipeline -> figures + tables + summary.json
# equivalently:
python scripts/run.py                 # regenerate results/summary.json
python scripts/make_figures.py        # regenerate figures from results
```

The manuscript's numbers are wired to `summary.json` via `\sync{}` macros, so a
fresh run updates the paper automatically.

## Extend / tweak (use it in your own project)

Every knob is exposed so you can adapt the decoder to your own codes and noise:

- **Code & distance** — repetition/surface code, code distance `d`.
- **Noise regimes** — i.i.d., biased, burst, neighbour-correlated, read-out;
  add your own by supplying a channel sampler.
- **Transfer-operator model** — Markov order / truncation rank, calibration
  sample size.
- **Evaluation** — trials per regime, physical error grid, seeds.
- **Add inputs** — point the estimator at your own calibration syndromes to fit
  the transfer operator to real hardware data.

See the config block and CLI flags in `scripts/run.py` for the full list, and
import the decoder directly:

```python
from specops_transfer_decoder import SpectralDecoder   # channel-adaptive MAP decoder
```

## Cite this work

If you use this package, please cite the paper:

```bibtex
@article{huynh2026topoqec,
  title  = {Transfer-Operator Decoding of Structured Noise in Quantum Error Correction},
  author = {Huynh, Molena},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

License: MIT.
