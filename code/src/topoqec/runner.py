"""End-to-end decoding experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol:

  build code (repetition distance d, or surface-like)
  fit the neural decoder on syndromes sampled from the eval error rate
  for each noise regime R:
      for each decoder:
          Monte-Carlo `trials` errors under R at p_eval
          decode; record logical error, confidence, abstention
  sweep p over a grid (iid) to trace logical-vs-physical error curves
  build the risk-coverage curve from the fused decoder's confidences
  check the cyclic-relabeling automorphism invariance -> integrity flag

Aggregates: per-(regime, decoder) logical error rate + ECE, threshold curves,
risk-coverage curve, and a "headline" dict (best decoder, fused-vs-majority
improvement, fused ECE). Uses the package's standard RunProvenance / summarize /
JSON provenance structure so every reported number traces to an artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import metrics, noise
from .codes import make_code
from .config import Config
from .decoders import NeuralDecoder, RLDecoder, SpectralDecoder, build_decoders
from .seed import RunProvenance, set_seed
from .symmetry import is_automorphism_invariant


def _sample_dataset(code, regime, n, p, rng, q):
    """Return (syndromes, errors) for fitting / evaluating a decoder."""
    S, E = [], []
    for _ in range(n):
        err, synd_flips = noise.sample(regime, code.n_data, p, rng, q=q)
        s = code.syndrome(err)
        if synd_flips is not None and s.size == synd_flips.size:
            s = (s ^ synd_flips).astype(np.int8)
        S.append(s)
        E.append(err)
    return np.array(S, dtype=np.int8), np.array(E, dtype=np.int8)


def _evaluate(code, decoder, regime, trials, p, rng, q, abstain):
    """Monte-Carlo one decoder under one regime. Returns per-trial records."""
    fails, confs, committed, corrects = [], [], [], []
    for _ in range(trials):
        err, synd_flips = noise.sample(regime, code.n_data, p, rng, q=q)
        s = code.syndrome(err)
        if synd_flips is not None and s.size == synd_flips.size:
            s = (s ^ synd_flips).astype(np.int8)
        corr, conf = decoder.decode(s)
        fail = code.logical_error(err, corr)
        fails.append(fail)
        confs.append(float(conf))
        committed.append(1 if conf >= abstain else 0)
        corrects.append(1 - fail)
    return {
        "fails": fails, "confs": confs, "committed": committed, "corrects": corrects,
    }


def _train_rl(code, roster, rng, trials, p, q):
    """Warm up the contextual bandit on its own Monte-Carlo stream."""
    actions = {k: v for k, v in roster.items() if k in ("matching", "topology", "fused", "bp")}
    if not actions:
        return None
    rl = RLDecoder(code, actions, rng)
    for _ in range(trials):
        err, synd_flips = noise.sample("iid", code.n_data, p, rng, q=q)
        s = code.syndrome(err)
        if synd_flips is not None and s.size == synd_flips.size:
            s = (s ^ synd_flips).astype(np.int8)
        a = rl.select(s)
        corr, _ = actions[rl.actions[a]].decode(s)
        reward = 1.0 - code.logical_error(err, corr)
        rl.update(s, a, reward)
    return rl


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    rng = set_seed(cfg.seed)
    code = make_code(cfg.code, cfg.distance)

    # ---- fit the neural decoder on a mixed-regime training stream ---------- #
    # The neural decoder is a learned *coset selector*: it sees the syndrome and
    # predicts which logical coset the true correction lies in, relative to the
    # syndrome-determined base chain. Training pairs are (syndrome, true error);
    # the decoder derives the coset label internally. This is the regime where a
    # learned decoder beats fixed minimum-weight matching: under biased / bursty
    # noise the most-likely coset is not always the lightest one.
    # Per-regime calibration streams are retained so that the channel-adaptive
    # spectral decoder can be matched to each operating channel using exactly the
    # same labelled data the neural decoder is trained on (a fair comparison).
    train_S, train_E = [], []
    train_by_regime: Dict[str, np.ndarray] = {}
    for regime in cfg.regimes:
        s, e = _sample_dataset(code, regime, cfg.train_trials // len(cfg.regimes),
                               cfg.p_eval, rng, cfg.q_meas)
        train_S.append(s)
        train_E.append(e)
        train_by_regime[regime] = e
    train_S = np.vstack(train_S)
    train_E = np.vstack(train_E)
    neural = NeuralDecoder(code, kind="logreg").fit(train_S, train_E)

    # The spectral decoder is channel-adaptive: it carries one transfer-operator
    # model that is re-estimated from the operating regime's calibration sample
    # immediately before that regime is evaluated (see the per-regime loop).
    spectral = SpectralDecoder(code)
    roster = build_decoders(code, rng, neural=neural, spectral=spectral)
    rl = _train_rl(code, roster, rng, cfg.train_trials, cfg.p_eval, cfg.q_meas)
    if rl is not None:
        roster["rl"] = rl
    decoder_names = list(roster.keys())

    # ---- per-(regime, decoder) evaluation at p_eval ----------------------- #
    by_regime: Dict[str, Dict[str, Dict]] = {}
    per_decoder_fail: Dict[str, List[int]] = {d: [] for d in decoder_names}
    per_decoder_conf: Dict[str, List[float]] = {d: [] for d in decoder_names}
    per_decoder_correct: Dict[str, List[int]] = {d: [] for d in decoder_names}

    for regime in cfg.regimes:
        by_regime[regime] = {}
        # Channel-adaptive step: match the spectral transfer operator to this
        # regime's calibration sample before evaluating it.
        if "spectral" in roster:
            roster["spectral"].fit(train_by_regime[regime])
        for dname in decoder_names:
            rec = _evaluate(code, roster[dname], regime, cfg.trials,
                            cfg.p_eval, rng, cfg.q_meas, cfg.abstain_threshold)
            ler = metrics.logical_error_rate(rec["fails"])
            ece = metrics.expected_calibration_error(rec["confs"], rec["corrects"])
            coverage = float(np.mean(rec["committed"]))
            by_regime[regime][dname] = {
                "logical_error_rate": round(ler, 6),
                "ece": round(ece, 6),
                "coverage": round(coverage, 6),
            }
            per_decoder_fail[dname].extend(rec["fails"])
            per_decoder_conf[dname].extend(rec["confs"])
            per_decoder_correct[dname].extend(rec["corrects"])

    decoders_summary = {}
    for dname in decoder_names:
        decoders_summary[dname] = {
            "logical_error_rate": metrics.summarize(
                [float(x) for x in per_decoder_fail[dname]]
            ),
            "ece": round(metrics.expected_calibration_error(
                per_decoder_conf[dname], per_decoder_correct[dname]), 6),
        }

    # ---- threshold curves: logical vs physical error rate (iid) ----------- #
    # Trace the core decoders (the headline comparison); the learned/selector
    # decoders are summarised at p_eval above and omitted here to keep the smoke
    # run fast. The curve answers "how does logical error scale with p?".
    curve_decoders = [d for d in ("majority", "matching", "topology", "bp", "lookup",
                                  "fused", "spectral")
                      if d in decoder_names]
    # The threshold sweep is under i.i.d. noise, where the spectral transfer
    # operator is uniform and the decoder provably coincides with the optimal
    # leader: we calibrate it on the i.i.d. stream so the curve is honest (it
    # tracks lookup, claiming no i.i.d. advantage).
    if "spectral" in roster:
        roster["spectral"].fit(train_by_regime["iid"])
    threshold_curves: Dict[str, List[float]] = {d: [] for d in curve_decoders}
    for p in cfg.p_grid:
        for dname in curve_decoders:
            rec = _evaluate(code, roster[dname], "iid", cfg.trials,
                            p, rng, cfg.q_meas, cfg.abstain_threshold)
            threshold_curves[dname].append(round(metrics.logical_error_rate(rec["fails"]), 6))

    # ---- risk-coverage curve for the headline (spectral) decoder ---------- #
    headline_decoder = ("spectral" if "spectral" in decoder_names
                        else "fused" if "fused" in decoder_names else "matching")
    rc_curves = {}
    rc_curves[headline_decoder] = metrics.risk_coverage_curve(
        per_decoder_conf[headline_decoder], per_decoder_fail[headline_decoder]
    )
    if "matching" in decoder_names and "matching" != headline_decoder:
        rc_curves["matching"] = metrics.risk_coverage_curve(
            per_decoder_conf["matching"], per_decoder_fail["matching"]
        )

    # ---- integrity: cyclic-relabeling automorphism invariance ------------- #
    auto_ok = True
    if cfg.code == "repetition":
        check_errors = [noise.sample_iid(code.n_data, cfg.p_eval, rng) for _ in range(500)]
        auto_ok = is_automorphism_invariant(
            code, roster["matching"].decode, check_errors,
            shifts=[1, 2, code.n_data // 2], tol=0.06,
        )

    # ---- headline ---------------------------------------------------------- #
    best_decoder = min(decoder_names,
                       key=lambda d: decoders_summary[d]["logical_error_rate"]["mean"])
    maj = decoders_summary.get("majority", {}).get(
        "logical_error_rate", {}).get("mean", float("nan"))

    def _mean(dname):
        return decoders_summary.get(dname, {}).get(
            "logical_error_rate", {}).get("mean", float("nan"))

    spectral_ler = _mean("spectral")
    lookup_ler = _mean("lookup")
    fused_ler = _mean("fused") if "fused" in decoder_names else spectral_ler
    improvement = round(maj / max(1e-9, fused_ler), 3) if maj == maj else 0.0
    spectral_vs_lookup = (round(lookup_ler / max(1e-9, spectral_ler), 3)
                          if spectral_ler == spectral_ler and lookup_ler == lookup_ler else None)

    # Per-regime spectral advantage over the i.i.d.-optimal lookup leader, with a
    # two-sample z-score, so the structured-noise wins are explicit and auditable.
    spectral_by_regime = {}
    if "spectral" in decoder_names and "lookup" in decoder_names:
        for regime in cfg.regimes:
            sp = by_regime[regime]["spectral"]["logical_error_rate"]
            lk = by_regime[regime]["lookup"]["logical_error_rate"]
            se = float(np.sqrt(max(sp * (1 - sp), 1e-12) / cfg.trials
                               + max(lk * (1 - lk), 1e-12) / cfg.trials))
            spectral_by_regime[regime] = {
                "spectral_ler": round(sp, 6),
                "lookup_ler": round(lk, 6),
                "abs_reduction": round(lk - sp, 6),
                "rel_reduction": round((lk - sp) / lk, 4) if lk > 0 else 0.0,
                "z": round((lk - sp) / max(se, 1e-12), 2),
            }

    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "p_grid": list(cfg.p_grid),
        "automorphism_invariant": bool(auto_ok),
        "decoders": decoders_summary,
        "by_regime": by_regime,
        "threshold_curves": threshold_curves,
        "risk_coverage": rc_curves,
        "headline": {
            "best_decoder": best_decoder,
            "best_logical_error_rate": decoders_summary[best_decoder]["logical_error_rate"]["mean"],
            "spectral_logical_error_rate": round(float(spectral_ler), 6) if spectral_ler == spectral_ler else None,
            "lookup_logical_error_rate": round(float(lookup_ler), 6) if lookup_ler == lookup_ler else None,
            "spectral_vs_lookup_improvement": spectral_vs_lookup,
            "spectral_ece": decoders_summary.get("spectral", {}).get("ece"),
            "spectral_by_regime": spectral_by_regime,
            "fused_logical_error_rate": round(float(fused_ler), 6) if fused_ler == fused_ler else None,
            "majority_logical_error_rate": round(float(maj), 6) if maj == maj else None,
            "fused_vs_majority_improvement": improvement,
            "fused_ece": decoders_summary.get("fused", {}).get(
                "ece", decoders_summary.get(headline_decoder, {}).get("ece")),
            "headline_decoder": headline_decoder,
            "n_decoders": len(decoder_names),
            "distance": cfg.distance,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
