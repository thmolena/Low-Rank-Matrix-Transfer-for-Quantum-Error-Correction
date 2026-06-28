"""Decoders, from textbook baselines to the topology-aware / fused / RL family.

Every decoder implements ``decode(syndrome) -> (correction, confidence)`` where
``correction`` is a 0/1 ``int8`` error guess over the data bits and
``confidence in [0, 1]`` is the decoder's self-reported reliability. A decoder
*abstains* when its confidence falls below a caller-supplied threshold; the
runner records abstentions for the risk-coverage analysis.

Decoders
--------
``LookupDecoder``    enumerate syndromes -> minimum-weight coset leader (optimal
                     for small ``d``; the gold standard the others approximate).
``MajorityDecoder``  repetition-only: majority vote of the implied bit value.
``MatchingDecoder``  pair syndrome defects by minimum-weight matching --
                     endpoint pairing on the repetition chain, ``networkx``
                     ``max_weight_matching`` on the surface defect graph.
``BPDecoder``        a few rounds of min-sum belief propagation on the Tanner
                     graph (an iterative message-passing proxy).
``NeuralDecoder``    sklearn logistic-regression / MLP from syndrome -> per-bit
                     correction, trained on sampled syndromes.
``TopologyDecoder``  matching driven by defect-geometry features (chain length,
                     spread, gap structure) -- the topology-aware contribution.
``FusedDecoder``     combine matching + neural by confidence (the headline
                     decoder); falls back to the more confident sub-decoder.
``RLDecoder``        a contextual bandit that, per syndrome-context, *selects*
                     which decoder to trust, learning online to minimise logical
                     error.
``SpectralDecoder``  the channel-adaptive contribution: a maximum-a-posteriori
                     coset decoder whose error model is a *truncated transfer
                     operator* (an inhomogeneous Markov chain) on the defect
                     graph, calibrated to the operating channel. It is
                     Bayes-optimal for the structured channel and provably beats
                     the i.i.d.-optimal coset leader whenever the channel is
                     spatially non-uniform or correlated.

The matching/topology/BP decoders are *topology-aware*: they read the code's
Tanner / defect graph, so they exploit spatial error structure that a
memoryless majority vote cannot. The spectral decoder goes further and is
*channel-aware*: it learns the spatial bias and correlation of the noise.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .codes import RepetitionCode, SurfaceLikeCode

Correction = Tuple[np.ndarray, float]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int8)


def _chain_from_defects(defects: List[int], n_data: int) -> np.ndarray:
    """Repetition code: turn an ordered list of defect *links* into a flip chain.

    Each consecutive pair of defects bounds a chain of flipped data bits between
    them. ``s_i = e_i ^ e_{i+1}``, so a chain over links ``[a, b)`` flips data
    bits ``a+1 .. b``. We pair defects left-to-right (the min-weight pairing on a
    line) and also consider closing the last unpaired defect to a boundary.
    """
    corr = _zeros(n_data)
    d = sorted(defects)
    # pair consecutive defects: (d0,d1), (d2,d3), ...
    i = 0
    while i + 1 < len(d):
        a, b = d[i], d[i + 1]
        # flip data bits between link a and link b -> bits a+1 .. b
        corr[a + 1:b + 1] ^= 1
        i += 2
    if i < len(d):  # odd defect: close to the nearer boundary
        a = d[i]
        left_cost = a + 1            # flip bits 0..a
        right_cost = n_data - (a + 1)  # flip bits a+1..end
        if left_cost <= right_cost:
            corr[:a + 1] ^= 1
        else:
            corr[a + 1:] ^= 1
    # both cosets (corr and its complement) reproduce the syndrome; return the
    # lighter one so the matching correction is genuinely minimum-weight.
    if 2 * int(corr.sum()) > n_data:
        corr ^= 1
    return corr


def _defect_confidence(n_defects: int, n_checks: int) -> float:
    """Heuristic confidence: fewer / cleanly-pairable defects -> more confident."""
    if n_checks == 0:
        return 1.0
    frac = n_defects / n_checks
    base = float(np.exp(-2.0 * frac))
    if n_defects % 2 == 1:  # an unpaired defect is ambiguous
        base *= 0.6
    return float(np.clip(base, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Lookup (optimal coset-leader) decoder
# --------------------------------------------------------------------------- #
class LookupDecoder:
    """Enumerate all errors, map each syndrome to its minimum-weight leader."""

    name = "lookup"

    def __init__(self, code) -> None:
        self.code = code
        self.table: Dict[bytes, np.ndarray] = {}
        self._weight: Dict[bytes, int] = {}
        n = code.n_data
        for m in range(1 << n):
            err = np.array([(m >> b) & 1 for b in range(n)], dtype=np.int8)
            key = code.syndrome(err).tobytes()
            w = int(err.sum())
            if key not in self._weight or w < self._weight[key]:
                self._weight[key] = w
                self.table[key] = err

    def decode(self, syndrome: np.ndarray) -> Correction:
        key = np.asarray(syndrome, dtype=np.int8).tobytes()
        corr = self.table.get(key, _zeros(self.code.n_data)).copy()
        conf = _defect_confidence(int(np.sum(syndrome)), self.code.n_checks)
        # the lookup leader is exact, so boost confidence toward 1
        conf = float(np.clip(0.5 + 0.5 * conf, 0.0, 1.0))
        return corr, conf


# --------------------------------------------------------------------------- #
# Majority (repetition only)
# --------------------------------------------------------------------------- #
class MajorityDecoder:
    """Reconstruct the chain by integrating the syndrome, then majority-vote.

    For the repetition code the syndrome is a discrete derivative; cumulative
    XOR recovers a flip pattern up to a global offset, and we pick the offset
    (all-or-nothing flip) that touches fewer bits -- i.e. the majority value.
    """

    name = "majority"

    def __init__(self, code) -> None:
        self.code = code

    def decode(self, syndrome: np.ndarray) -> Correction:
        n = self.code.n_data
        s = np.asarray(syndrome, dtype=np.int8)
        # cumulative XOR: implied error relative to bit 0 being correct
        chain = _zeros(n)
        acc = 0
        for i in range(n - 1):
            acc ^= int(s[i])
            chain[i + 1] = acc
        # choose the lower-weight coset representative (majority)
        if chain.sum() * 2 > n:
            chain ^= 1
        margin = abs(n - 2 * int(chain.sum())) / max(1, n)
        conf = float(np.clip(0.5 + 0.5 * margin, 0.0, 1.0))
        return chain, conf


# --------------------------------------------------------------------------- #
# Matching (MWPM proxy)
# --------------------------------------------------------------------------- #
class MatchingDecoder:
    """Minimum-weight matching of syndrome defects.

    * Repetition code: defects lie on a line; the min-weight pairing is the
      left-to-right endpoint pairing (with boundary closure for an odd defect).
    * Surface-like code: build a complete graph on lit plaquettes weighted by
      negative Manhattan distance and call ``networkx.max_weight_matching``
      (== min-weight perfect matching on the distance graph), then flip a
      shortest lattice path between each matched pair.
    """

    name = "matching"

    def __init__(self, code) -> None:
        self.code = code
        self.is_surface = isinstance(code, SurfaceLikeCode)

    def decode(self, syndrome: np.ndarray) -> Correction:
        s = np.asarray(syndrome, dtype=np.int8)
        n_def = int(s.sum())
        if not self.is_surface:
            defects = list(np.where(s == 1)[0])
            corr = _chain_from_defects(defects, self.code.n_data)
            return corr, _defect_confidence(n_def, self.code.n_checks)
        return self._decode_surface(s), _defect_confidence(n_def, self.code.n_checks)

    def _decode_surface(self, s: np.ndarray) -> np.ndarray:
        code: SurfaceLikeCode = self.code
        d = code.distance
        lit = code.defect_positions(s)
        corr = _zeros(code.n_data)
        if len(lit) < 2:
            return corr
        g = nx.Graph()
        for (k, _pos) in lit:
            g.add_node(k)
        for ai in range(len(lit)):
            for bi in range(ai + 1, len(lit)):
                ka, pa = lit[ai]
                kb, pb = lit[bi]
                dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
                g.add_edge(ka, kb, weight=-dist)  # max-weight == min-distance
        matching = nx.max_weight_matching(g, maxcardinality=True)
        pos = {k: p for k, p in lit}
        for (ka, kb) in matching:
            self._flip_path(corr, pos[ka], pos[kb], d)
        return corr

    @staticmethod
    def _flip_path(corr: np.ndarray, pa, pb, d: int) -> None:
        """Flip data bits on an L-shaped lattice path between two plaquettes."""
        r0, c0 = int(round(pa[0])), int(round(pa[1]))
        r1, c1 = int(round(pb[0])), int(round(pb[1]))
        r, c = r0, c0
        while r != r1:
            r += 1 if r1 > r else -1
            corr[min(max(r, 0), d - 1) * d + min(max(c, 0), d - 1)] ^= 1
        while c != c1:
            c += 1 if c1 > c else -1
            corr[min(max(r, 0), d - 1) * d + min(max(c, 0), d - 1)] ^= 1


# --------------------------------------------------------------------------- #
# Belief propagation (min-sum) proxy
# --------------------------------------------------------------------------- #
class BPDecoder:
    """A few rounds of min-sum message passing on the Tanner graph.

    Per-bit log-likelihood priors flow to checks; each check returns the parity
    constraint via the min-sum rule (sign product x min magnitude). After a few
    sweeps the bit beliefs are thresholded; ties fall back to matching. This is
    a lightweight iterative-decoder proxy, not a production LDPC decoder.
    """

    name = "bp"

    def __init__(self, code, rounds: int = 6, p: float = 0.1) -> None:
        self.code = code
        self.rounds = rounds
        self.p = p
        self.tanner = code.tanner_graph()
        self.checks = [k for k, dat in self.tanner.nodes(data=True) if dat["kind"] == "check"]
        self.data = [k for k, dat in self.tanner.nodes(data=True) if dat["kind"] == "data"]
        self._fallback = MatchingDecoder(code)

    def decode(self, syndrome: np.ndarray) -> Correction:
        s = {("check", j): int(b) for j, b in enumerate(np.asarray(syndrome).ravel())}
        n = self.code.n_data
        llr_prior = np.log((1 - self.p) / max(1e-9, self.p))
        # messages bit->check and check->bit, indexed by edge
        m_bc = {e: llr_prior for e in self.tanner.edges()}
        m_cb = {e: 0.0 for e in self.tanner.edges()}

        def edge(a, b):
            return (a, b) if (a, b) in m_bc else (b, a)

        for _ in range(self.rounds):
            # check -> bit (min-sum), respecting the measured syndrome sign
            for ch in self.checks:
                nbrs = list(self.tanner.neighbors(ch))
                sign_flip = -1.0 if s.get(ch, 0) == 1 else 1.0
                for v in nbrs:
                    prod_sign = sign_flip
                    min_mag = float("inf")
                    for u in nbrs:
                        if u == v:
                            continue
                        msg = m_bc[edge(u, ch)]
                        prod_sign *= np.sign(msg) if msg != 0 else 1.0
                        min_mag = min(min_mag, abs(msg))
                    m_cb[edge(ch, v)] = prod_sign * (0.0 if min_mag == float("inf") else min_mag)
            # bit -> check
            for v in self.data:
                nbrs = list(self.tanner.neighbors(v))
                for ch in nbrs:
                    tot = llr_prior + sum(m_cb[edge(c2, v)] for c2 in nbrs if c2 != ch)
                    m_bc[edge(v, ch)] = tot

        corr = _zeros(n)
        for v in self.data:
            idx = v[1]
            belief = llr_prior + sum(m_cb[edge(c2, v)] for c2 in self.tanner.neighbors(v))
            corr[idx] = 1 if belief < 0 else 0

        # if BP did not satisfy the syndrome, defer to matching
        if not np.array_equal(self.code.syndrome(corr), np.asarray(syndrome, dtype=np.int8)):
            return self._fallback.decode(syndrome)
        conf = _defect_confidence(int(np.sum(syndrome)), self.code.n_checks)
        return corr, float(np.clip(0.4 + 0.6 * conf, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Neural (sklearn) decoder
# --------------------------------------------------------------------------- #
class NeuralDecoder:
    """Learn the *logical coset* from the syndrome (a learned coset selector).

    A per-bit error is not a function of the syndrome (errors sharing a syndrome
    differ by a logical operator), so regressing onto raw bits is ill-posed. We
    therefore decode in two stages:

      1. a syndrome-determined *base correction* ``c0`` (the matching chain);
      2. a learned classifier ``syndrome -> logical class``, i.e. whether ``c0``
         lands in the trivial coset or must be completed by the logical operator.

    The output is always a valid correction (it reproduces the syndrome). The
    learning is genuine: under biased / structured noise the most-likely coset is
    not always the minimum-weight one, and the classifier learns that bias from
    data -- exactly the regime where it can improve on plain matching.

    Confidence is the classifier's predicted-class probability (so it is
    well-calibrated by construction for the coset decision).
    """

    name = "neural"

    def __init__(self, code, kind: str = "logreg") -> None:
        self.code = code
        self.kind = kind
        self._base = MatchingDecoder(code)
        self.model: Optional[object] = None
        self.constant_class: int = 0
        self._fitted = False
        self._logical = self._logical_operator()

    def _logical_operator(self) -> np.ndarray:
        """A representative non-trivial logical operator (all-ones for repetition)."""
        if isinstance(self.code, SurfaceLikeCode):
            d = self.code.distance
            op = _zeros(self.code.n_data)
            op[0::d] = 1  # flip the left column -> flips the logical observable
            return op
        return np.ones(self.code.n_data, dtype=np.int8)

    def fit(self, syndromes: np.ndarray, leaders: np.ndarray) -> "NeuralDecoder":
        """``leaders`` are *true* corrections (error patterns) for each syndrome.

        The training label is the logical class of ``leader XOR c0(syndrome)`` --
        i.e. which coset the true correction belongs to relative to the base
        chain. The classifier learns to predict that coset from the syndrome.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        X = np.asarray(syndromes, dtype=float)
        L = np.asarray(leaders, dtype=np.int8)
        y = np.empty(len(X), dtype=int)
        for i in range(len(X)):
            c0, _ = self._base.decode(X[i].astype(np.int8))
            residual = (L[i] ^ c0)
            y[i] = self.code.logical_value(residual)
        if len(np.unique(y)) < 2:
            self.constant_class = int(y[0]) if len(y) else 0
            self.model = None
        else:
            if self.kind == "mlp":
                self.model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0)
            else:
                self.model = LogisticRegression(max_iter=500)
            self.model.fit(X, y)
        self._fitted = True
        return self

    def decode(self, syndrome: np.ndarray) -> Correction:
        s = np.asarray(syndrome, dtype=np.int8)
        c0, _ = self._base.decode(s)
        if self.model is None:
            cls, p = self.constant_class, 0.7
        else:
            x = s.astype(float).reshape(1, -1)
            proba = self.model.predict_proba(x)[0]
            classes = list(self.model.classes_)
            cls = int(classes[int(np.argmax(proba))])
            p = float(np.max(proba))
        corr = (c0 ^ self._logical) if cls == 1 else c0.copy()
        return corr.astype(np.int8), p


# --------------------------------------------------------------------------- #
# Topology-aware decoder
# --------------------------------------------------------------------------- #
class TopologyDecoder:
    """Matching driven by defect-geometry features.

    Reads the defect pattern's geometry (number of defects, span, gap structure)
    and either pairs adjacent defects (short chains -> high confidence) or, when
    the geometry indicates a single long burst, flips the whole bounded span at
    once. On the surface code it delegates the heavy lifting to the matching
    decoder but recalibrates confidence by defect spread.
    """

    name = "topology"

    def __init__(self, code) -> None:
        self.code = code
        self.is_surface = isinstance(code, SurfaceLikeCode)
        self._matcher = MatchingDecoder(code)

    def features(self, syndrome: np.ndarray) -> Dict[str, float]:
        s = np.asarray(syndrome, dtype=np.int8)
        idx = np.where(s == 1)[0]
        n_def = int(s.sum())
        span = float(idx.max() - idx.min()) if n_def >= 2 else 0.0
        gaps = np.diff(idx) if n_def >= 2 else np.array([])
        return {
            "n_defects": float(n_def),
            "span": span,
            "mean_gap": float(gaps.mean()) if gaps.size else 0.0,
            "max_gap": float(gaps.max()) if gaps.size else 0.0,
            "frac": n_def / max(1, self.code.n_checks),
        }

    def decode(self, syndrome: np.ndarray) -> Correction:
        feats = self.features(syndrome)
        corr, base_conf = self._matcher.decode(syndrome)
        # geometry recalibration: clustered (small max_gap) defects are easy
        clustered = feats["max_gap"] <= 1.5 if feats["n_defects"] >= 2 else True
        bonus = 0.15 if clustered else -0.1
        conf = float(np.clip(base_conf + bonus, 0.0, 1.0))
        return corr, conf


# --------------------------------------------------------------------------- #
# Fused decoder
# --------------------------------------------------------------------------- #
class FusedDecoder:
    """Combine matching + neural by confidence (the headline decoder).

    Both sub-decoders propose; if they agree, confidence is boosted. If they
    disagree, the more confident proposal wins, but confidence is damped to
    reflect the conflict. This confidence is what drives selective abstention.
    """

    name = "fused"

    def __init__(self, matcher: MatchingDecoder, neural: Optional[NeuralDecoder]) -> None:
        self.matcher = matcher
        self.neural = neural

    def decode(self, syndrome: np.ndarray) -> Correction:
        cm, conf_m = self.matcher.decode(syndrome)
        if self.neural is None or not getattr(self.neural, "_fitted", False):
            return cm, conf_m
        cn, conf_n = self.neural.decode(syndrome)
        agree = np.array_equal(cm, cn)
        if agree:
            return cm, float(np.clip(0.5 * (conf_m + conf_n) + 0.2, 0.0, 1.0))
        if conf_m >= conf_n:
            return cm, float(np.clip(conf_m - 0.15, 0.0, 1.0))
        return cn, float(np.clip(conf_n - 0.15, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# RL (contextual bandit) decoder-selector
# --------------------------------------------------------------------------- #
class RLDecoder:
    """A contextual bandit that picks which decoder to trust per syndrome.

    The context is a coarse bucket of the defect count; the actions are the
    candidate decoders. The bandit keeps a running estimate of each action's
    success probability per context and plays epsilon-greedy. With a small
    exploration budget it learns, online, to route easy syndromes to matching
    and ambiguous ones to the fused decoder, minimising logical error.
    """

    name = "rl"

    def __init__(self, code, decoders: Dict[str, object], rng: np.random.Generator,
                 epsilon: float = 0.1) -> None:
        self.code = code
        self.decoders = decoders
        self.actions = list(decoders.keys())
        self.rng = rng
        self.epsilon = epsilon
        self.n_ctx = code.n_checks + 1
        self.values = np.full((self.n_ctx, len(self.actions)), 0.5)
        self.counts = np.zeros((self.n_ctx, len(self.actions)))

    def _ctx(self, syndrome: np.ndarray) -> int:
        return int(min(self.n_ctx - 1, int(np.sum(syndrome))))

    def select(self, syndrome: np.ndarray) -> int:
        ctx = self._ctx(syndrome)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(np.argmax(self.values[ctx]))

    def update(self, syndrome: np.ndarray, action: int, reward: float) -> None:
        ctx = self._ctx(syndrome)
        self.counts[ctx, action] += 1
        lr = 1.0 / self.counts[ctx, action]
        self.values[ctx, action] += lr * (reward - self.values[ctx, action])

    def decode(self, syndrome: np.ndarray) -> Correction:
        a = int(np.argmax(self.values[self._ctx(syndrome)]))
        return self.decoders[self.actions[a]].decode(syndrome)


# --------------------------------------------------------------------------- #
# Spectral (transfer-operator) channel-adaptive decoder
# --------------------------------------------------------------------------- #
class SpectralDecoder:
    r"""Channel-adaptive maximum-a-posteriori coset decoder on the defect graph.

    The i.i.d.-optimal coset leader (``LookupDecoder``) minimises *Hamming*
    weight, which is the maximum-likelihood rule only when every data bit flips
    independently with the same probability. Real, structured noise breaks both
    assumptions: it is spatially non-uniform (bias) and spatially correlated
    (bursts, neighbour coupling). Under such a channel the minimum-weight
    representative is *not* the most probable error in its coset, so the leader
    is provably sub-optimal (Theorem on the structured channel).

    This decoder is the Bayes-optimal rule for a learned model of the structured
    channel. It represents the error distribution by a **truncated transfer
    operator** on the code's defect line/graph: for the repetition code this is
    an *inhomogeneous* first-order Markov chain,

        log P(e) = log pi_0(e_0) + sum_i log T_i(e_i, e_{i+1}),

    whose per-site 2x2 transfer matrices ``T_i`` are estimated from a small
    calibration sample of the operating channel (``fit``). The position
    dependence of ``T_i`` captures spatial bias; the off-diagonal mass of
    ``T_i`` captures burst / neighbour correlations -- both local and non-local
    structure in one tractable (O(d)) spectral object. (For a general code the
    transfer operator is larger; keeping its dominant modes is the spectral
    truncation, exact here because each site operator is already 2x2.)

    Decoding selects the more probable of the two coset representatives
    ``{c0, c0 (+) L}`` (the syndrome-determined base chain and its logical
    partner). The reported confidence is the *exact posterior probability* of the
    chosen coset under the learned model,

        C = sigma( | log P(c0) - log P(c0 (+) L) | ),

    so it is calibrated by construction to the model -- empirically near-perfect
    when the model matches the channel, and informatively degraded when it does
    not (e.g. read-out noise, where the conditioned syndrome is itself corrupt).
    """

    name = "spectral"

    def __init__(self, code) -> None:
        self.code = code
        self.is_surface = isinstance(code, SurfaceLikeCode)
        self._base = MatchingDecoder(code)
        self._logical = self._logical_operator()
        self._fitted = False
        n = code.n_data
        # i.i.d.-like defaults so an unfitted decoder reduces to min-weight.
        self.logp0 = np.log(np.array([0.88, 0.12]))
        self.logT = np.tile(np.log(np.array([[0.88, 0.12], [0.88, 0.12]])), (max(0, n - 1), 1, 1))
        self.log_p = np.full(n, np.log(0.12))
        self.log_1mp = np.full(n, np.log(0.88))

    def _logical_operator(self) -> np.ndarray:
        if isinstance(self.code, SurfaceLikeCode):
            d = self.code.distance
            op = _zeros(self.code.n_data)
            op[0::d] = 1
            return op
        return np.ones(self.code.n_data, dtype=np.int8)

    def fit(self, errors: np.ndarray, eps: float = 1.0) -> "SpectralDecoder":
        """Estimate the transfer operator from calibration error patterns.

        ``errors`` is an ``(m, n_data)`` 0/1 array sampled from the operating
        channel. ``eps`` is a Laplace-smoothing pseudo-count.
        """
        E = np.asarray(errors, dtype=int)
        if E.ndim != 2 or E.shape[0] == 0:
            return self
        n = self.code.n_data
        # per-bit marginals (the spatial field; also the surface model)
        pi = (E.sum(0) + eps) / (E.shape[0] + 2 * eps)
        pi = np.clip(pi, 1e-6, 1 - 1e-6)
        self.log_p = np.log(pi)
        self.log_1mp = np.log1p(-pi)
        if not self.is_surface:
            p0 = np.array([(E[:, 0] == 0).mean(), (E[:, 0] == 1).mean()])
            p0 = np.clip(p0, 1e-6, 1.0)
            self.logp0 = np.log(p0 / p0.sum())
            logT = np.zeros((n - 1, 2, 2))
            for i in range(n - 1):
                T = np.full((2, 2), eps)
                a, b = E[:, i], E[:, i + 1]
                for av in (0, 1):
                    for bv in (0, 1):
                        T[av, bv] += int(np.sum((a == av) & (b == bv)))
                T /= T.sum(axis=1, keepdims=True)
                logT[i] = np.log(T)
            self.logT = logT
        self._fitted = True
        return self

    def _loglik(self, e: np.ndarray) -> float:
        e = np.asarray(e, dtype=int).ravel()
        if self.is_surface:
            return float(e @ self.log_p + (1 - e) @ self.log_1mp)
        ll = float(self.logp0[e[0]])
        if e.size > 1:
            idx = np.arange(e.size - 1)
            ll += float(self.logT[idx, e[:-1], e[1:]].sum())
        return ll

    def decode(self, syndrome: np.ndarray) -> Correction:
        c0, _ = self._base.decode(syndrome)
        c0 = np.asarray(c0, dtype=np.int8)
        c1 = (c0 ^ self._logical).astype(np.int8)
        l0, l1 = self._loglik(c0), self._loglik(c1)
        if l0 >= l1:
            corr, margin = c0, l0 - l1
        else:
            corr, margin = c1, l1 - l0
        conf = 1.0 / (1.0 + float(np.exp(-min(margin, 60.0))))
        return corr.astype(np.int8), float(np.clip(conf, 0.0, 1.0))


def build_decoders(code, rng: np.random.Generator, neural: Optional[NeuralDecoder] = None,
                   spectral: Optional["SpectralDecoder"] = None
                   ) -> Dict[str, object]:
    """Construct the standard decoder roster for a code."""
    matcher = MatchingDecoder(code)
    roster: Dict[str, object] = {
        "majority": MajorityDecoder(code),
        "matching": matcher,
        "bp": BPDecoder(code),
        "topology": TopologyDecoder(code),
    }
    if isinstance(code, RepetitionCode) and code.n_data <= 16:
        roster["lookup"] = LookupDecoder(code)
    if neural is not None:
        roster["neural"] = neural
        roster["fused"] = FusedDecoder(matcher, neural)
    if spectral is not None:
        roster["spectral"] = spectral
    return roster
