"""Paired cluster bootstrap, intersection-union, Holm, McNemar (PLAN §7.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

ALPHA = 0.05
BOOTSTRAPS = 10_000
H5_MARGIN = -0.01  # −1 point, on the fraction scale of metrics.py


def connected_components(sources: Sequence[Sequence[Hashable]]) -> list[int]:
    """Cluster per item: items sharing any source clip, transitively (appendix, XM-Test)."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for clips in sources:
        for clip in clips[1:]:
            parent[find(clip)] = find(clips[0])
    roots: dict = {}
    return [roots.setdefault(find(clips[0]), len(roots)) for clips in sources]


def nested_recall(correct: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> float:
    """Mean over `outer` of mean recall over `inner`. Pairs a resample lost are skipped."""
    outer_n, inner_n = int(outer.max()) + 1, int(inner.max()) + 1
    cell = outer * inner_n + inner
    counts = np.bincount(cell, minlength=outer_n * inner_n).reshape(outer_n, inner_n)
    hits = np.bincount(cell, weights=correct, minlength=outer_n * inner_n).reshape(outer_n, inner_n)
    seen = counts > 0
    recall = np.divide(hits, counts, out=np.zeros_like(hits), where=seen)
    per_outer = recall.sum(1)[seen.any(1)] / seen.sum(1)[seen.any(1)]
    return float(per_outer.mean())


@dataclass
class Term:
    """One benchmark of a composite. outer/inner default to plain accuracy. Build XRS terms with
    cmm_term, avhbench_term and xm_term."""
    clusters: np.ndarray
    outer: np.ndarray
    inner: np.ndarray

    def score(self, correct: np.ndarray, idx: np.ndarray | None = None) -> float:
        """correct is [seeds, items]; seeds are averaged."""
        if idx is None:
            idx = np.arange(correct.shape[1])
        return float(np.mean([nested_recall(row[idx], self.outer[idx], self.inner[idx]) for row in correct]))


def _codes(values: Sequence) -> np.ndarray:
    return np.unique(np.asarray(values, dtype=object).astype(str), return_inverse=True)[1]


def accuracy_term(clusters: Sequence | None, n: int) -> Term:
    clusters = _codes(range(n) if clusters is None else clusters)
    return Term(clusters, np.zeros(n, int), np.zeros(n, int))


def cmm_term(sample_ids: Sequence, golds: Sequence[str]) -> Term:
    n = len(golds)
    return Term(_codes(sample_ids), np.zeros(n, int), _codes(golds))


def avhbench_term(tasks: Sequence[str], golds: Sequence[str]) -> Term:
    n = len(golds)
    return Term(np.arange(n), _codes(tasks), _codes(golds))


def xm_term(sources: Sequence[Sequence[str]], twin_ids: Sequence[str], families: Sequence[str], classes: Sequence[str]) -> Term:
    links = [(*s, f"twin:{t}") for s, t in zip(sources, twin_ids)]
    return Term(np.asarray(connected_components(links)), _codes(families), _codes(classes))


def paired_delta_bootstrap(terms: Mapping[str, Term], a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray],
                           B: int = BOOTSTRAPS, seed: int = 0, alpha: float = ALPHA) -> dict:
    """Δ = mean over terms of score(a) − score(b). Same resampled clusters for both systems,
    stratified by term. a[term] and b[term] are [seeds, items] booleans; seed counts may differ."""
    rng = np.random.default_rng(seed)
    members = {name: [np.flatnonzero(t.clusters == c) for c in np.unique(t.clusters)] for name, t in terms.items()}
    delta = float(np.mean([t.score(a[n]) - t.score(b[n]) for n, t in terms.items()]))
    boot = np.empty(B)
    for i in range(B):
        values = []
        for name, term in terms.items():
            groups = members[name]
            picks = rng.integers(len(groups), size=len(groups))
            idx = np.concatenate([groups[p] for p in picks])
            values.append(term.score(a[name], idx) - term.score(b[name], idx))
        boot[i] = np.mean(values)
    return {
        "delta": delta,
        "lower": float(np.quantile(boot, alpha)),  # one-sided lower 95%
        "p": float((1 + np.sum(boot <= 0)) / (B + 1)),
        "seed_range_a": _seed_range(terms, a),
        "seed_range_b": _seed_range(terms, b),
    }


def _seed_range(terms: Mapping[str, Term], correct: Mapping[str, np.ndarray]) -> list[float]:
    """Min and max composite over seeds; §7.4 asks to always report it."""
    seeds = min(correct[n].shape[0] for n in terms)
    scores = [np.mean([t.score(correct[n][s:s + 1]) for n, t in terms.items()]) for s in range(seeds)]
    return [float(min(scores)), float(max(scores))]


def holm(pvalues: Mapping[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Step-down rejections."""
    ordered = sorted(pvalues, key=pvalues.get)
    rejected = {name: False for name in pvalues}
    for rank, name in enumerate(ordered):
        if pvalues[name] > alpha / (len(ordered) - rank):
            break
        rejected[name] = True
    return rejected


def intersection_union(lowers: Sequence[float], margin: float = 0.0) -> bool:
    return len(lowers) > 0 and all(lb > margin for lb in lowers)


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    """Exact two-sided p on discordant pairs. Secondary, per-benchmark accuracy only."""
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def decide_backbone(results: Mapping[str, Mapping], alpha: float = ALPHA) -> dict[str, bool]:
    """§7.4 Holm family within one backbone, keyed by contrast: "H3", "H3b" and on the primary
    backbone "H4_A5", "H4_A4", "H4_A3". A contrast passes on lower bound > 0 and Holm rejection;
    the bootstrap p carries Holm's adjustment since the 95% bound itself is unadjusted."""
    rejected = holm({k: r["p"] for k, r in results.items()}, alpha)
    return {k: rejected[k] and results[k]["lower"] > 0 for k in results}


def decide(backbones: Mapping[str, Mapping[str, Mapping]], alpha: float = ALPHA) -> dict:
    """backbones[name] holds "H1", "H5" and the Holm-family contrasts, each from paired_delta_bootstrap."""
    holm_family = {b: {k: v for k, v in r.items() if k not in ("H1", "H5")} for b, r in backbones.items()}
    return {
        "H1": intersection_union([r["H1"]["lower"] for r in backbones.values()]),
        "H5": intersection_union([r["H5"]["lower"] for r in backbones.values()], H5_MARGIN),
        "per_backbone": {b: decide_backbone(family, alpha) for b, family in holm_family.items()},
    }
