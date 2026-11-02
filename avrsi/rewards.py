"""Reward terms of PLAN §3.3. Verdicts arrive already reward-resolved (critic.verdicts_for_reward)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from .parse import Item, ParsedPath, canonical_answer, gold_answer

N_DEFAULT = 3
N_SILENCE = 1


def r_outcome(path: ParsedPath, item: Item) -> float:
    if not path.valid:
        return 0.0
    return float(canonical_answer(path.answer, item) == gold_answer(item))


def s_m(verdicts: Sequence[str], n: int = N_DEFAULT) -> float:
    n_m = len(verdicts)
    if n_m == 0:
        return 0.0
    supported = sum(v == "supported" for v in verdicts)
    return min(1.0, supported / n) * supported / n_m


def r_xm(audio_verdicts: Sequence[str], visual_verdicts: Sequence[str], silence: bool = False) -> float:
    return 0.5 * (s_m(audio_verdicts, N_SILENCE if silence else N_DEFAULT) + s_m(visual_verdicts))


def pairwise_r(valid: Sequence[bool], outcomes: Mapping[tuple[int, int], str | None]) -> list[list[float]]:
    """r[k][j] = r(k vs j). outcomes[(a, b)] is the parsed verdict with path a shown as A and b as B;
    one entry per order that ran. None is a parse failure, a tie."""
    K = len(valid)
    r = [[0.5] * K for _ in range(K)]
    for j in range(K):
        for k in range(j + 1, K):
            if not valid[j] or not valid[k]:
                value = 0.5 if valid[j] == valid[k] else float(valid[j])
            else:
                value = _ordered_mean(j, k, outcomes)
            r[j][k], r[k][j] = value, 1.0 - value
    return r


def _ordered_mean(j: int, k: int, outcomes: Mapping[tuple[int, int], str | None]) -> float:
    scores = []
    if (j, k) in outcomes:
        scores.append({"A": 1.0, "B": 0.0}.get(outcomes[(j, k)], 0.5))
    if (k, j) in outcomes:
        scores.append({"A": 0.0, "B": 1.0}.get(outcomes[(k, j)], 0.5))
    if not scores:
        raise KeyError(f"no AV-pair verdict for well-formed pair ({j}, {k})")
    return sum(scores) / len(scores)


def r_pair(r: Sequence[Sequence[float]]) -> list[float]:
    K = len(r)
    if K < 2:
        return [0.0] * K
    return [sum(r[k][j] for j in range(K) if j != k) / (K - 1) for k in range(K)]


def r_total(outcome: float, fmt: float, pair: float, xm: float, lam: float, alpha: float, gamma: float) -> float:
    return outcome + lam * fmt + alpha * pair + gamma * xm


def r_total_rl_comm(outcome: float, fmt: float, r_arr: float, r_avc: float, lam: float) -> float:
    """A7: r_format + r_arr + r_avc replace R_pair, R_xm and R_format; format counted once."""
    return outcome + lam * fmt + r_arr + r_avc


def group_advantages(totals: Sequence[float]) -> list[float]:
    mean = sum(totals) / len(totals)
    return [t - mean for t in totals]
