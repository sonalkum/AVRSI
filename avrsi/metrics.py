"""PLAN §7 metrics and §9 planted-error scoring. Rates are fractions in [0, 1]."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .parse import normalise_tokens

GATE_DETECTION, GATE_H2_MARGIN, GATE_ORDER, GATE_SUPPORTED, GATE_PARSE = 0.70, 0.05, 0.85, (0.6, 0.95), 0.10


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def accuracy(preds: Sequence, golds: Sequence) -> float:
    return _mean(p == g for p, g in zip(preds, golds))


def balanced_accuracy(preds: Sequence, golds: Sequence) -> float:
    """Mean recall over the answer classes present in golds."""
    hits: dict = defaultdict(list)
    for p, g in zip(preds, golds):
        hits[g].append(p == g)
    return _mean(_mean(v) for v in hits.values())


def f1(preds: Sequence, golds: Sequence, positive: str = "yes") -> float:
    tp = sum(p == positive and g == positive for p, g in zip(preds, golds))
    fp = sum(p == positive and g != positive for p, g in zip(preds, golds))
    fn = sum(p != positive and g == positive for p, g in zip(preds, golds))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def yes_rate(preds: Sequence) -> float:
    return _mean(p == "yes" for p in preds)


def xm_balanced_accuracy(rows: Sequence[Mapping]) -> float:
    """Mean over families of that family's mean recall over canonical classes. rows: family, answer_class, correct."""
    recall: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        recall[row["family"]][row["answer_class"]].append(row["correct"])
    return _mean(_mean(_mean(v) for v in classes.values()) for classes in recall.values())


def pa_hr(preds: Sequence, golds: Sequence) -> tuple[float, float]:
    yes = [p == "yes" for p, g in zip(preds, golds) if g == "yes"]
    no = [p == "no" for p, g in zip(preds, golds) if g == "no"]
    return _mean(yes), _mean(no)


def cmm_score(preds: Sequence, golds: Sequence) -> float:
    pa, hr = pa_hr(preds, golds)
    return (pa + hr) / 2


def avhbench_score(rows: Sequence[Mapping]) -> float:
    """Mean of A→V and V→A balanced accuracy. rows: task ("a2v", "v2a", ...), pred, gold."""
    scores = []
    for task in ("a2v", "v2a"):
        subset = [r for r in rows if r["task"] == task]
        if not subset:
            raise ValueError(f"AVHBench task {task} missing")
        scores.append(balanced_accuracy([r["pred"] for r in subset], [r["gold"] for r in subset]))
    return _mean(scores)


def xrs(cmm: float, avhbench: float, xm_test: float) -> float:
    return (cmm + avhbench + xm_test) / 3


def swap_sensitivity(rows: Sequence[Mapping]) -> float:
    """P(correct on both twins), over complete twin pairs. rows: twin_id, kind, correct."""
    twins: dict = defaultdict(dict)
    for row in rows:
        twins[row["twin_id"]]["aligned" if row["kind"] == "aligned" else "swapped"] = row["correct"]
    return _mean(t["aligned"] and t["swapped"] for t in twins.values() if len(t) == 2)


def modality_dominance_index(rows: Sequence[Mapping]) -> float:
    """P(follows V | conflict) − P(follows A | conflict). rows: kind, pred, follows_A, follows_V. Aligned and null excluded."""
    conflict = [r for r in rows if r["kind"] != "aligned" and r["follows_A"] is not None and r["follows_V"] is not None]
    return _mean(r["pred"] == r["follows_V"] for r in conflict) - _mean(r["pred"] == r["follows_A"] for r in conflict)


def hallucinated_claim_rate(verdicts: Sequence[str]) -> float:
    """Unsupported sentences / all, verdicts pooled over <audio>, <visual>, <fusion>."""
    return _mean(v == "unsupported" for v in verdicts)


def evidence_precision_recall(claimed: set, labels: set) -> tuple[float, float]:
    hits = len(claimed & labels)
    return (hits / len(claimed) if claimed else float("nan")), (hits / len(labels) if labels else float("nan"))


def attribution_accuracy(placed: Sequence[str], true_segment: Sequence[str]) -> float:
    return accuracy(placed, true_segment)


def audit_preference(wins: Sequence[float]) -> float:
    """Win rate of AVRSI; a tie counts 0.5."""
    return _mean(wins)


def reasoning_length(token_counts: Sequence[int]) -> float:
    return _mean(token_counts)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney, ties at half. labels: 1 = planted error, 0 = control."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    observed = accuracy(a, b)
    expected = sum(np.mean([x == c for x in a]) * np.mean([x == c for x in b]) for c in set(a) | set(b))
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def self_preference_rate(wins_for_own_style: Sequence[float]) -> float:
    return _mean(wins_for_own_style)


def order_consistency(forward: Sequence[str | None], backward: Sequence[str | None]) -> float:
    """1 − P(verdict flips). forward has path X as A, backward has it as B; a flip is a different
    preferred path. Parse failures are ties (§3.3)."""
    mirror = {"A": "B", "B": "A"}
    same = [(f if f in mirror else "Tie") == mirror.get(b, "Tie") for f, b in zip(forward, backward)]
    return _mean(same)


def tie_rate(verdicts: Sequence[str | None]) -> float:
    return _mean(v not in ("A", "B") for v in verdicts)


def supported_rate(verdicts: Sequence[str]) -> float:
    return _mean(v == "supported" for v in verdicts)


def fusion_token_share(fusion_sentences: Sequence[str], path_text_tokens: int) -> float:
    fusion = sum(len(normalise_tokens(s)) for s in fusion_sentences)
    return fusion / path_text_tokens if path_text_tokens else 0.0


def reward_audit_correlation(groups: Sequence[tuple[Sequence[float], Sequence[float]]]) -> float:
    """Pearson r of within-group-centred reward and audit score. Offline monitoring only."""
    rewards, audits = [], []
    for r, a in groups:
        rewards += list(np.asarray(r, float) - np.mean(r))
        audits += list(np.asarray(a, float) - np.mean(a))
    if np.std(rewards) == 0 or np.std(audits) == 0:
        return float("nan")
    return float(np.corrcoef(rewards, audits)[0, 1])


@dataclass
class PlantedResult:
    ptype: str  # false_audio, false_visual, modality_swapped, control
    segment: str  # audio or visual
    verdict: str | None  # verdict on the planted sentence; None = parse failure
    score: float | None = None  # AUROC score


def planted_summary(results: Sequence[PlantedResult], segment: str, ptypes: Sequence[str] | None = None) -> dict:
    """One view: A-check is segment audio, V-check visual, J-check either. Parse failures excluded."""
    scoped = [r for r in results if r.segment == segment]
    usable = [r for r in scoped if r.verdict is not None]
    errors = [r for r in usable if r.ptype != "control" and (ptypes is None or r.ptype in ptypes)]
    control = [r for r in usable if r.ptype == "control"]
    detection = _mean(r.verdict in ("unsupported", "unverifiable") for r in errors)
    supported = _mean(r.verdict == "supported" for r in control)
    scored = [r for r in usable if r.score is not None and not np.isnan(r.score)]
    return {
        "detection": detection,
        "supported_on_control": supported,
        "balanced_accuracy": (detection + supported) / 2,
        "parse_failure_rate": 1 - len(usable) / len(scoped) if scoped else float("nan"),
        "auroc": auroc([r.score for r in scored], [r.ptype != "control" for r in scored]) if scored else float("nan"),
    }


def planted_balanced_accuracy(results: Sequence[PlantedResult]) -> float:
    """Guard condition 2: mean over the audio and visual views. No usable result scores 0, so it blocks."""
    values = [planted_summary(results, s)["balanced_accuracy"] for s in ("audio", "visual")]
    return 0.0 if any(np.isnan(v) for v in values) else _mean(values)


def gate0_report(isolated: Sequence[PlantedResult], joint: Sequence[PlantedResult],
                 order_consistency_by_segment: Mapping[str, float], pair_parse_failure_rate: float) -> dict:
    a, v = planted_summary(isolated, "audio"), planted_summary(isolated, "visual")
    j_a, j_v = planted_summary(joint, "audio"), planted_summary(joint, "visual")
    a_false = planted_summary(isolated, "audio", ["false_audio"])["detection"]
    j_false = planted_summary(joint, "audio", ["false_audio"])["detection"]
    joint_parse = _mean([j_a["parse_failure_rate"], j_v["parse_failure_rate"]])
    return {
        "detection": (a["detection"] + v["detection"]) / 2,
        "a_detection": a["detection"], "v_detection": v["detection"],
        "h2_gain": a_false - j_false,
        "a_false_audio": a_false, "j_false_audio": j_false,
        "modality_swapped": {
            "a_check": planted_summary(isolated, "audio", ["modality_swapped"])["detection"],
            "v_check": planted_summary(isolated, "visual", ["modality_swapped"])["detection"],
            "j_check_audio": planted_summary(joint, "audio", ["modality_swapped"])["detection"],
            "j_check_visual": planted_summary(joint, "visual", ["modality_swapped"])["detection"],
        },
        "auroc": {"a_check": a["auroc"], "v_check": v["auroc"], "j_check_audio": j_a["auroc"], "j_check_visual": j_v["auroc"]},
        "order_consistency": _mean(order_consistency_by_segment.values()),
        "supported_on_control": (a["supported_on_control"] + v["supported_on_control"]) / 2,
        "supported_by_view": {"a_check": a["supported_on_control"], "v_check": v["supported_on_control"],
                              "j_check": (j_a["supported_on_control"] + j_v["supported_on_control"]) / 2},
        "parse_failure": {"a_check": a["parse_failure_rate"], "v_check": v["parse_failure_rate"],
                          "j_check": joint_parse, "av_pair": pair_parse_failure_rate},
    }


def gate0_criteria(report: Mapping) -> dict[str, bool]:
    low, high = GATE_SUPPORTED
    return {
        "detection": report["detection"] >= GATE_DETECTION,
        "h2_isolation": report["h2_gain"] >= GATE_H2_MARGIN,
        "order_consistency": report["order_consistency"] >= GATE_ORDER,
        "supported_on_control": low <= report["supported_on_control"] <= high,
        "parse_failure": all(rate <= GATE_PARSE for rate in report["parse_failure"].values()),
    }


def gate0_passed(report: Mapping) -> bool:
    return all(gate0_criteria(report).values())


def gate0_decision(reports: Mapping[str, Mapping], primary: str = "qwen3_omni") -> dict:
    """§9: fewer than two passes stops the project; a failed primary moves the programme to the
    passing backbone with the highest isolated detection."""
    passing = [b for b, r in reports.items() if gate0_passed(r)]
    programme = primary if primary in passing else max(passing, key=lambda b: reports[b]["detection"], default=None)
    return {"passing": passing, "proceed": len(passing) >= 2, "programme_backbone": programme if len(passing) >= 2 else None}
