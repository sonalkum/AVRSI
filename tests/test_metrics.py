import math

import numpy as np
import pytest

from avrsi.metrics import (PlantedResult, accuracy, attribution_accuracy, audit_preference, auroc, avhbench_score, balanced_accuracy, cmm_score,
                           cohen_kappa, evidence_precision_recall, f1, fusion_token_share, gate0_criteria, gate0_decision, gate0_passed,
                           gate0_report, hallucinated_claim_rate, modality_dominance_index, order_consistency, pa_hr, planted_balanced_accuracy,
                           planted_summary, reasoning_length, reward_audit_correlation, self_preference_rate, supported_rate, swap_sensitivity,
                           tie_rate, xm_balanced_accuracy, xrs, yes_rate)

G = ["yes"] * 8 + ["no"] * 2


def test_always_no_cannot_win_balanced_accuracy():
    preds = ["no"] * 10
    assert accuracy(preds, ["no"] * 8 + ["yes"] * 2) == 0.8
    assert balanced_accuracy(preds, ["no"] * 8 + ["yes"] * 2) == 0.5


def test_f1_and_yes_rate():
    preds = ["yes"] * 4 + ["no"] * 4 + ["yes", "no"]
    assert f1(preds, G) == pytest.approx(2 * 4 / (2 * 4 + 1 + 4))
    assert yes_rate(preds) == 0.5
    assert f1(["no"] * 10, G) == 0


def test_cmm_pa_hr():
    golds = ["yes", "yes", "no", "no"]
    assert pa_hr(["yes", "no", "no", "no"], golds) == (0.5, 1.0)
    assert cmm_score(["yes", "no", "no", "no"], golds) == 0.75
    assert cmm_score(["yes"] * 4, golds) == 0.5  # always-yes gets chance


def test_avhbench_mean_of_a2v_v2a_only():
    rows = [
        {"task": "a2v", "pred": "yes", "gold": "yes"}, {"task": "a2v", "pred": "yes", "gold": "no"},
        {"task": "v2a", "pred": "no", "gold": "no"}, {"task": "v2a", "pred": "yes", "gold": "yes"},
        {"task": "matching", "pred": "no", "gold": "yes"},
    ]
    assert avhbench_score(rows) == 0.75
    with pytest.raises(ValueError):
        avhbench_score(rows[:2])


def test_xm_balanced_accuracy_is_mean_over_families_of_class_recall():
    rows = [
        {"family": "F1", "answer_class": "yes", "correct": True},
        {"family": "F1", "answer_class": "yes", "correct": True},
        {"family": "F1", "answer_class": "no", "correct": False},
        {"family": "F4", "answer_class": "both", "correct": True},
        {"family": "F4", "answer_class": "neither", "correct": True},
    ]
    assert xm_balanced_accuracy(rows) == 0.75  # F1 = 0.5, F4 = 1


def test_xrs_unweighted():
    assert xrs(0.3, 0.6, 0.9) == pytest.approx(0.6)


def test_swap_sensitivity_penalises_invariance():
    rows = [
        {"twin_id": "a", "kind": "aligned", "correct": True}, {"twin_id": "a", "kind": "audio_swap", "correct": False},
        {"twin_id": "b", "kind": "aligned", "correct": True}, {"twin_id": "b", "kind": "video_swap", "correct": True},
        {"twin_id": "c", "kind": "aligned", "correct": True},
    ]
    assert swap_sensitivity(rows) == 0.5


def test_mdi_visual_dominated_and_exclusions():
    rows = [
        {"kind": "audio_swap", "pred": "yes", "follows_A": "no", "follows_V": "yes"},
        {"kind": "video_swap", "pred": "no", "follows_A": "yes", "follows_V": "no"},
        {"kind": "audio_swap", "pred": "yes", "follows_A": None, "follows_V": None},  # F3
        {"kind": "aligned", "pred": "yes", "follows_A": "no", "follows_V": "no"},
    ]
    assert modality_dominance_index(rows) == 1.0
    balanced = [{"kind": "audio_swap", "pred": "yes", "follows_A": "yes", "follows_V": "no"},
                {"kind": "audio_swap", "pred": "no", "follows_A": "yes", "follows_V": "no"}]
    assert modality_dominance_index(balanced) == 0.0


def test_reasoning_metrics():
    assert hallucinated_claim_rate(["unsupported", "supported", "vacuous", "unsupported"]) == 0.5
    assert evidence_precision_recall({"a", "b"}, {"b", "c", "d"}) == (0.5, pytest.approx(1 / 3))
    assert attribution_accuracy(["audio", "visual"], ["audio", "audio"]) == 0.5
    assert audit_preference([1, 0, 0.5, 0.5]) == 0.5
    assert reasoning_length([10, 20]) == 15


def test_auroc():
    assert auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert auroc([0, 1], [1, 0]) == 0.0
    assert auroc([1, 1], [1, 0]) == 0.5
    assert math.isnan(auroc([1], [1]))


def test_kappa():
    a = ["x", "x", "y", "y"]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, ["x", "y", "x", "y"]) == 0.0
    assert cohen_kappa(a, ["y", "y", "x", "x"]) == -1.0


def test_order_consistency():
    assert order_consistency(["A", "B", "Tie"], ["B", "A", "Tie"]) == 1.0
    assert order_consistency(["A", "A"], ["A", "A"]) == 0.0  # slot bias flips the preferred path
    assert order_consistency(["A", None], ["Tie", "Tie"]) == 0.5


def test_health_rates():
    assert tie_rate(["A", "Tie", None, "B"]) == 0.5
    assert supported_rate(["supported", "vacuous"]) == 0.5
    assert fusion_token_share(["one two"], 8) == 0.25
    assert self_preference_rate([1, 0]) == 0.5


def test_reward_audit_correlation_is_within_group():
    # Pooled, these would correlate strongly negatively; within groups the relation is perfect.
    groups = [([1, 2], [10, 11]), ([100, 101], [-5, -4])]
    assert reward_audit_correlation(groups) == pytest.approx(1.0)
    assert math.isnan(reward_audit_correlation([([1, 1], [2, 3])]))


def results(detected, supported, n=10, fails=0):
    rows = []
    for segment, ptype in (("audio", "false_audio"), ("visual", "false_visual")):
        rows += [PlantedResult(ptype, segment, "unsupported" if i < detected * n else "supported") for i in range(n)]
        rows += [PlantedResult("control", segment, "supported" if i < supported * n else "unsupported") for i in range(n)]
        rows += [PlantedResult(ptype, segment, None) for _ in range(fails)]
    return rows


def test_planted_summary_and_parse_exclusion():
    summary = planted_summary(results(0.8, 0.9, fails=20), "audio")
    assert summary["detection"] == 0.8 and summary["supported_on_control"] == 0.9
    assert summary["balanced_accuracy"] == pytest.approx(0.85)
    assert summary["parse_failure_rate"] == 0.5


def test_unverifiable_counts_as_detected_vacuous_does_not():
    rows = [PlantedResult("false_audio", "audio", "unverifiable"), PlantedResult("false_audio", "audio", "vacuous")]
    assert planted_summary(rows, "audio")["detection"] == 0.5


def test_lenient_judge_scores_half():
    assert planted_balanced_accuracy(results(0.0, 1.0)) == 0.5


def test_abstaining_judge_scores_half():
    rows = [PlantedResult("false_audio", s, "unverifiable") for s in ("audio", "visual")]
    rows += [PlantedResult("control", s, "unverifiable") for s in ("audio", "visual")]
    assert planted_balanced_accuracy(rows) == 0.5


def test_all_parse_failures_cannot_pass_guard():
    assert planted_balanced_accuracy([PlantedResult("false_audio", "audio", None), PlantedResult("control", "visual", None)]) == 0.0


def good_report(**overrides):
    report = gate0_report(results(0.9, 0.8), results(0.7, 0.8), {"audio": 0.9, "visual": 0.9}, 0.0)
    return {**report, **overrides}


def test_gate0_report_values():
    report = good_report()
    assert report["detection"] == pytest.approx(0.9) and report["h2_gain"] == pytest.approx(0.2)
    assert report["order_consistency"] == pytest.approx(0.9) and report["supported_on_control"] == 0.8
    assert gate0_passed(report)


@pytest.mark.parametrize("overrides,key", [
    ({"detection": 0.69}, "detection"),
    ({"h2_gain": 0.049}, "h2_isolation"),
    ({"order_consistency": 0.84}, "order_consistency"),
    ({"supported_on_control": 0.96}, "supported_on_control"),
    ({"supported_on_control": 0.59}, "supported_on_control"),
    ({"parse_failure": {"a_check": 0.11, "v_check": 0, "j_check": 0, "av_pair": 0}}, "parse_failure"),
    ({"parse_failure": {"a_check": 0, "v_check": 0, "j_check": 0, "av_pair": 0.2}}, "parse_failure"),
])
def test_gate0_each_criterion_can_fail(overrides, key):
    report = good_report(**overrides)
    criteria = gate0_criteria(report)
    assert not criteria[key] and not gate0_passed(report)
    assert sum(criteria.values()) == 4


def test_gate0_boundaries_pass():
    assert gate0_passed(good_report(detection=0.70, h2_gain=0.05, order_consistency=0.85, supported_on_control=0.95))


def test_good_joint_critic_fails_gate():
    assert not gate0_passed(gate0_report(results(0.9, 0.8), results(0.9, 0.8), {"audio": 0.9}, 0.0))


def test_h2_uses_false_audio_items_only():
    isolated = results(0.5, 0.8) + [PlantedResult("modality_swapped", "audio", "unsupported")] * 100
    joint = results(0.5, 0.8)
    assert gate0_report(isolated, joint, {"audio": 1}, 0)["h2_gain"] == 0


def test_stop_rule():
    good, bad = good_report(), good_report(detection=0.1)
    assert not gate0_decision({"qwen3_omni": good, "avf": bad, "qwen2_5_omni": bad})["proceed"]
    assert gate0_decision({"qwen3_omni": good, "avf": good})["programme_backbone"] == "qwen3_omni"


def test_primary_failure_moves_programme_to_best_detection():
    decision = gate0_decision({"qwen3_omni": good_report(detection=0.1), "avf": good_report(detection=0.8), "qwen2_5_omni": good_report(detection=0.9)})
    assert decision["proceed"] and decision["programme_backbone"] == "qwen2_5_omni"


def test_numpy_scores_ok():
    assert auroc(np.array([1.0, 0.0]), np.array([1, 0])) == 1.0
