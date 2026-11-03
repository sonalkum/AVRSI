from dataclasses import replace

import pytest
from conftest import CLASSES, make_disjoint

import random

from avrsi.config import resolve_arm
from avrsi.critic import A_CHECK, J_CHECK, V_CHECK, BackendExecutor, actor_prompt
from avrsi.metrics import PlantedResult, gate0_criteria, gate0_report
from avrsi.models.base import ACTOR, CRITIC, Generation, WithholdingError
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item, jaccard, normalise_tokens
from avrsi.perturb import AUDIO_STYLES, VISUAL_STYLES, build_planted
from avrsi.sync import (PARAPHRASES, Guard, GuardScores, GuardSet, actor_outcome, attach_auroc, evaluate_actor, gate0_run, modality_probes,
                        pair_order_consistency, planted_results, rollout_draws, run_probes, self_preference)

BASE = "<audio>Noise rises. Voices fade. Wind blows.</audio><visual>Shapes move. Lights dim. Colours shift.</visual><fusion>They agree.</fusion><answer>yes</answer>"


def resolve(xm_item):
    audio = "fake://silence" if xm_item.audio_label is None else f"fake://{xm_item.audio_label}"
    return Item(xm_item.item_id, xm_item.question, xm_item.answer, xm_item.options, True, xm_item.kind == "silence",
                video=f"fake://{xm_item.visual_label}", audio=audio, meta=vars(xm_item))


@pytest.fixture(scope="module")
def guard_set(xm):
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    planted = build_planted(carriers, {i.item_id: BASE for i in carriers}, make_disjoint(), random.Random(0), per_type=64)
    return GuardSet([resolve(i) for i in xm["guard"]], planted, resolve=resolve)


def scores(outcome, ba):
    return GuardScores(outcome, ba)


def guard(tmp_path=None, arm="A8"):
    g = Guard(resolve_arm(arm), tmp_path / "guard.jsonl" if tmp_path else None)
    g.set_reference(scores(0.70, 0.80))
    return g


def test_margin_boundaries_inclusive():
    g = guard()
    assert g.decide(200, scores(0.68, 0.78)).handoff
    g = guard()
    assert not g.decide(200, scores(0.6799, 0.80)).handoff
    g = guard()
    d = g.decide(200, scores(0.70, 0.7799))
    assert not d.handoff and d.condition_1 and not d.condition_2


def test_lenient_good_answerer_is_blocked():
    d = guard().decide(200, scores(0.95, 0.55))
    assert not d.handoff and d.condition_1 and not d.condition_2


def test_blocked_keeps_incumbent_handoff_moves_it():
    g = guard()
    g.decide(200, scores(0.60, 0.80))
    assert g.incumbent.outcome == 0.70
    g.decide(400, scores(0.75, 0.85))
    assert g.incumbent.outcome == 0.75
    assert not g.decide(600, scores(0.72, 0.85)).handoff  # measured against the new incumbent


def test_theta0_drift_alarm_catches_slow_slide():
    g = guard()
    outcome, flagged = 0.70, []
    for step in range(1, 5):
        outcome -= 0.015  # each step inside the moving margin
        d = g.decide(step, scores(outcome, 0.80))
        assert d.handoff
        flagged.append(d.drift_alarm)
    assert flagged == [False, False, False, True]


def test_drift_alarm_on_condition_two_and_does_not_block():
    g = guard()
    g.incumbent = scores(0.7, 0.74)
    d = g.decide(1, scores(0.7, 0.74))
    assert d.drift_alarm and d.handoff


def test_block_rate_alarm_strictly_above_25():
    g = guard()
    for step, (o, expect) in enumerate([(0.7, False), (0.7, False), (0.7, False), (0.1, False), (0.1, True)]):
        assert g.decide(step, scores(o, 0.8)).block_alarm is expect
    assert g.block_rate == 0.4


def test_decision_log_written(tmp_path):
    import json
    g = guard(tmp_path)
    g.decide(200, scores(0.7, 0.8))
    rows = [json.loads(line) for line in (tmp_path / "guard.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "theta0_reference" and rows[1]["event"] == "decision" and rows[1]["step"] == 200
    assert rows[1]["handoff"] is True and "block_rate" in rows[1]


def test_a10_minibatch_is_non_strict_outcome_only():
    g = Guard(resolve_arm("A10"))
    d = g.decide_minibatch(10, 0.5, 0.5)
    assert d.handoff and d.condition_2 is None
    assert not g.decide_minibatch(20, 0.49, 0.5).handoff


def test_guard_counts_1792_exact(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = evaluate_actor(backend, guard_set, resolve_arm("A8"), actor_prompt, samplings.greedy, samplings.critic)
    assert (result.generations, result.check_passes, result.pair_passes) == (1024, 256, 512)
    assert result.generations + result.check_passes + result.pair_passes == 1792


def test_guard_judges_with_the_actor_not_the_incumbent(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    evaluate_actor(backend, guard_set, resolve_arm("A8"), actor_prompt, samplings.greedy, samplings.critic)
    assert backend.calls["critic"] == 0 and backend.calls["reference"] == 0


def test_condition_2_view_is_arm_specific(guard_set, samplings):
    for arm_id, wanted in (("A8", {A_CHECK, V_CHECK}), ("A15", {J_CHECK}), ("A2", {J_CHECK})):
        seen = []

        class Spy:
            calls = 0

            def run(self, item, passes):
                seen.extend(p.view for p in passes)
                return ["1: supported"] * len(passes)

        planted_results(Spy(), guard_set.planted[:256:7], resolve_arm(arm_id).guard_view, resolve)
        assert set(seen) == wanted


@pytest.mark.parametrize("arm_id,views", [("A8", {("audio",), ("video",)}), ("A15", {("audio", "video")})])
def test_guard_checks_in_the_arms_view(arm_id, views, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    evaluate_actor(backend, guard_set, resolve_arm(arm_id), actor_prompt, samplings.greedy, samplings.critic)
    assert {x.modalities for _, x in backend.seen_inputs if x.prompt.startswith("You are checking")} == views


def test_isolated_check_withholds_in_guard(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    planted_results(BackendExecutor(backend, samplings.critic, ACTOR), guard_set.planted, "isolated", resolve)
    assert all(bool(x.audio) != bool(x.video) for _, x in backend.seen_inputs)


def test_honest_vs_lenient_planted_scores(guard_set, samplings):
    arm = resolve_arm("A8")
    honest = FakeBackend(CLASSES)
    lenient = FakeBackend(CLASSES)
    lenient.leniency[ACTOR] = 1.0
    a = evaluate_actor(honest, guard_set, arm, actor_prompt, samplings.greedy, samplings.critic)
    b = evaluate_actor(lenient, guard_set, arm, actor_prompt, samplings.greedy, samplings.critic)
    assert a.planted_ba == 1.0 and b.planted_ba == 0.5


def test_isolated_check_catches_modality_swap(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    executor = BackendExecutor(backend, samplings.critic, ACTOR)
    swapped = [p for p in guard_set.planted if p.ptype == "modality_swapped"]
    isolated = planted_results(executor, swapped, "isolated", resolve)
    assert all(r.verdict in ("unsupported", "unverifiable") for r in isolated)


def test_parse_failures_excluded_not_counted_against(guard_set):
    class Bad:
        def run(self, item, passes):
            return ["garbage"]

    results = planted_results(Bad(), guard_set.planted[:3], "isolated", resolve)
    assert all(r.verdict is None for r in results)


def test_order_consistency_slot_biased_critic(guard_set):
    class SlotA:
        def run(self, item, passes):
            return ["A"] * len(passes)

    overall, by_segment = pair_order_consistency(SlotA(), guard_set.planted[:8], resolve)
    assert overall == 0.0 and set(by_segment) <= {"audio", "visual"}


def test_outcome_is_strict_unless_reference_fallback(samplings):
    class Plain(FakeBackend):
        def generate(self, inputs, sampling, role="actor"):
            from avrsi.models.base import Generation
            return [Generation("yes") for _ in inputs]

    items = [Item("a", "q", "yes", video="v")]
    assert actor_outcome(Plain(), items, actor_prompt, samplings.greedy)[0] == 0.0
    assert actor_outcome(Plain(), items, actor_prompt, samplings.greedy, use_fallback=True)[0] == 1.0


def test_reference_uses_fallback_only_on_flag(guard_set, samplings):
    class Plain(FakeBackend):
        def _rollout(self, role, sampling):
            from avrsi.models.base import Generation
            return Generation("I would say no")

    arm = resolve_arm("A8")
    flagged = replace(guard_set, use_fallback_reference=True)
    strict = evaluate_actor(Plain(CLASSES), flagged, arm, actor_prompt, samplings.greedy, samplings.critic)
    ref = evaluate_actor(Plain(CLASSES), flagged, arm, actor_prompt, samplings.greedy, samplings.critic, reference=True)
    unflagged = evaluate_actor(Plain(CLASSES), guard_set, arm, actor_prompt, samplings.greedy, samplings.critic, reference=True)
    assert strict.outcome == 0.0 and ref.outcome > 0.2 and unflagged.outcome == 0.0


def test_gate0_end_to_end_on_fake(xm, samplings):
    backend = FakeBackend(CLASSES)
    bases = {i.item_id: BASE for i in xm["gate"]}
    planted = build_planted(xm["gate"], bases, make_disjoint(), random.Random(0), per_type=125)
    executor = BackendExecutor(backend, samplings.critic, "reference")
    run = gate0_run(executor, executor, planted, resolve)
    assert run["pair_passes"] == 1000
    report = gate0_report(run["isolated"], run["joint"], run["order_consistency"], run["pair_parse_failure_rate"])
    assert report["detection"] == 1.0 and report["a_detection"] == 1.0
    assert report["supported_on_control"] == 1.0  # saturated: this fake must fail the 0.95 ceiling
    assert report["order_consistency"] == 1.0
    assert report["h2_gain"] == 0.0  # the fake's J-check sees truth too; isolation has no edge here
    criteria = gate0_criteria(report)
    assert criteria["detection"] and not criteria["supported_on_control"] and not criteria["h2_isolation"]


def test_auroc_sampled_path(guard_set, samplings):
    planted = guard_set.planted[:12]
    results = [PlantedResult(p.ptype, p.segment, "unsupported") for p in planted]
    backend = FakeBackend(CLASSES)
    attach_auroc(backend, samplings.critic, samplings.rollout, results, planted, "isolated", "reference", resolve=resolve)
    assert all(r.score is not None for r in results)
    assert backend.calls["reference"] == 4 * 12


def test_probes_on_fake(guard_set, samplings):
    executor = BackendExecutor(FakeBackend(CLASSES), samplings.critic, "reference")
    probes = run_probes(executor, guard_set.planted, resolve)
    assert probes["vacuous_returned"] == 1.0
    assert set(probes) == {"vacuous_returned", "fusion_laundering_rewarded", "question_leak_passed", "paraphrase_raised_s_m", "paraphrase_usable"}
    assert probes["paraphrase_usable"] >= 1 and probes["paraphrase_raised_s_m"] == 1.0


def test_paraphrases_survive_dedupe_against_every_style():
    for segment, styles in (("audio", AUDIO_STYLES), ("visual", VISUAL_STYLES)):
        for template in PARAPHRASES[segment]:
            for style in styles:
                a, b = normalise_tokens(template.format(c="dog")), normalise_tokens(style.format(c="dog"))
                assert jaccard(a, b) < 0.8


class Recording:
    """Answers every check "unverifiable" and every pair with `reply`, keeping what it was asked."""

    def __init__(self, reply="Tie"):
        self.reply, self.checks, self.pairs = reply, [], []

    def run(self, item, passes):
        if passes[0].view == "av_pair":
            self.pairs.append(passes)
            return [self.reply(p) if callable(self.reply) else self.reply for p in passes]
        self.checks.append((item, passes[0]))
        return ["\n".join(f"{i + 1}: unverifiable" for i in range(passes[0].n_sentences))]


def test_leak_probe_only_audio_items_answered_no(guard_set):
    executor = Recording()
    run_probes(executor, guard_set.planted, resolve)
    asked = [(item, p) for item, p in executor.checks if "?." not in p.prompt and item.question.rstrip("?") + "." in p.prompt]
    assert asked and all(p.segment == "audio" and item.answer == "no" for item, p in asked)


def test_laundering_probe_runs_both_orders(guard_set):
    def prefers_moved(p):
        first = p.prompt.split("Reasoning B:")[0]
        return "A" if "The clip also contains" in first else "B"

    executor = Recording(prefers_moved)
    assert run_probes(executor, guard_set.planted, resolve)["fusion_laundering_rewarded"] == 1.0
    assert all([p.pair for p in passes] == [(0, 1), (1, 0)] for passes in executor.pairs)
    slot_biased = Recording("A")
    assert run_probes(slot_biased, guard_set.planted, resolve)["fusion_laundering_rewarded"] == 0.5


def test_order_consistency_honest_critic_is_one(guard_set):
    class Honest:
        def __init__(self, sentence):
            self.sentence = sentence

        def run(self, item, passes):
            return ["B" if self.sentence in p.prompt.split("Reasoning B:")[0] else "A" for p in passes]

    for p in guard_set.planted[:12]:
        assert pair_order_consistency(Honest(p.sentence), [p], resolve)[0] == 1.0


def test_gate0_order_consistency_honest_critic_is_one(guard_set):
    class Honest:
        calls = 0

        def __init__(self, sentence):
            self.sentence = sentence

        def run(self, item, passes):
            return ["B" if self.sentence in p.prompt.split("Reasoning B:")[0] else "A" for p in passes]

    class Plain:
        def run(self, item, passes):
            return ["1: supported"] * len(passes)

    for p in guard_set.planted[:12]:
        run = gate0_run(Plain(), Honest(p.sentence), [p], resolve)
        assert run["order_consistency"][p.segment] == 1.0


@pytest.mark.parametrize("mode,both", [("isolated", False), ("joint", True)])
def test_auroc_respects_the_view(mode, both, guard_set, samplings):
    planted = guard_set.planted[:8]
    results = [PlantedResult(p.ptype, p.segment, "unsupported") for p in planted]
    backend = FakeBackend(CLASSES)
    attach_auroc(backend, samplings.critic, samplings.rollout, results, planted, mode, "reference", resolve=resolve)
    assert all((bool(x.audio) and bool(x.video)) is both for _, x in backend.seen_inputs)


def test_supported_alarm_fires_above_095():
    g = guard()
    assert not g.decide(1, GuardScores(0.7, 0.8, supported_on_control=0.95)).supported_alarm
    assert g.decide(2, GuardScores(0.7, 0.8, supported_on_control=0.951)).supported_alarm
    assert not g.decide(3, scores(0.7, 0.8)).supported_alarm


def test_guard_reports_supported_rate_on_controls(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    backend.leniency[ACTOR] = 1.0
    result = evaluate_actor(backend, guard_set, resolve_arm("A8"), actor_prompt, samplings.greedy, samplings.critic)
    assert result.supported_on_control == 1.0
    assert guard().decide(1, result).supported_alarm


def test_self_preference_prefers_own_valid_path(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    rate = self_preference(backend, guard_set, samplings.greedy, samplings.critic, n=8)
    assert 0 <= rate <= 1

    class Malformed(FakeBackend):
        def _rollout(self, role, sampling):
            return Generation("not a path")

    assert self_preference(Malformed(CLASSES), guard_set, samplings.greedy, samplings.critic, n=8) == 0.0

    class OwnStyle(FakeBackend):
        def _rollout(self, role, sampling):
            return Generation(BASE.replace("Noise rises", "Mine says"))

        def _pair(self, inputs):
            return "A" if "Mine says" in inputs.prompt.split("Reasoning B:")[0] else "B"

    assert self_preference(OwnStyle(CLASSES), guard_set, samplings.greedy, samplings.critic, n=8) == 1.0


def test_self_preference_generates_and_judges_as_the_critic(guard_set, samplings):
    backend = FakeBackend(CLASSES)
    self_preference(backend, guard_set, samplings.greedy, samplings.critic, n=8)
    generated = [x for _, x in backend.seen_inputs if not x.prompt.startswith("Two reasoning")]
    assert len(generated) == 8 and {role for role, _ in backend.seen_inputs} == {CRITIC}


def test_t8_probes_catch_a_stream_that_never_arrives(guard_set, samplings):
    items = [i for i in guard_set.items if i.meta["audio_label"]]
    report = modality_probes(FakeBackend(CLASSES), items, samplings.critic, n=4)
    assert report["audio_changes_output"] > 0 and report["visual_changes_output"] > 0

    class Deaf(FakeBackend):
        def generate(self, inputs, sampling, role="actor"):
            return [Generation("1: unverifiable") for _ in inputs]

    with pytest.raises(WithholdingError, match="audio"):
        modality_probes(Deaf(CLASSES), items, samplings.critic, n=4)


def test_rollout_draws_use_theta0(samplings):
    backend = FakeBackend()
    draws = rollout_draws(backend, [Item("a", "q", "yes", video="fake://dog", audio="fake://dog")], samplings.rollout, 3)
    assert len(draws["a"]) == 3 and backend.calls["reference"] == 3 and backend.calls["actor"] == 0


