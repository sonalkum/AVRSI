import random
from dataclasses import replace

import pytest
import torch

from avrsi.config import resolve_arm
from avrsi.critic import (A_CHECK, ACTOR_PROMPT, AV_PAIR, J_CHECK, V_CHECK, BackendExecutor, CriticPass, actor_prompt, check_prompt,
                          guard_check_volume, guard_run_volume, max_passes, pair_prompt, parse_check, parse_pair, plan_passes,
                          prompt_digest, run_volume, score_group, unsupported_score)
from avrsi.models.base import Sampling
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item, ParsedPath, parse_path

FROZEN = "660b1eee105aacf769d4d11ec92a0b1361d5612091bcbb4c3397c359c64621df"


def paths(n, valid=True):
    return [ParsedPath("", valid, ["a.", "b."], ["c."], [], "yes") if valid else ParsedPath("", False) for _ in range(n)]


def views(passes):
    return [p.view for p in passes]


def test_prompts_frozen():
    assert prompt_digest() == FROZEN


@pytest.mark.parametrize("arm,count", [("A2", 6), ("A8", 14), ("A15", 14), ("A9", 20), ("A11", 44), ("A6", 8), ("A3", 14), ("A4", 14),
                                       ("A5", 14), ("A10", 14), ("A12", 14), ("A1", 0), ("A0", 0), ("A7", 0)])
def test_pass_counts(arm, count):
    assert max_passes(resolve_arm(arm)) == count


def test_a8_exact_passes(a8, yes_item, rng):
    plan = plan_passes(a8, paths(4), yes_item, rng)
    assert views(plan).count(AV_PAIR) == 6 and views(plan).count(A_CHECK) == 4 and views(plan).count(V_CHECK) == 4


def test_a15_uses_only_j_check(yes_item, rng):
    plan = plan_passes(resolve_arm("A15"), paths(4), yes_item, rng)
    assert views(plan).count(J_CHECK) == 8 and A_CHECK not in views(plan) and V_CHECK not in views(plan)


def test_a3_is_joint_a4_is_isolated(yes_item, rng):
    assert J_CHECK in views(plan_passes(resolve_arm("A3"), paths(4), yes_item, rng))
    assert A_CHECK in views(plan_passes(resolve_arm("A4"), paths(4), yes_item, rng))


def test_a9_runs_both_orders_of_every_pair(yes_item, rng):
    pairs = [p.pair for p in plan_passes(resolve_arm("A9"), paths(4), yes_item, rng) if p.view == AV_PAIR]
    assert sorted(pairs) == [(j, k) for j in range(4) for k in range(4) if j != k]


def test_single_order_is_randomised(a8, yes_item):
    rng = random.Random(1)
    orders = {p.pair for _ in range(20) for p in plan_passes(a8, paths(2), yes_item, rng) if p.view == AV_PAIR}
    assert orders == {(0, 1), (1, 0)}


def test_malformed_paths_reduce_passes(a8, yes_item, rng):
    mix = paths(3) + paths(1, valid=False)
    assert len(plan_passes(a8, mix, yes_item, rng)) == 3 + 6
    assert len(plan_passes(a8, paths(4, valid=False), yes_item, rng)) == 0


def test_empty_segment_gets_no_pass(a8, yes_item, rng):
    p = paths(2)
    p[0].visual = []
    assert views(plan_passes(a8, p, yes_item, rng)).count(V_CHECK) == 1


@pytest.mark.parametrize("arm,volume", [("A8", 716_800), ("A11", 2_252_800), ("A3", 716_800), ("A4", 716_800)])
def test_run_volume(arm, volume):
    assert run_volume(resolve_arm(arm)) == volume


def test_a3_short_volume():
    assert run_volume(resolve_arm("A3"), steps=1600) == 179_200


def test_guard_volumes(a8):
    assert guard_check_volume() == 1792
    assert guard_run_volume(200, 6400) == 59_136
    assert guard_run_volume(200, 6400) + run_volume(a8) == 775_936


def test_view_modalities(a8, yes_item, rng):
    by_view = {p.view: p.modalities for p in plan_passes(a8, paths(4), yes_item, rng)}
    assert by_view[A_CHECK] == ("audio",) and by_view[V_CHECK] == ("video",) and by_view[AV_PAIR] == ("audio", "video")
    assert CriticPass(J_CHECK, "").modalities == ("audio", "video")


def test_j_check_prompt_identical_to_isolated_prompt(a8, yes_item, rng):
    p = paths(4)
    iso = [x for x in plan_passes(a8, p, yes_item, random.Random(0)) if x.view != AV_PAIR]
    joint = [x for x in plan_passes(resolve_arm("A15"), p, yes_item, random.Random(0)) if x.view != AV_PAIR]
    assert [x.prompt for x in iso] == [x.prompt for x in joint]


def test_pair_prompt_is_anonymised(yes_item):
    text = pair_prompt(yes_item, paths(1)[0], paths(1)[0])
    assert "Reasoning A" in text and "A8" not in text and "policy" not in text.lower()


def test_actor_prompt_states_format(yes_item):
    text = actor_prompt(yes_item)
    assert all(t in text for t in ("<audio>", "<visual>", "<fusion>", "<answer>")) and yes_item.question in text
    assert "Example" not in text and "Example" in actor_prompt(yes_item, exemplar=True)
    assert ACTOR_PROMPT.startswith("Answer")


def test_check_prompt_numbers_sentences(yes_item):
    assert "1. a.\n2. b." in check_prompt(yes_item, "audio", ["a.", "b."])


@pytest.mark.parametrize("text,n,expected", [
    ("1: supported\n2: vacuous", 2, ["supported", "vacuous"]),
    (" 1. UNSUPPORTED \n2) unverifiable.", 2, ["unsupported", "unverifiable"]),
    ("Here you go:\n1: supported", 1, ["supported"]),
    ("1: supported", 2, None),                     # count mismatch
    ("1: supported\n2: supported\n3: vacuous", 2, None),
    ("1: supported\n1: supported", 2, None),       # duplicate number
    ("2: supported\n1: supported", 2, None),       # order
    ("1: maybe\n2: supported", 2, None),           # unknown verdict
    ("", 1, None),
])
def test_parse_check(text, n, expected):
    assert parse_check(text, n) == expected


@pytest.mark.parametrize("text,expected", [("A", "A"), (" b. ", "B"), ("TIE", "Tie"), ("A is better", None), ("", None), ("C", None)])
def test_parse_pair(text, expected):
    assert parse_pair(text) == expected


class Scripted:
    def __init__(self, reply):
        self.reply = reply

    def run(self, item, passes):
        return [self.reply(p) for p in passes]


def test_check_failure_unverifiable_in_reward(a8, yes_item, rng):
    result = score_group(a8, Scripted(lambda p: "nonsense"), yes_item, paths(2), rng)
    assert result.audio[0] == ["unverifiable", "unverifiable"]
    assert result.check_failures == 4 and result.pair_failures == 1


def test_score_group_records_shown_order(a8, yes_item):
    result = score_group(a8, Scripted(lambda p: "A" if p.view == AV_PAIR else "1: supported\n2: supported"), yes_item, paths(2), random.Random(3))
    (pair, verdict), = result.pair_outcomes.items()
    assert verdict == "A" and set(pair) == {0, 1}



def test_executor_withholds_per_view(a8, rng):
    item = Item("c", "q here", "yes", video="fake://piano", audio="fake://dog")
    backend = FakeBackend()
    executor = BackendExecutor(backend, replace(_greedy(), max_new_tokens=8))
    parsed = [parse_path("<audio>The sound of dog can be heard.</audio><visual>The video shows piano.</visual><fusion></fusion><answer>yes</answer>", item)] * 2
    score_group(a8, executor, item, parsed, rng)
    seen = {x.prompt.split("against what is ")[1][:5] if "against" in x.prompt else "pair": x for _, x in backend.seen_inputs}
    assert seen["heard"].video is None and seen["heard"].audio == "fake://dog"
    assert seen["seen."].audio is None and seen["seen."].video == "fake://piano"
    assert seen["pair"].audio and seen["pair"].video
    assert all(role == "critic" for role, _ in backend.seen_inputs)


def _greedy():
    return Sampling(0.0, 1.0, False, 64)


class LogProbBackend(FakeBackend):
    supports_logprob_scoring = True

    def encode(self, text):
        return list(range(len(text.split())))

    def score_logprobs(self, inputs, completions, role="actor"):
        self.seen = completions[0].text
        return [torch.tensor([-1.0, -2.0, -3.0, -4.0])]


def test_auroc_score_prefers_logprob():
    yes_item = Item("c", "q", "yes", video="fake://piano", audio="fake://dog")
    check = CriticPass(A_CHECK, "p", segment="audio", n_sentences=2)
    backend = LogProbBackend()
    value = unsupported_score(backend, yes_item, check, 1, ["supported", "vacuous"], None)
    assert value == -4.0
    assert backend.seen == "1: supported\n2: unsupported"


def test_auroc_score_sampled_fallback(yes_item):
    check = CriticPass(A_CHECK, "p", segment="audio", n_sentences=1)
    outputs = ["1: unsupported", "1: supported", "bad", "1: unsupported"]
    assert unsupported_score(FakeBackend(), yes_item, check, 0, None, outputs) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        unsupported_score(FakeBackend(), yes_item, check, 0, None, None)


