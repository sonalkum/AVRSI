import random
from dataclasses import replace

import pytest

from avrsi.config import resolve_arm
from avrsi.models.base import Generation
from avrsi.parse import Item, parse_path
from avrsi.rewards import r_total_rl_comm
from avrsi.rlcomm import I_TASK, S_TASK, RLCommScorer, r_arr, r_avc
from avrsi.train import group_rewards

ITEM = Item("i", "Is there a dog?", "yes", is_xm=True)
TEXT = "<audio>A dog barks.</audio><visual>A dog runs.</visual><fusion></fusion><answer>{}</answer>"


def coherence(scores):
    return lambda task, query, content: scores[task]


def test_r_arr_needs_correct_answer_and_similarity_over_omega():
    assert r_arr("a", "r", True, coherence({S_TASK: 0.81})) == 1.0
    assert r_arr("a", "r", True, coherence({S_TASK: 0.8})) == 0.0  # strict, eq 3
    assert r_arr("a", "r", False, coherence({S_TASK: 0.99})) == 0.0


def test_r_avc_cases_of_eq_4():
    score = coherence({I_TASK: 0.3})
    assert r_avc("a", "v", True, True, score) == pytest.approx(1.3)
    assert r_avc("a", "v", True, False, score) == pytest.approx(0.3)
    assert r_avc("a", "v", False, False, score) == 0.0


def scorer(scores, references):
    return RLCommScorer(lambda item: references.append(item.item_id) or "reference", coherence(scores))


def test_scorer_on_paths():
    refs = []
    right, wrong, broken = parse_path(TEXT.format("yes"), ITEM), parse_path(TEXT.format("no"), ITEM), parse_path("junk", ITEM)
    result = scorer({S_TASK: 0.9, I_TASK: 0.5}, refs)(ITEM, [right, wrong, broken])
    assert result == [(1.0, 1.5), (0.0, 0.5), (0.0, 0.0)]


def test_reference_runs_once_per_item():
    refs = []
    s = scorer({S_TASK: 0.9, I_TASK: 0.5}, refs)
    path = parse_path(TEXT.format("yes"), ITEM)
    s(ITEM, [path]), s(ITEM, [path])
    assert refs == ["i"]


def test_coherence_gets_the_audio_and_visual_segments():
    calls = []

    def spy(task, query, content):
        calls.append((task, query, content))
        return 0.5

    RLCommScorer(lambda item: "reference", spy)(ITEM, [parse_path(TEXT.format("yes"), ITEM)])
    assert calls == [(S_TASK, "A dog barks.", "reference"), (I_TASK, "A dog barks.", "A dog runs.")]


def test_a7_total_counts_format_once():
    arm = replace(resolve_arm("A7"), lam=1.0)
    gens = [Generation(TEXT.format("yes")), Generation("junk")]
    totals, details = group_rewards(arm, ITEM, gens, None, random.Random(0), scorer({S_TASK: 0.9, I_TASK: 0.5}, []))
    assert totals == [r_total_rl_comm(1.0, 1.0, 1.0, 1.5, 1.0), 0.0] and details["passes"] == 0
