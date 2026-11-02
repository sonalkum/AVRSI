import pytest

from avrsi.critic import BackendExecutor, score_group, verdicts_for_reward
from avrsi.models.base import Sampling
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item, parse_path
from avrsi.rewards import (group_advantages, pairwise_r, r_outcome, r_pair, r_total, r_total_rl_comm, r_xm, s_m)

BASE = "<audio>{a}</audio><visual>{v}</visual><fusion>{f}</fusion><answer>{x}</answer>"


def path(a="A dog.", v="A man.", f="", x="yes"):
    return BASE.format(a=a, v=v, f=f, x=x)


S, U, W, Z = "supported", "unsupported", "unverifiable", "vacuous"


def test_r_outcome_requires_well_formed(yes_item):
    assert r_outcome(parse_path(path(x="Yes."), yes_item), yes_item) == 1
    assert r_outcome(parse_path(path(x="No"), yes_item), yes_item) == 0
    assert r_outcome(parse_path("<answer>yes</answer>", yes_item), yes_item) == 0  # never the fallback


def test_r_outcome_yes_with_extra_words_is_not_normalised_away(yes_item):
    assert r_outcome(parse_path(path(x="Yes, it is"), yes_item), yes_item) == 0


def test_r_outcome_mcq_letter(mcq_item):
    assert r_outcome(parse_path(path(x="B"), mcq_item), mcq_item) == 1
    assert r_outcome(parse_path(path(x="A"), mcq_item), mcq_item) == 0


@pytest.mark.parametrize("verdicts,n,expected", [
    ([], 3, 0.0),
    ([S], 3, 1 / 3),
    ([S, S], 3, 2 / 3),
    ([S, S, S], 3, 1.0),
    ([S] * 6, 3, 1.0),
    ([S, S, S, U, U, U], 3, 0.5),
    ([S, U], 3, 1 / 3 * 1 / 2),
    ([U, W, Z], 3, 0.0),
    ([S], 1, 1.0),
    ([S, U], 1, 0.5),
])
def test_s_m(verdicts, n, expected):
    assert s_m(verdicts, n) == pytest.approx(expected)


def test_r_xm_silence_uses_n1_for_audio_only():
    assert r_xm([S], [S, S, S], silence=True) == pytest.approx(1.0)
    assert r_xm([S], [S, S, S], silence=False) == pytest.approx(0.5 * (1 / 3 + 1))
    assert r_xm([S, S, S], [S], silence=True) == pytest.approx(0.5 * (1 + 1 / 3))  # S_V keeps N=3


def test_check_failure_makes_segment_unverifiable():
    assert verdicts_for_reward(None, 3) == [W, W, W]
    assert s_m(verdicts_for_reward(None, 3)) == 0.0


def test_pair_symmetry_and_means():
    r = pairwise_r([True, True, True], {(0, 1): "A", (2, 0): "A", (1, 2): "Tie"})
    assert r[0][1] == 1 and r[1][0] == 0
    assert r[2][0] == 1 and r[0][2] == 0
    assert r[1][2] == r[2][1] == 0.5
    for j in range(3):
        for k in range(3):
            if j != k:
                assert r[j][k] == 1 - r[k][j]
    assert r_pair(r) == [0.5, 0.25, 0.75]


def test_pair_order_is_mapped_back_to_paths():
    assert pairwise_r([True, True], {(1, 0): "A"})[1][0] == 1
    assert pairwise_r([True, True], {(1, 0): "B"})[0][1] == 1


def test_parse_failure_is_tie():
    assert pairwise_r([True, True], {(0, 1): None})[0][1] == 0.5


def test_malformed_loses_every_pair():
    r = pairwise_r([True, False, True], {(0, 2): "A"})
    assert r[0][1] == 1 and r[1][0] == 0 and r[2][1] == 1 and r[1][2] == 0


def test_both_malformed_tie():
    assert pairwise_r([False, False], {})[0][1] == 0.5


def test_missing_verdict_for_valid_pair_raises():
    with pytest.raises(KeyError):
        pairwise_r([True, True], {})


@pytest.mark.parametrize("first,second,expected", [
    ("A", "B", 1.0),       # j wins in both orders
    ("A", "A", 0.5),       # the critic just likes slot A
    ("A", "Tie", 0.75),
    ("B", "Tie", 0.25),
    ("A", None, 0.75),     # a failure is a tie for that order
])
def test_a9_two_order_mean(first, second, expected):
    r = pairwise_r([True, True], {(0, 1): first, (1, 0): second})
    assert r[0][1] == expected and r[1][0] == 1 - expected


def test_r_total_weights():
    assert r_total(1, 1, 0.5, 0.25, lam=1, alpha=2, gamma=4) == 1 + 1 + 1 + 1
    assert r_total_rl_comm(1, 1, 0.25, 0.25, lam=1) == 2.5


def test_advantages_no_std_normalisation():
    assert group_advantages([1.0, 2.0, 3.0, 6.0]) == [-2.0, -1.0, 0.0, 3.0]
    assert sum(group_advantages([0.3, 0.1, 0.9])) == pytest.approx(0)
    assert group_advantages([2.0, 2.0]) == [0.0, 0.0]


# Degenerate-policy table, PLAN §3.3, driven through the rule-based critic.
@pytest.fixture
def clip():
    return Item("c", "What happens here today?", "yes", video="fake://piano", audio="fake://dog")


def score(paths, clip, a8, rng):
    backend = FakeBackend()
    parsed = [parse_path(p, clip) for p in paths]
    verdicts = score_group(a8, BackendExecutor(backend, Sampling(0, 1, False, 8)), clip, parsed, rng)
    return [r_xm(a, v) for a, v in zip(verdicts.audio, verdicts.visual)], verdicts


def honest_audio():
    return "The sound of dog can be heard. You can hear dog clearly. There is a dog sound outside."


def test_wrong_modality_padding_blocked(clip, a8, rng):
    honest = path(a=honest_audio(), v="The video shows piano.")
    padded = path(a=honest_audio() + " The video shows dog. The video shows piano. The video shows rain.", v="The video shows piano.")
    (s_honest, s_padded), verdicts = score([honest, padded], clip, a8, rng)
    assert verdicts.audio[1][3:] == [W, W, W]
    assert s_padded < s_honest


def test_vacuous_truth_blocked(clip, a8, rng):
    honest = path(a=honest_audio(), v="The video shows piano.")
    vacuous = path(a=honest_audio() + " Something is making a noise. Something else may occur.", v="The video shows piano.")
    (s_honest, s_vacuous), verdicts = score([honest, vacuous], clip, a8, rng)
    assert verdicts.audio[1][3:] == [Z, Z]
    assert s_vacuous < s_honest


def test_duplication_blocked(clip, a8, rng):
    once = path(a="The sound of dog can be heard.", v="The video shows piano.")
    copied = path(a="The sound of dog can be heard. " * 5, v="The video shows piano.")
    (s_once, s_copied), _ = score([once, copied], clip, a8, rng)
    assert s_once == s_copied


def test_paraphrase_passes_dedupe_as_the_plan_warns(clip, a8, rng):
    once = path(a="The sound of dog can be heard.", v="The video shows piano.")
    para = path(a="The sound of dog can be heard. You can hear dog clearly.", v="The video shows piano.")
    (s_once, s_para), _ = score([once, para], clip, a8, rng)
    assert s_para > s_once  # known gap, covered by the Gate 0 probe


def test_question_echo_blocked_on_natural_not_on_xm(clip, a8, rng):
    echo = path(a="What happens here today? The sound of dog can be heard.", v="The video shows piano.")
    assert parse_path(echo, clip).audio == ["The sound of dog can be heard."]
    xm = Item("c", clip.question, "yes", is_xm=True)
    assert len(parse_path(echo, xm).audio) == 2


def test_saying_little_is_limited(clip, a8, rng):
    little = path(a="The sound of dog can be heard.", v="The video shows piano.")
    more = path(a=honest_audio(), v="The video shows piano.")
    (s_little, s_more), _ = score([little, more], clip, a8, rng)
    assert s_little < s_more


def test_fusion_laundering_over_cap_is_malformed(clip, a8, rng):
    honest = path(a=honest_audio(), v="The video shows piano.")
    laundered = path(a=honest_audio(), v="The video shows piano.", f="One thing. Two things. Three things.")
    (_, s_laundered), verdicts = score([honest, laundered], clip, a8, rng)
    assert s_laundered == 0.0
    assert pairwise_r([True, False], verdicts.pair_outcomes)[0][1] == 1.0


def test_malformed_gets_no_passes(clip, a8, rng):
    _, verdicts = score([path(), "garbage", path(), path()], clip, a8, rng)
    assert verdicts.passes == 3 + 2 * 3
    assert all(1 not in pair for pair in verdicts.pair_outcomes)


