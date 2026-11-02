import json
import random
from dataclasses import replace

import pytest
import torch
from conftest import CLASSES, make_disjoint
from test_sync import BASE, resolve

from avrsi.config import resolve_arm
from avrsi.critic import AV_PAIR
from avrsi.grpo import grpo_loss
from avrsi.models.base import ACTOR, Generation
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item
from avrsi.perturb import build_planted
from avrsi.sync import GuardSet
from avrsi.train import group_rewards, grpo_update, train, xm_dev_scorer


@pytest.fixture(scope="module")
def guard_set(xm):
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    planted = build_planted(carriers, {i.item_id: BASE for i in carriers}, make_disjoint(), random.Random(0), per_type=64)
    return GuardSet([resolve(i) for i in xm["guard"]], planted, resolve=resolve)


@pytest.fixture(scope="module")
def natural():
    return [Item(f"n{i}", "Is something here now?", "yes" if i % 2 else "no", video=f"fake://{c}", audio=f"fake://{c}")
            for i, c in enumerate(CLASSES)]


@pytest.fixture(scope="module")
def xm_train(xm):
    return [resolve(i) for i in xm["train"]]


def run(arm, natural, xm_train, guard_set, samplings, steps=4, **kwargs):
    return train(arm, kwargs.pop("backend"), natural, xm_train, guard_set, samplings, seed=0, steps=steps, **kwargs)


def small(arm_id="A8", **over):
    return replace(resolve_arm(arm_id), T=2, lr=1e-3, **over)


def test_critic_refreshes_and_rewards_vary(natural, xm_train, guard_set, samplings, tmp_path):
    backend = FakeBackend(CLASSES)
    start = backend.policy.detach().clone()
    result = run(small(), natural, xm_train, guard_set, samplings, backend=backend, out_dir=tmp_path)
    assert len(result.guard.decisions) == 2 and backend.sync_count == 2
    assert torch.equal(backend.critic_table, backend.policy.detach())
    assert not torch.equal(backend.policy.detach(), start)
    assert torch.equal(backend.reference_table, start)  # θ_0 stays for the KL
    assert all(s["reward_group_spread"] > 0 for s in result.steps)
    assert len({round(s["reward_mean"], 6) for s in result.steps}) > 1
    assert (tmp_path / "steps.jsonl").exists() and (tmp_path / "guard.jsonl").exists()


def test_critic_stale_between_syncs(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    run(small(), natural, xm_train, guard_set, samplings, backend=backend, steps=3)
    assert backend.sync_count == 1 and not torch.equal(backend.critic_table, backend.policy.detach())


def test_rewards_are_scored_by_theta_c(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    run(small(), natural, xm_train, guard_set, samplings, backend=backend, steps=1)
    roles = {r for r, x in backend.seen_inputs if x.prompt.startswith(("You are checking", "Two reasoning"))}
    assert "critic" in roles


def test_lenient_actor_is_blocked_by_guard(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)

    def corrupt(b, step):
        if step == 1:
            b.leniency[ACTOR] = 1.0
        return {}

    arm = replace(small(), dev_every=1)
    result = run(arm, natural, xm_train, guard_set, samplings, backend=backend, dev_score=corrupt)
    assert backend.sync_count == 0
    assert all(not d.handoff and not d.condition_2 for d in result.guard.decisions)
    assert result.guard.decisions[-1].block_alarm and result.guard.decisions[-1].drift_alarm
    assert backend.leniency["critic"] == 0.0


def test_guard_runs_once_as_reference_plus_every_T(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = run(small(), natural, xm_train, guard_set, samplings, backend=backend, steps=6)
    assert len(result.guard.decisions) == 3 and result.guard.theta0 is not None


def test_pass_counts_per_step_bounded(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = run(small(), natural, xm_train, guard_set, samplings, backend=backend)
    assert all(s["passes"] <= 14 * 8 for s in result.steps) and max(s["passes"] for s in result.steps) == 112


def test_a1_runs_no_critic_passes_and_no_guard(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = run(small("A1"), natural, xm_train, None, samplings, backend=backend)
    assert all(s["passes"] == 0 for s in result.steps) and result.guard is None
    assert backend.calls["critic"] == 0 and backend.sync_count == 0


def test_a5_never_syncs(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    arm = replace(resolve_arm("A5"), lr=1e-3)
    result = run(arm, natural, xm_train, None, samplings, backend=backend)
    assert backend.sync_count == 0 and result.guard is None and backend.calls["critic"] > 0


def test_a2_pairs_only_but_guards_with_j_check(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = run(small("A2"), natural, xm_train, guard_set, samplings, backend=backend)
    assert all(s["passes"] <= 6 * 8 for s in result.steps) and result.guard.decisions
    guard_checks = [x for r, x in backend.seen_inputs if r == ACTOR and x.prompt.startswith("You are checking")]
    assert guard_checks and all(x.audio and x.video for x in guard_checks)


def test_a15_checks_see_both_modalities(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    run(small("A15"), natural, xm_train, guard_set, samplings, backend=backend)
    checks = [x for r, x in backend.seen_inputs if r == "critic" and x.prompt.startswith("You are checking")]
    assert checks and all(x.audio and x.video for x in checks)


def test_a8_checks_are_isolated(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    run(small(), natural, xm_train, guard_set, samplings, backend=backend, steps=1)
    checks = [x for r, x in backend.seen_inputs if r == "critic" and x.prompt.startswith("You are checking")]
    assert checks and all(bool(x.audio) != bool(x.video) for x in checks)


def test_a9_and_a11(natural, xm_train, guard_set, samplings):
    backend = FakeBackend(CLASSES)
    result = run(replace(small("A9"), T=None), natural, xm_train, guard_set, samplings, backend=backend, steps=1)
    assert result.steps[0]["passes"] <= 20 * 8
    backend = FakeBackend(CLASSES)
    result = run(replace(small("A11"), T=None), natural, xm_train, guard_set, samplings, backend=backend, steps=1)
    assert result.steps[0]["passes"] <= 44 * 8 and result.steps[0]["passes"] > 14 * 8


def test_a10_minibatch_guard_loads_no_xm_guard(natural, xm_train, samplings):
    backend = FakeBackend(CLASSES)
    result = run(small("A10"), natural, xm_train, None, samplings, backend=backend)
    assert len(result.guard.decisions) == 2 and all(d.condition_2 is None for d in result.guard.decisions)


class Replying:
    def __init__(self):
        self.calls = 0

    def run(self, item, passes):
        self.calls += len(passes)
        return ["A" if p.view == AV_PAIR else "\n".join(f"{i + 1}: supported" for i in range(p.n_sentences)) for p in passes]


@pytest.mark.parametrize("arm_id", ["A3", "A4"])
def test_external_critic_replaces_all_self_views(arm_id, natural, xm_train, samplings):
    backend, external = FakeBackend(CLASSES), Replying()
    result = run(small(arm_id), natural, xm_train, None, samplings, backend=backend, external_executor=external)
    assert external.calls > 0 and backend.calls["critic"] == 0 and result.guard is None


def test_gemini_arm_requires_executor(natural, xm_train, samplings):
    with pytest.raises(ValueError):
        run(small("A3"), natural, xm_train, None, samplings, backend=FakeBackend(CLASSES))


def test_a7_needs_undefined_scorer(natural, xm_train, samplings):
    with pytest.raises(NotImplementedError, match="r_arr"):
        run(small("A7"), natural, xm_train, None, samplings, backend=FakeBackend(CLASSES))


def test_a7_with_scorer(natural, xm_train, samplings):
    result = run(small("A7"), natural, xm_train, None, samplings, backend=FakeBackend(CLASSES),
                 rl_comm_scorer=lambda item, paths: [(0.5, 0.5) for _ in paths])
    assert result.steps[0]["passes"] == 0


def test_a0_refuses_training(natural, xm_train, samplings):
    with pytest.raises(ValueError):
        run(resolve_arm("A0"), natural, xm_train, None, samplings, backend=FakeBackend(CLASSES))


def test_xm_guard_arm_requires_guard_set(natural, xm_train, samplings):
    with pytest.raises(ValueError):
        run(small(), natural, xm_train, None, samplings, backend=FakeBackend(CLASSES))


def test_silence_item_uses_n1():
    arm = resolve_arm("A8")

    class AllSupported:
        def run(self, item, passes):
            return ["A" if p.view == AV_PAIR else "1: supported" for p in passes]

    text = "<audio>Quiet.</audio><visual>A view.</visual><fusion></fusion><answer>no</answer>"
    item = Item("s", "q", "no", is_xm=True, silence=True, video="v", audio="a")
    totals, _ = group_rewards(arm, item, [Generation(text)] * 2, AllSupported(), random.Random(0))
    # outcome 1 + format 1 + pair mean 0.5 + xm ½(1 + 1/3)
    assert sum(totals) / 2 == pytest.approx(1 + 1 + 0.5 + 2 / 3)


def test_update_favours_higher_advantage():
    backend = FakeBackend(CLASSES, seed=0)
    item_input = None
    good = Generation("", [backend.ids["<answer>"], backend.ids["yes"]])
    bad = Generation("", [backend.ids["<answer>"], backend.ids["no"]])
    optimizer = torch.optim.SGD(backend.trainable_parameters(), lr=1.0)
    before = backend.score_logprobs([item_input, item_input], [good, bad])
    grpo_update(backend, optimizer, [(item_input, good, 1.0), (item_input, bad, -1.0)], beta=0.0)
    after = backend.score_logprobs([item_input, item_input], [good, bad])
    assert after[0] > before[0] and after[1] < before[1]


def test_sequential_update_equals_batch_mean():
    backend = FakeBackend(CLASSES, seed=0)
    good = Generation("", [backend.ids["<audio>"], backend.ids["</audio>"], backend.ids["<visual>"]])
    bad = Generation("", [backend.ids["<answer>"], backend.ids["no"]])
    opt = torch.optim.SGD(backend.trainable_parameters(), lr=0.0)
    grpo_update(backend, opt, [(None, good, 1.0), (None, bad, -0.5)], beta=0.01)
    sequential = backend.policy.grad.clone()

    backend.policy.grad = None
    logps = backend.score_logprobs([None, None], [good, bad])
    refs = backend.score_logprobs([None, None], [good, bad], "reference")
    n = 2
    pad = lambda t: torch.nn.functional.pad(t, (0, n - t.numel()))  # noqa: E731
    lp, rf = torch.stack([pad(t) for t in logps]), torch.stack([pad(t) for t in refs])
    mask = torch.tensor([[1, 1], [1, 0]])
    loss, _ = grpo_loss(lp, lp.detach(), rf.detach(), torch.tensor([1.0, -0.5]), mask, 0.01)
    loss.backward()
    assert torch.allclose(sequential, backend.policy.grad)


def test_dev_scorer_runs_on_xm_rows(xm, samplings):
    backend = FakeBackend(CLASSES)
    scorer = xm_dev_scorer([resolve(i) for i in xm["dev"][:40]], samplings)
    assert 0 <= scorer(backend, 400)["xm_dev_balanced_accuracy"] <= 1


def test_dev_only_every_dev_every(natural, xm_train, guard_set, samplings):
    seen = []
    arm = replace(small(), dev_every=2)
    run(arm, natural, xm_train, guard_set, samplings, backend=FakeBackend(CLASSES), dev_score=lambda b, s: seen.append(s) or {})
    assert seen == [2, 4]


def test_step_log_is_json(natural, xm_train, guard_set, samplings, tmp_path):
    run(small(), natural, xm_train, guard_set, samplings, backend=FakeBackend(CLASSES), out_dir=tmp_path, steps=1)
    row = json.loads((tmp_path / "steps.jsonl").read_text().splitlines()[0])
    assert {"loss", "kl", "tie_rate", "supported_rate", "fusion_token_share", "step_seconds", "tokens_mean"} <= set(row)
