"""Checkpoints, --resume, and main() end to end on the fake backend."""
import json
import random
from dataclasses import replace

import pytest
import torch
from conftest import CLASSES, make_disjoint
from test_sync import BASE, resolve

from avrsi.config import resolve_arm
from avrsi.critic import prompt_digest
from avrsi.data import REGISTRY
from avrsi.models.base import Generation
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item
from avrsi.perturb import build_planted, freeze, save_planted
from avrsi.sync import GuardSet
from avrsi.train import load_checkpoint, main, save_checkpoint, train, truncate_jsonl, xm_dev_scorer


@pytest.fixture(scope="module")
def guard_set(xm):
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    planted = build_planted(carriers, {i.item_id: BASE for i in carriers}, make_disjoint(), random.Random(0), per_type=64)
    return GuardSet([resolve(i) for i in xm["guard"][::16]], planted[::16], resolve=resolve)


@pytest.fixture(scope="module")
def data(xm):
    natural = [Item(f"n{i}", "Is something here now?", "yes" if i % 2 else "no", video=f"fake://{c}", audio=f"fake://{c}")
               for i, c in enumerate(CLASSES)]
    return natural, [resolve(i) for i in xm["train"]], [resolve(i) for i in xm["dev"][:20]]


ARM = replace(resolve_arm("A8"), T=3, dev_every=2, lr=1e-3)


def go(data, guard_set, samplings, out, steps, resume=False, arm=ARM, seed=0, **kwargs):
    natural, xm_train, dev = data
    backend = kwargs.pop("backend", FakeBackend(CLASSES))
    result = train(arm, backend, natural, xm_train, guard_set, samplings, xm_dev_scorer(dev, samplings), seed, steps, out, resume=resume, **kwargs)
    return backend, result


def logs(out):
    rows = {}
    for name in ("steps", "dev", "guard"):
        lines = [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
        rows[name] = [{k: v for k, v in row.items() if k != "step_seconds"} for row in lines]
    return rows


@pytest.mark.parametrize("stop", [4, 5])
def test_resumed_run_equals_uninterrupted(stop, data, guard_set, samplings, tmp_path):
    whole = tmp_path / "whole"
    full, _ = go(data, guard_set, samplings, whole, 7)
    broken = tmp_path / "broken"
    go(data, guard_set, samplings, broken, stop)  # step 5 is past the checkpoint at 4, as a crash leaves it
    resumed, result = go(data, guard_set, samplings, broken, 7, resume=True)
    assert [s["step"] for s in result.steps] == [5, 6, 7]
    assert logs(whole) == logs(broken)
    assert torch.equal(full.policy, resumed.policy) and torch.equal(full.critic_table, resumed.critic_table)
    assert full.sync_count == resumed.sync_count and len(result.guard.decisions) == 2


def test_dev_scores_reach_dev_jsonl(data, guard_set, samplings, tmp_path):
    _, result = go(data, guard_set, samplings, tmp_path, 4)
    rows = [json.loads(line) for line in (tmp_path / "dev.jsonl").read_text().splitlines()]
    assert rows == result.dev and [r["step"] for r in rows] == [2, 4]


def test_checkpoint_holds_everything(data, guard_set, samplings, tmp_path):
    go(data, guard_set, samplings, tmp_path, 4)
    assert (tmp_path / "checkpoints" / "LATEST").read_text() == "step_000004"
    assert not (tmp_path / "checkpoints" / "step_000002").exists()
    state = torch.load(tmp_path / "checkpoints" / "step_000004" / "state.pt", weights_only=False)
    assert {"step", "optimizer", "python_rng", "torch_rng", "guard"} <= set(state) and state["step"] == 4
    assert state["guard"]["theta0"] and state["guard"]["incumbent"] and len(state["guard"]["decisions"]) == 1


def test_resume_without_checkpoint_refuses(data, guard_set, samplings, tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"prompt_digest": prompt_digest()}))
    with pytest.raises(FileNotFoundError):
        go(data, guard_set, samplings, tmp_path, 4, resume=True)


def test_resume_refuses_changed_prompts(data, guard_set, samplings, tmp_path):
    go(data, guard_set, samplings, tmp_path, 2)
    (tmp_path / "run.json").write_text(json.dumps({"prompt_digest": "old"}))
    with pytest.raises(ValueError, match="prompts changed"):
        go(data, guard_set, samplings, tmp_path, 4, resume=True)


def test_truncate_keeps_stepless_rows_and_drops_torn_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"event": "theta0_reference"}\n{"step": 2}\n{"step": 3}\n{"step": ')
    truncate_jsonl(path, 2)
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"event": "theta0_reference"}, {"step": 2}]


def test_seed_changes_the_run_and_repeats(data, guard_set, samplings):
    def batches(seed):
        natural, xm_train, _ = data
        backend = FakeBackend(CLASSES)
        train(replace(ARM, T=None), backend, natural, xm_train, guard_set, samplings, None, seed, 2)
        return [x.prompt for role, x in backend.seen_inputs if role == "actor" and x.prompt.startswith("Answer")]

    assert batches(1) == batches(1) and batches(1) != batches(2)


class BareAnswer(FakeBackend):
    """Rollouts with no tags, so only the fallback extractor can read them."""

    def _rollout(self, role, sampling):
        return Generation("yes", [self.ids["yes"], self.ids["</answer>"]])


def test_reference_uses_the_fallback_under_the_20_percent_rule(data, guard_set, samplings):
    natural, xm_train, _ = data
    flagged = replace(guard_set, use_fallback_reference=True)
    arm = replace(ARM, T=None, dev_every=100)
    with_flag = train(arm, BareAnswer(CLASSES), natural, xm_train, flagged, samplings, None, 0, 1)
    without = train(arm, BareAnswer(CLASSES), natural, xm_train, guard_set, samplings, None, 0, 1)
    assert with_flag.guard.theta0.outcome > 0 and without.guard.theta0.outcome == 0


def test_rollouts_carry_the_fill(data, samplings):
    natural = [Item("n", "Is something here now?", "yes", video="fake://dog", modalities=("video",))]
    backend = FakeBackend(CLASSES)
    train(replace(resolve_arm("A1"), lr=1e-3), backend, natural, [], None, samplings, None, 0, 1, fill="silent_track")
    rollouts = [x for role, x in backend.seen_inputs if role == "actor"]
    assert rollouts and all(x.fill == "silent_track" for x in rollouts)


@pytest.fixture
def run_dir(tmp_path, xm, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in ("train", "dev", "guard"):
        freeze(xm[name][:64] if name != "dev" else xm[name], frozen / f"{name}.jsonl")
    natural = [Item(f"n{i}", "Is something here now?", "yes" if i % 2 else "no", video=f"fake://{c}", audio=f"fake://{c}")
               for i, c in enumerate(CLASSES)]
    monkeypatch.setitem(REGISTRY, "toy", lambda root: natural)
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    planted = build_planted(carriers, {i.item_id: BASE for i in carriers}, make_disjoint(), random.Random(0), per_type=4)

    def invoke(name, backend, *extra, report=None):
        path = tmp_path / f"{name}_planted.json"
        save_planted(planted, report or {"use_fallback_reference": False}, path)
        monkeypatch.setattr("avrsi.models.load_backend", lambda model, device: backend)
        out = tmp_path / name
        main(["--model", "qwen3_omni", "--natural", "toy", "--natural-root", "", "--xm-dir", str(frozen), "--planted", str(path),
              "--media-root", "m", "--out", str(out), "--steps", "1", *extra])
        return out

    return invoke


def test_main_runs_and_logs_the_digest(run_dir):
    out = run_dir("a", FakeBackend(CLASSES), "--arm", "A8")
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["prompt_digest"] == prompt_digest() and manifest["model"]["licence"] == "Apache-2.0" and manifest["model"]["transformers"]
    assert manifest["t8_probes"]["audio_changes_output"] > 0
    assert (out / "steps.jsonl").exists() and (out / "adapter" / "fake.pt").exists()


def test_main_passes_the_seed(run_dir):
    prompts = {}
    for seed in (1, 2):
        backend = FakeBackend(CLASSES)
        run_dir(f"s{seed}", backend, "--arm", "A1", "--seed", str(seed))
        prompts[seed] = [x.prompt for role, x in backend.seen_inputs if role == "actor" and x.prompt.startswith("Answer")]
    assert prompts[1] != prompts[2]


def test_main_seeds_before_attach_lora(run_dir):
    class Recording(FakeBackend):
        drawn = None

        def attach_lora(self, r):
            Recording.drawn = torch.rand(1).item()

    run_dir("first", Recording(CLASSES), "--arm", "A1", "--seed", "7")
    torch.manual_seed(7)
    assert Recording.drawn == torch.rand(1).item()


def test_main_keeps_the_fallback_flag(run_dir):
    out = run_dir("f", BareAnswer(CLASSES), "--arm", "A8", report={"use_fallback_reference": True})
    reference = json.loads((out / "guard.jsonl").read_text().splitlines()[0])
    assert reference["event"] == "theta0_reference" and reference["outcome"] > 0


def test_main_a14_needs_T(run_dir):
    with pytest.raises(ValueError, match="needs --T"):
        run_dir("t", FakeBackend(CLASSES), "--arm", "A14")
    with pytest.raises(ValueError, match="T=200 not in"):
        run_dir("t2", FakeBackend(CLASSES), "--arm", "A14", "--T", "200")
    out = run_dir("t3", FakeBackend(CLASSES), "--arm", "A14", "--T", "500")
    assert json.loads((out / "run.json").read_text())["arm"]["T"] == 500


def test_main_a14_T_list_follows_tuning(run_dir):
    with pytest.raises(ValueError, match="T=100 not in"):
        run_dir("t", FakeBackend(CLASSES), "--arm", "A14", "--T", "100", "--tuned", '{"T": 100}')
    run_dir("t2", FakeBackend(CLASSES), "--arm", "A14", "--T", "200", "--tuned", '{"T": 100}')


def test_main_refuses_T_on_other_arms(run_dir):
    with pytest.raises(ValueError, match="T=100 not in"):
        run_dir("t", FakeBackend(CLASSES), "--arm", "A8", "--T", "100")


def test_main_wires_the_a7_scorer(run_dir, monkeypatch):
    seen = {}
    scorer = lambda item, paths: [(0.0, 0.0) for _ in paths]  # noqa: E731
    monkeypatch.setattr("avrsi.rlcomm.load_rl_comm_scorer", lambda config: seen.update(config) or scorer)
    run_dir("a7", FakeBackend(CLASSES), "--arm", "A7")
    assert "audio-flamingo-3" in seen["lalm"]


def test_main_scopes_the_gemini_spend_cap(run_dir, monkeypatch):
    seen = {}

    class External:
        def complete(self, prompt, media):
            return "A"

    monkeypatch.setattr("avrsi.judges.make_judge", lambda config, scope: seen.update(scope=scope) or External())
    monkeypatch.setattr("avrsi.judges.media.view_media", lambda *args: [])
    run_dir("a3", FakeBackend(CLASSES), "--arm", "A3")
    assert seen["scope"] == "A3"


def test_main_passes_resume(run_dir, monkeypatch):
    seen = {}
    monkeypatch.setattr("avrsi.train.train", lambda *args, **kwargs: seen.update({"resume": args[13]}))
    run_dir("r", FakeBackend(CLASSES), "--arm", "A1", "--resume")
    assert seen["resume"] is True


def test_checkpoint_restores_the_torch_rng(tmp_path):
    backend = FakeBackend(CLASSES)
    optimizer = torch.optim.AdamW(backend.trainable_parameters(), lr=1e-3)
    save_checkpoint(tmp_path, 1, backend, optimizer, random.Random(0), None)
    expected = torch.rand(3)
    load_checkpoint(tmp_path, FakeBackend(CLASSES), 1e-3, random.Random(1), None)
    assert torch.equal(torch.rand(3), expected)
