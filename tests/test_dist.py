"""Two real processes under torchrun: gloo, CPU, fake backend."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from dist_worker import run_grads

from avrsi import dist

ROOT = Path(__file__).resolve().parent.parent


def launch(out, *args, world=2):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": f"{ROOT}:{ROOT / 'tests'}:{ROOT / '.venv' / 'run_lib'}:{os.environ.get('PYTHONPATH', '')}",
           "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={world}", "--master_addr=127.0.0.1", f"--master_port={port}",
               str(ROOT / "tests" / "dist_worker.py"), args[0], str(out), str(world), *map(str, args[1:])]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[:2500]


def ranks(out, world=2):
    return [torch.load(Path(out) / f"rank{r}.pt", weights_only=False) for r in range(world)]


def rows(path):
    return [{k: v for k, v in json.loads(line).items() if k != "step_seconds"} for line in Path(path).read_text().splitlines()]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("two")
    launch(out, "train", 4)
    return out


def test_params_critic_and_guard_stay_identical_across_ranks(run):
    a, b = ranks(run)
    assert torch.equal(a["policy"], b["policy"]) and torch.equal(a["critic"], b["critic"])
    assert a["decisions"] == b["decisions"] and len(a["decisions"]) == 2
    assert a["syncs"] == b["syncs"] == 1 + sum(a["decisions"])  # the start-up copy plus each handoff


def test_params_moved_from_the_broadcast_start(run):
    from avrsi.models.fake import FakeBackend

    from conftest import CLASSES
    assert not torch.equal(ranks(run)[0]["policy"], FakeBackend(CLASSES, seed=0).policy.detach())


def test_rollouts_are_shared_out_for_a_global_batch(run):
    a, b = ranks(run)
    records = a["steps"]
    assert [r["trajectories"] for r in records] == [8] * 4  # B=4 x K=2, over both ranks
    assert [r["passes"] for r in records] == [r["passes"] for r in b["steps"]]
    assert a["calls"]["actor"] > 0 and b["calls"]["actor"] > 0


def test_only_rank_zero_writes(run):
    assert len(rows(run / "steps.jsonl")) == 4 and len(rows(run / "dev.jsonl")) == 2
    assert len(rows(run / "guard.jsonl")) == 3  # θ_0's reference and two decisions
    assert (run / "checkpoints" / "LATEST").read_text() == "step_000004"
    state = torch.load(run / "checkpoints" / "step_000004" / "state.pt", weights_only=False)
    assert len(state["rank_rngs"]) == 2 and state["rank_rngs"][0]["python"] != state["rank_rngs"][1]["python"]


def test_guard_metrics_cover_all_ranks(run):
    reference = rows(run / "guard.jsonl")[0]
    assert reference["generations"] == 8 and reference["pair_passes"] == 2 * 8 and reference["check_passes"] == 8


def test_resume_equals_an_uninterrupted_two_rank_run(run, tmp_path):
    launch(tmp_path, "train", 2)
    (tmp_path / "steps.jsonl").open("a").write('{"step": 3}\n')  # past the checkpoint, as a crash leaves it
    launch(tmp_path, "resume", 4)
    whole, resumed = ranks(run), ranks(tmp_path)
    for x, y in zip(whole, resumed):
        assert torch.equal(x["policy"], y["policy"]) and torch.equal(x["critic"], y["critic"])
    assert rows(run / "steps.jsonl") == rows(tmp_path / "steps.jsonl")
    assert rows(run / "guard.jsonl") == rows(tmp_path / "guard.jsonl")


def test_resume_refuses_another_world_size(run, tmp_path):
    for name in ("checkpoints", "run.json"):
        subprocess.run(["cp", "-r", str(run / name), str(tmp_path)], check=True)
    with pytest.raises(AssertionError, match="2-rank|rank run"):
        launch(tmp_path, "resume", 6, world=3)


def test_summed_gradients_equal_one_process_on_the_whole_batch(tmp_path):
    single = tmp_path / "one"
    single.mkdir()
    run_grads(dist.LOCAL, single)
    launch(tmp_path, "grads")
    two = ranks(tmp_path)
    one = torch.load(single / "rank0.pt", weights_only=False)
    assert torch.equal(two[0]["policy"], two[1]["policy"])
    assert torch.allclose(two[0]["policy"], one["policy"], atol=1e-6)
    assert two[0]["stats"]["loss"] == pytest.approx(one["stats"]["loss"]) and two[0]["stats"]["kl"] == pytest.approx(one["stats"]["kl"])


@pytest.mark.skipif(not (ROOT / ".venv" / "run_lib" / "peft").is_dir(), reason="peft absent")
def test_real_thinker_trains_and_resumes_on_two_ranks(tmp_path):
    whole, broken = tmp_path / "whole", tmp_path / "broken"
    for path in (whole, broken):
        path.mkdir()
    launch(whole, "real", 4)
    a, b = ranks(whole)
    assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"])) and all(torch.equal(x, y) for x, y in zip(a["critic"], b["critic"]))
    assert (whole / "checkpoints" / "step_000004" / "adapters" / "actor" / "adapter_model.safetensors").exists()
    assert rows(whole / "steps.jsonl")[-1]["trajectories"] == 4
    launch(broken, "real", 2)
    launch(broken, "real_resume", 4)
    for x, y in zip(ranks(whole), ranks(broken)):
        assert all(torch.equal(p, q) for p, q in zip(x["params"], y["params"]))


def test_local_comm_is_inert():
    assert dist.LOCAL.is_main and dist.LOCAL.sum(3) == 3 and dist.LOCAL.gather_flat([1]) == [1]
