"""Run by test_dist.py under torchrun, on the fake backend with gloo."""
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from conftest import CLASSES, make_clips, make_disjoint
from test_sync import BASE, resolve

from avrsi import dist
from avrsi import train as train_module
from avrsi.config import resolve_arm
from avrsi.critic import actor_prompt
from avrsi.models.base import Generation, Sampling
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item
from avrsi.perturb import build_planted, build_xm, split_sources
from avrsi.sync import Guard, GuardScores, GuardSet, actor_outcome, evaluate_actor
from avrsi.train import Samplings, grpo_update, train, xm_dev_scorer

SIZES = {"train": 100, "guard": 1024, "dev": 512, "gate": 100, "test": 100}
SAMPLINGS = Samplings(Sampling(1.0, 1.0, True, 64), Sampling(0.0, 1.0, False, 64), Sampling(0.0, 1.0, False, 64), self_preference_items=2)
ARM = replace(resolve_arm("A8"), B=4, K=2, T=2, dev_every=2, lr=1e-3)


def fixture():
    xm = build_xm(split_sources(make_clips(1000), SIZES, seed=1), make_disjoint(), seed=1, sizes=SIZES)
    carriers = [i for i in xm["guard"] if i.planted_carrier]
    planted = build_planted(carriers, {i.item_id: BASE for i in carriers}, make_disjoint(), random.Random(0), per_type=64)
    guard_set = GuardSet([resolve(i) for i in xm["guard"][::128]], planted[::32], resolve=resolve)
    natural = [Item(f"n{i}", "Is something here now?", "yes" if i % 2 else "no", video=f"fake://{c}", audio=f"fake://{c}") for i, c in enumerate(CLASSES)]
    return natural, [resolve(i) for i in xm["train"]], [resolve(i) for i in xm["dev"][:8]], guard_set


def save_state(path, comm, backend, extra=None):
    torch.save({"policy": backend.policy.detach(), "critic": backend.critic_table, "syncs": backend.sync_count, "calls": backend.calls, **(extra or {})},
               Path(path) / f"rank{comm.rank}.pt")


def run_training(comm, out, steps, resume):
    natural, xm_train, dev, guard_set = fixture()
    backend = FakeBackend(CLASSES, seed=comm.rank)  # different on purpose: broadcast must fix the start
    result = train(ARM, backend, natural, xm_train, guard_set, SAMPLINGS, xm_dev_scorer(dev, SAMPLINGS, comm=comm), seed=0, steps=steps, out_dir=out,
                   resume=resume, comm=comm)
    save_state(out, comm, backend, {"decisions": [d.handoff for d in result.guard.decisions], "steps": result.steps})


def run_grads(comm, out):
    """Fixed rollouts split across ranks, to compare with one process doing them all."""
    backend = FakeBackend(CLASSES, seed=0)
    ids = backend.ids
    tokens = [ids[t] for t in ("<audio>", "</audio>", "<visual>", "</visual>", "<fusion>", "</fusion>", "<answer>", "yes", "no", "</answer>")]
    rollouts = [(None, Generation("", tokens[i:i + 2 + i % 3]), (-1.0) ** i * (i + 1)) for i in range(8)]
    optimizer = torch.optim.SGD(backend.trainable_parameters(), lr=0.1)
    stats = grpo_update(backend, optimizer, rollouts[comm.rank::comm.world], beta=0.1, comm=comm)
    save_state(out, comm, backend, {"stats": stats})


def guard_scores(comm):
    """The guard on yes, no, yes, no items under a backend that always answers yes: rank 0's shard scores
    1, rank 1's 0, and only the gathered outcome is 1/2. Returns the global and the rank-local results."""
    guard_set = fixture()[3]
    yes, no = ([i for i in guard_set.items if i.answer == answer] for answer in ("yes", "no"))
    guard_set = replace(guard_set, items=[x for pair in zip(yes, no) for x in pair])
    scores = evaluate_actor(FakeBackend(CLASSES), guard_set, ARM, actor_prompt, SAMPLINGS.greedy, SAMPLINGS.critic, comm=comm)
    guard = Guard(ARM)
    guard.set_reference(GuardScores(0.5, 0.5))
    local = actor_outcome(FakeBackend(CLASSES), guard_set.items[comm.rank::comm.world], actor_prompt, SAMPLINGS.greedy)[0]
    return {"scores": asdict(scores), "decision": asdict(guard.decide(1, scores)), "local_outcome": local}


def run_diverge(comm, out):
    """Rank 1's parameters drift after each update; the first checkpoint must refuse to save them."""
    update = train_module.grpo_update

    def drifting(backend, *args, **kwargs):
        stats = update(backend, *args, **kwargs)
        if comm.rank == 1:
            backend.policy.data[0, 0] += 1e-3
        return stats

    train_module.grpo_update = drifting
    run_training(comm, out, 2, False)


def run_real(comm, out, steps, resume):
    """A1 on the tiny real thinker and real PEFT, text only: adapters saved by rank 0, loaded by both."""
    from qwen3_tiny import BASE, tiny_thinker
    from test_qwen3_real import Tiny

    torch.manual_seed(0)
    backend = Tiny({**BASE, "lora_exclude_modules": r"(audio_tower|visual)\..*", "gradient_checkpointing": True, "logprob_batch": 2, "generate_batch": 2},
                   tiny_thinker(0))
    torch.manual_seed(comm.rank)  # different on purpose: broadcast must fix the start
    backend.attach_lora(4)
    natural = [Item(f"n{i}", "Is it here?", "yes", modalities=()) for i in range(4)]
    arm = replace(resolve_arm("A1"), B=2, K=2, dev_every=2, lr=1e-2, max_new_tokens=4)
    sampling = Sampling(1.0, 1.0, True, 4)
    train(arm, backend, natural, [], None, Samplings(sampling, sampling, sampling), seed=0, steps=steps, out_dir=out, resume=resume, comm=comm)
    params = [p.detach().clone() for p in backend.trainable_parameters()]
    critic = [p.detach().clone() for n, p in backend.model.named_parameters() if ".critic." in n]
    torch.save({"params": params, "critic": critic}, Path(out) / f"rank{comm.rank}.pt")


if __name__ == "__main__":
    comm, device, _ = dist.setup()
    assert device == "cpu" and comm.world == int(sys.argv[3])
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "grads":
        run_grads(comm, out)
    elif mode == "guard":
        torch.save(guard_scores(comm), Path(out) / f"rank{comm.rank}.pt")
    elif mode == "diverge":
        run_diverge(comm, out)
    elif mode.startswith("real"):
        run_real(comm, out, int(sys.argv[4]), mode == "real_resume")
    else:
        run_training(comm, out, int(sys.argv[4]), mode == "resume")
    dist.close(comm)
