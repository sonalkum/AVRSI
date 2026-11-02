"""Run by test_dist.py under torchrun, on the fake backend with gloo."""
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch
from conftest import CLASSES, make_clips, make_disjoint
from test_sync import BASE, resolve

from avrsi import dist
from avrsi.config import resolve_arm
from avrsi.models.base import Generation, Sampling
from avrsi.models.fake import FakeBackend
from avrsi.parse import Item
from avrsi.perturb import build_planted, build_xm, split_sources
from avrsi.sync import GuardSet
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


if __name__ == "__main__":
    comm, device, _ = dist.setup()
    assert device == "cpu" and comm.world == int(sys.argv[3])
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "grads":
        run_grads(comm, out)
    else:
        run_training(comm, out, int(sys.argv[4]), mode == "resume")
    dist.close(comm)
