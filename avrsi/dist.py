"""Multi-process training (torchrun). Every rank holds a full model, so there is no DDP wrapper: ranks
split the prompts and gradients are all-reduced by hand. LOCAL is the single-process case, all no-ops."""
from __future__ import annotations

import hashlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist

TIMEOUT = timedelta(hours=6)  # ranks wait for rank 0's checkpoints and self-preference


class Comm:
    rank, world = 0, 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def gather(self, value) -> list:
        """Every rank's value, in rank order, on every rank."""
        return [value]

    def gather_flat(self, values: list) -> list:
        return [v for part in self.gather(values) for v in part]

    def sum(self, value):
        return sum(self.gather(value))

    def barrier(self) -> None:
        pass

    def broadcast_params(self, params) -> None:
        pass

    def reduce_grads(self, params) -> None:
        pass

    def assert_identical(self, params) -> None:
        pass


class Distributed(Comm):
    def __init__(self):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()

    def gather(self, value) -> list:
        out = [None] * self.world
        dist.all_gather_object(out, value)
        return out

    def barrier(self) -> None:
        dist.barrier()

    def broadcast_params(self, params) -> None:
        for p in params:
            dist.broadcast(p.data, 0)

    def reduce_grads(self, params) -> None:
        """Sums over ranks. Callers divide by the global trajectory count, not the local one."""
        device = params[0].device
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().to(device) for p in params])
        dist.all_reduce(flat)
        for p, part in zip(params, flat.split([p.numel() for p in params])):
            p.grad = part.view_as(p).to(p.device)

    def assert_identical(self, params) -> None:
        digest = hashlib.sha256()
        for p in params:
            digest.update(p.detach().cpu().numpy().tobytes())
        digests = self.gather(digest.hexdigest())
        if len(set(digests)) != 1:
            raise RuntimeError(f"ranks diverged: {digests}")


LOCAL = Comm()


def setup(gpus_per_rank: int = 1) -> tuple[Comm, str, list[int] | None]:
    """(comm, this rank's first device, its GPUs when it spans several). Reads torchrun's environment."""
    cuda = torch.cuda.is_available()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    first = rank * gpus_per_rank
    gpus = list(range(first, first + gpus_per_rank)) if cuda and gpus_per_rank > 1 else None
    if cuda:
        torch.cuda.set_device(first)
    device = f"cuda:{first}" if cuda else "cpu"
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return LOCAL, device, gpus
    dist.init_process_group("nccl" if cuda else "gloo", timeout=TIMEOUT)
    return Distributed(), device, gpus


def close(comm: Comm) -> None:
    if comm.world > 1:
        comm.barrier()
        dist.destroy_process_group()
