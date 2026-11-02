"""GRPO objective with per-token KL to the initial policy; no ε clipping (PLAN §3.7)."""
from __future__ import annotations

import torch


def per_token_kl(logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """Unbiased per-token KL(π_θ || π_ref) estimate for tokens sampled from π_θ, as in GRPO."""
    diff = ref_logps - logps
    return torch.exp(diff) - diff - 1


def grpo_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Tensors are [trajectories, tokens]; advantages is [trajectories]. Each trajectory is averaged
    over its tokens, then trajectories are averaged."""
    mask = mask.to(logps.dtype)
    ratio = torch.exp(logps - old_logps)
    kl = per_token_kl(logps, ref_logps)
    per_token = -ratio * advantages.unsqueeze(1) + beta * kl
    per_trajectory = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
    stats = {"kl": float(((kl * mask).sum(1) / mask.sum(1).clamp(min=1)).mean().detach())}
    return per_trajectory.mean(), stats
