import inspect

import pytest
import torch

from avrsi.grpo import grpo_loss, per_token_kl


def test_no_clipping_parameter_exists():
    names = set(inspect.signature(grpo_loss).parameters)
    assert not names & {"epsilon", "eps", "clip", "clip_range"}


def test_ratio_is_unclipped():
    logps, old = torch.tensor([[2.0]]), torch.tensor([[0.0]])
    loss, _ = grpo_loss(logps, old, logps, torch.tensor([1.0]), torch.ones(1, 1), beta=0.0)
    assert float(loss) == pytest.approx(-float(torch.exp(torch.tensor(2.0))))  # 7.39 would be 1.2 under ε=0.2


def test_ratio_one_gradient_is_policy_gradient():
    logps = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]], requires_grad=True)
    adv = torch.tensor([1.0, -2.0])
    loss, _ = grpo_loss(logps, logps.detach(), logps.detach(), adv, torch.ones(2, 2), beta=0.0)
    loss.backward()
    # Each token: -adv / tokens / trajectories.
    assert logps.grad.flatten().tolist() == pytest.approx([-0.25, -0.25, 0.5, 0.5])


def test_per_token_kl_zero_at_reference_and_positive_elsewhere():
    x = torch.tensor([-1.0, -2.0])
    assert per_token_kl(x, x).abs().max() == 0
    assert (per_token_kl(x, x + 0.5) > 0).all() and (per_token_kl(x, x - 0.5) > 0).all()


def test_kl_gradient_pulls_toward_reference():
    logps = torch.tensor([[-3.0]], requires_grad=True)
    ref = torch.tensor([[-1.0]])
    loss, stats = grpo_loss(logps, logps.detach(), ref, torch.zeros(1), torch.ones(1, 1), beta=0.01)
    loss.backward()
    assert logps.grad.item() < 0 and stats["kl"] > 0  # descent raises logps toward ref


def test_mask_ignores_padding_and_normalises_per_trajectory():
    logps = torch.tensor([[-1.0, -1.0], [-1.0, 99.0]], requires_grad=True)
    mask = torch.tensor([[1, 1], [1, 0]])
    loss, _ = grpo_loss(logps, logps.detach(), logps.detach(), torch.ones(2), mask, beta=0.0)
    loss.backward()
    assert loss.item() == pytest.approx(-1.0)
    assert logps.grad.flatten().tolist() == pytest.approx([-0.25, -0.25, -0.5, 0.0])


def test_per_token_kl_weight_scales_with_beta():
    logps, ref = torch.tensor([[-2.0]]), torch.tensor([[-1.0]])
    a, _ = grpo_loss(logps, logps, ref, torch.zeros(1), torch.ones(1, 1), beta=0.01)
    b, _ = grpo_loss(logps, logps, ref, torch.zeros(1), torch.ones(1, 1), beta=0.02)
    assert float(b) == pytest.approx(2 * float(a))
