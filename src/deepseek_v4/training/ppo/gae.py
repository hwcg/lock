"""Generalized Advantage Estimation。"""
from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,        # [B, T]
    values: torch.Tensor,         # [B, T]
    mask: torch.Tensor,           # [B, T]，1=有效
    gamma: float = 1.0,
    lam: float = 0.95,
    bootstrap: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE-λ 计算优势与回报。

    Args:
        rewards: 每 token 的即时回报（含 KL 惩罚）
        values:  每 token 的 value 估计
        mask:    1 = 有效 token, 0 = padding/已结束
        bootstrap: 是否在最后一步用 V 自举（PPO 通常 False，因为 episode 已结束）
    Returns:
        advantages: [B, T]
        returns:    [B, T] = advantages + values
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    # 末位 next_value：episode 内已结束 → 0
    if bootstrap:
        next_values = values[:, -1].clone()
    else:
        next_values = torch.zeros_like(values[:, -1])

    for t in reversed(range(T)):
        m = mask[:, t]
        next_v = next_values if t == T - 1 else values[:, t + 1]
        delta = rewards[:, t] + gamma * next_v * m - values[:, t]
        last_gae = delta + gamma * lam * last_gae * m
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_advantages_with_whitening(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    whiten: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GAE + advantage whitening（PPO 标配）。"""
    advantages, returns = compute_gae(rewards, values, mask, gamma=gamma, lam=lam)
    if whiten:
        m = mask.bool()
        a = advantages[m]
        mean = a.mean()
        std = a.std().clamp(min=eps)
        advantages = (advantages - mean) / std
        # padding 位置归零
        advantages = advantages * mask
    return advantages, returns
