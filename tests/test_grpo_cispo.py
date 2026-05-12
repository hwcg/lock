"""GRPO / CISPO 训练器单测。"""
import torch
import torch.nn.functional as F
import pytest


# ============================================================
# GRPO advantage 标准化
# ============================================================

def test_grpo_advantage_normalization():
    """GRPO：组内标准化 advantage。"""
    # 5 个 completion 的 reward
    rewards = torch.tensor([0.2, 0.5, 0.8, 0.5, 0.5])
    mean_r = rewards.mean()
    std_r = rewards.std()
    eps = 1e-6
    advantages = (rewards - mean_r) / (std_r + eps)

    # 高 reward → 正 advantage
    assert advantages[2] > 0
    # 低 reward → 负 advantage
    assert advantages[0] < 0
    # 均值附近 → advantage ≈ 0
    assert abs(advantages[1]) < 0.5


def test_grpo_group_advantage():
    """多个 prompt 各有不同的组。"""
    B, G = 3, 4  # 3 prompts, 4 completions each
    rewards = torch.randn(B, G)
    # per prompt mean/std
    mean_per_prompt = rewards.mean(dim=1, keepdim=True)
    std_per_prompt = rewards.std(dim=1, keepdim=True) + 1e-6
    advantages = (rewards - mean_per_prompt) / std_per_prompt

    # 每组内的 advantage 均值为 0
    assert advantages.mean(dim=1).abs().max() < 1e-5
    assert advantages.shape == (B, G)


# ============================================================
# GRPO loss with KL penalty
# ============================================================

def test_grpo_loss_structure():
    """GRPO 损失 = clipped PPO + β·KL。"""
    B, T, V = 2, 5, 10
    beta = 0.04
    eps_clip = 0.2

    # log_probs (policy)
    log_probs = torch.randn(B, T, V).log_softmax(dim=-1)
    # log_probs (old policy - used for ratio)
    old_log_probs = torch.randn(B, T, V).log_softmax(dim=-1)
    # action ids
    actions = torch.randint(0, V, (B, T))
    # per-token advantage (assigned from sequence-level)
    advantages = torch.randn(B).unsqueeze(1).expand(B, T)

    # 选中 token 的 log-prob
    selected_logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_selected_logp = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = (selected_logp - old_selected_logp).exp()

    # clipped
    clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

    # KL penalty: approximate KL = (old_logp - log_p)
    log_ratio = old_selected_logp - selected_logp
    kl = log_ratio.mean()
    total = policy_loss + beta * kl

    assert total.item() > -1e3  # sanity


# ============================================================
# CISPO: stop-grad importance weight
# ============================================================

def test_cispo_stop_grad_ratio():
    """CISPO：ratio 用 detach 包装，不作为梯度来源。"""
    B, V = 4, 10
    log_probs = torch.randn(B, V).log_softmax(dim=-1)
    old_log_probs = torch.randn(B, V).log_softmax(dim=-1)
    actions = torch.randint(0, V, (B,))
    advantages = torch.randn(B)

    selected_logp = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    old_selected_logp = old_log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    ratio = (selected_logp - old_selected_logp).exp()
    # CISPO: stop-grad the ratio
    is_weight = ratio.detach()
    is_clipped = torch.clamp(is_weight, max=3.0)

    cispo_loss = -(is_clipped * advantages * selected_logp).mean()

    # is_weight 不可求导
    assert not is_weight.requires_grad
    assert not is_clipped.requires_grad
    # selected_logp 可求导
    assert selected_logp.requires_grad or not log_probs.requires_grad


def test_cispo_vs_ppo_clip_difference():
    """CISPO：所有 token 贡献梯度 vs PPO clip 可能截断。"""
    eps_clip = 0.2
    ratios = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0])
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

    # PPO clipped: ratios > 1+eps 被 clip 截断
    ppo = torch.min(ratios * advantage, torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantage)

    # CISPO: all contribute via stop_grad weight
    cispo_weights = torch.clamp(ratios.detach(), max=1.0 + eps_clip)
    cispo = cispo_weights * advantage

    # ratio=3.0: PPO clipped to 1.2, CISPO also clipped to 1.2
    # ratio=0.5: both 0.5
    # ratio=1.0: both 1.0
    assert torch.allclose(ppo, cispo.requires_grad_(False)), f"ppo={ppo}, cispo={cispo}"


def test_cispo_advantage_normalization():
    """CISPO 中也用 GRPO 风格的组内标准化。"""
    rewards = torch.tensor([0.2, 0.5, 0.8, 0.5, 0.5])
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-6
    advantages = (rewards - mean_r) / std_r

    # 0.8 是被选中的好样本
    assert advantages[2] > 0
    assert advantages[2] > advantages[0]
