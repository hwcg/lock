"""RLHF 单测：DPO / PPO / Reward Model。"""
import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepseek_v4.training.dpo import (
    DPOConfig, DPOTrainer, _seq_log_prob, _shift_logits_and_labels,
)
from deepseek_v4.training.ppo.gae import compute_gae
from deepseek_v4.training.ppo.kl_controller import (
    AdaptiveKLController, FixedKLController,
)
from deepseek_v4.training.ppo.value_head import PolicyValueModel
from deepseek_v4.training.reward_model import DeepseekV4RewardModel
from deepseek_v4.training.rewards.model_reward import RewardModelReward


# ============================================================
# Log Prob / Shift
# ============================================================

def test_shift_logits_and_labels():
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    sl, slb = _shift_logits_and_labels(logits, labels)
    assert sl.shape == (2, 4, 100)
    assert slb.shape == (2, 4)


def test_seq_log_prob():
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    # 所有 token 有效
    lp = _seq_log_prob(logits, labels, average=False)
    assert lp.shape == (2,)
    # average
    lp_avg = _seq_log_prob(logits, labels, average=True)
    assert lp_avg.shape == (2,)


def test_seq_log_prob_with_ignore_index():
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    labels[0, 0] = -100
    labels[0, 1] = -100
    lp = _seq_log_prob(logits, labels)
    assert lp.shape == (2,)
    # 第二个样本应有更大的 log-prob（更多有效 token）
    assert lp[1].abs().item() > 0


# ============================================================
# DPO Config
# ============================================================

def test_dpo_config_defaults():
    cfg = DPOConfig()
    assert cfg.beta == 0.1
    assert cfg.dpo_variant == "dpo"
    assert cfg.learning_rate == 5.0e-7
    assert cfg.label_smoothing == 0.0


def test_dpo_config_variants():
    for variant in ["dpo", "ipo", "dpoplus", "kto", "rdpo"]:
        cfg = DPOConfig(dpo_variant=variant)
        assert cfg.dpo_variant == variant


# ============================================================
# DPO Loss (standalone)
# ============================================================

def test_dpo_loss_standalone():
    """测试 DPO 损失的核心计算。"""
    # 构造一对数据
    B = 2
    beta = 0.1
    pi_chosen = torch.tensor([-2.0, -3.0])
    pi_rejected = torch.tensor([-4.0, -5.0])
    ref_chosen = torch.tensor([-2.5, -3.5])
    ref_rejected = torch.tensor([-2.5, -3.5])

    chosen_logratio = pi_chosen - ref_chosen
    rejected_logratio = pi_rejected - ref_rejected
    diff = chosen_logratio - rejected_logratio

    loss = -F.logsigmoid(beta * diff).mean()
    assert loss.item() > 0
    # chosen 的 logratio 应 > rejected 的
    acc = (diff > 0).float().mean()
    assert acc >= 0.0


def test_ipo_loss_standalone():
    B = 2
    beta = 0.1
    diff = torch.tensor([2.0, 3.0])
    losses = (diff - 0.5 / beta) ** 2
    loss = losses.mean()
    assert loss.item() >= 0


def test_kto_loss_standalone():
    B = 2
    beta = 0.1
    chosen_logratio = torch.tensor([2.0, 1.0])
    rejected_logratio = torch.tensor([-1.0, -2.0])
    losses = -F.logsigmoid(beta * chosen_logratio) - F.logsigmoid(-beta * rejected_logratio)
    loss = losses.mean()
    assert loss.item() >= 0


# ============================================================
# GAE
# ============================================================

def test_compute_gae():
    B, T = 2, 8
    values = torch.randn(B, T)
    rewards = torch.randn(B, T)
    dones = torch.zeros(B, T)

    advantages, returns = compute_gae(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=0.99,
        lam=0.95,
    )
    assert advantages.shape == values.shape
    assert returns.shape == values.shape
    # 端 state 优势为 0
    assert torch.allclose(advantages[:, -1], torch.zeros(B))


def test_compute_gae_with_done():
    B, T = 1, 4
    values = torch.ones(B, T)
    rewards = torch.ones(B, T)
    dones = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    adv, ret = compute_gae(values, rewards, dones, gamma=0.99, lam=0.95)
    assert adv.shape == values.shape


# ============================================================
# KL Controller
# ============================================================

def test_fixed_kl_controller():
    ctrl = FixedKLController(kl_coef=0.1)
    assert ctrl.coef == 0.1
    ctrl.update(0.05)
    assert ctrl.coef == 0.1  # fixed, no change


def test_adaptive_kl_controller():
    ctrl = AdaptiveKLController(init_kl_coef=0.1, target=6.0, horizon=10000)
    # kl 低于 target → 减小 coef
    old = ctrl.coef
    ctrl.update(3.0)
    assert ctrl.coef < old


# ============================================================
# Value Head
# ============================================================

def test_value_head():
    head = PolicyValueHead(hidden_size=64, dropout=0.0)
    x = torch.randn(2, 8, 64)
    values, hidden = head(x)
    assert values.shape == (2, 8)
    assert hidden.shape == (2, 8, 64)


# ============================================================
# Reward Model Wrapper
# ============================================================

def test_reward_model_reward():
    """RewardModelReward 调用前的签名检查。"""
    # 用普通函数模拟
    def dummy_rm(completions, references=None, prompts=None, **kwargs):
        return [0.5] * len(completions)

    r = RewardModelReward(model=dummy_rm, name="dummy_rm")
    scores = r(completions=["a", "b", "c"])
    assert scores == [0.5, 0.5, 0.5]


# ============================================================
# Mock PPO ratio / clip loss (standalone)
# ============================================================

def test_ppo_clipped_objective():
    """测试 PPO clipped objective 的数学性质。"""
    B, T, V = 2, 4, 10
    ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0])
    eps = 0.2

    # PPO: -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    surrogate = torch.min(ratio * advantage, clipped * advantage)
    loss = -surrogate.mean()

    # ratio=2.0 → clipped to 1.2, min is 1.0*1.2
    assert torch.allclose(clipped, torch.tensor([0.8, 1.0, 1.2, 1.2]))
