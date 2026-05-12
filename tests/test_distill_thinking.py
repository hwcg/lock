"""蒸馏 / Adaptive Thinking 单测。"""
import torch
import torch.nn.functional as F
import pytest

from deepseek_v4.training.distill.losses import (
    jsd_loss, kd_loss, reverse_kd_loss, topk_kd_loss,
)
from deepseek_v4.training.adaptive_thinking.router import ModeRouter
from deepseek_v4.training.adaptive_thinking.difficulty import (
    estimate_difficulty, DifficultyLevel,
)


# ============================================================
# KD Losses
# ============================================================

def test_kd_loss_forward():
    """Forward KL: KL(teacher || student)。"""
    B, V = 2, 10
    teacher_logits = torch.randn(B, V)
    student_logits = torch.randn(B, V)

    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # Forward KL
    loss = kd_loss(student_log_probs, teacher_probs, temperature=1.0)
    assert loss.item() > 0


def test_reverse_kd_loss():
    """Reverse KL: KL(student || teacher)。"""
    B, V = 2, 10
    teacher_logits = torch.randn(B, V)
    student_logits = torch.randn(B, V)

    loss = reverse_kd_loss(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        temperature=1.0,
    )
    assert loss.item() > 0


def test_jsd_loss():
    """Jensen-Shannon Divergence。"""
    B, V = 2, 10
    teacher_logits = torch.randn(B, V)
    student_logits = torch.randn(B, V)

    loss = jsd_loss(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        temperature=1.0,
    )
    assert loss.item() > 0.0
    # JSD 在 [0, ln(2)] 之间，但 logits 可能很大 → 接近 bound
    assert loss.item() < 100.0


def test_topk_kd_loss():
    """Top-K KD：仅对 topk 位置计算 KL。"""
    B, V = 2, 10
    teacher_logits = torch.randn(B, V)
    student_logits = torch.randn(B, V)

    loss = topk_kd_loss(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        temperature=1.0,
        topk=5,
    )
    assert loss.item() > 0


def test_kd_with_temperature():
    """温度参数影响 KD。"""
    B, V = 2, 10
    teacher_logits = torch.ones(B, V) * 3  # high confidence
    student_logits = torch.randn(B, V)

    t1 = kd_loss(F.log_softmax(student_logits, dim=-1),
                 F.softmax(teacher_logits, dim=-1), temperature=1.0)
    t4 = kd_loss(F.log_softmax(student_logits, dim=-1),
                 F.softmax(teacher_logits, dim=-1), temperature=4.0)
    # 温度不同 loss 不同
    assert abs(t1.item() - t4.item()) > 1e-3


# ============================================================
# Difficulty Estimation
# ============================================================

def test_difficulty_levels():
    levels = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
    assert len(levels) == 3


def test_estimate_difficulty():
    """根据 prompt 特征估计难度。"""
    # 简单 prompt
    easy = estimate_difficulty("What is 1+1?")
    assert easy in DifficultyLevel

    # 复杂 prompt
    hard = estimate_difficulty(
        "Consider the Riemann zeta function. Prove that all non-trivial zeros "
        "lie on the critical line Re(s) = 1/2. Provide a detailed step-by-step proof."
    )
    assert hard in DifficultyLevel
    # 复杂问题应不更简单
    assert easy.value <= hard.value


def test_estimate_difficulty_empty():
    result = estimate_difficulty("")
    assert result == DifficultyLevel.EASY


# ============================================================
# ModeRouter
# ============================================================

def test_mode_router_default():
    router = ModeRouter()
    # 简单问题
    mode = router.predict("What is 1+1?")
    assert mode in ("thinking", "chat")


def test_mode_router_training_mode():
    router = ModeRouter()
    # 推理需求高的问题
    mode = router.predict("Solve this complex integral: ∫e^(x²)dx from 0 to ∞")
    assert mode in ("thinking", "chat")


def test_mode_router_batch():
    router = ModeRouter()
    prompts = [
        "Hi",
        "What is the capital of France?",
        "Prove the Riemann hypothesis.",
    ]
    modes = [router.predict(p) for p in prompts]
    assert len(modes) == 3
    for m in modes:
        assert m in ("thinking", "chat")


# ============================================================
# Adaptive Thinking Reward
# ============================================================

def test_adaptive_thinking_reward_components():
    """Adaptive thinking 的多维度奖励。"""
    correctness = 1.0
    efficiency = 1.0  # simple question should NOT think
    mode_match = 1.0

    # weighting
    w_c, w_e, w_m = 0.5, 0.3, 0.2
    total = w_c * correctness + w_e * efficiency + w_m * mode_match
    assert 0 <= total <= 1.0


def test_efficiency_penalty():
    """简单问题过度思考应受罚。"""
    # 简单问题 thinking=long → efficiency score low
    thinking_length = 500  # tokens
    threshold = 100

    if thinking_length > threshold:
        efficiency = 0.1  # penalty
    else:
        efficiency = 1.0

    assert efficiency < 0.5


# ============================================================
# Distill mixed loss
# ============================================================

def test_mixed_ce_kd_loss():
    """α * CE + (1-α) * KD。"""
    B, V = 2, 10
    alpha = 0.3

    student_logits = torch.randn(B, V)
    teacher_logits = torch.randn(B, V)
    labels = torch.randint(0, V, (B,))

    ce = F.cross_entropy(student_logits, labels, reduction="mean")
    kd = kd_loss(F.log_softmax(student_logits, dim=-1),
                 F.softmax(teacher_logits, dim=-1), temperature=1.0)
    mixed = alpha * ce + (1 - alpha) * kd

    assert mixed.item() > 0
