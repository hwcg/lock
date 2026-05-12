"""
蒸馏 loss（从 0 实现）。

包含：
- kd_loss          Forward KL: KL(teacher || student) —— 标准 distillation（Hinton 2015）
- reverse_kd_loss  Reverse KL: KL(student || teacher) —— mode-seeking，GKD 2023 用
- jsd_loss         JS divergence —— 介于 forward/reverse 之间
- topk_kd_loss     Top-K 蒸馏：仅在 teacher top-K logits 上比对（省通信）

所有损失支持：
- temperature 控制软化程度
- token-level mask 与 reduction
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ============================================================
# Helpers
# ============================================================

def _masked_mean(loss_per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 loss_per_token 用 mask 加权平均。"""
    denom = mask.sum().clamp(min=1.0)
    return (loss_per_token * mask).sum() / denom


# ============================================================
# Forward KL
# ============================================================

def kd_loss(
    student_logits: torch.Tensor,    # [B, T, V]
    teacher_logits: torch.Tensor,    # [B, T, V]
    mask: Optional[torch.Tensor] = None,   # [B, T]
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Forward KL: KL(teacher || student)。

    L = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
      = T^2 * sum p_t * (log p_t - log p_s)

    Hinton 标准蒸馏。乘 T^2 保持梯度规模与无温度时相当。
    """
    T = temperature
    t_log_probs = F.log_softmax(teacher_logits.float() / T, dim=-1)
    s_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
    t_probs = t_log_probs.exp()
    per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)   # [B, T]
    per_token = per_token * (T * T)
    if mask is not None:
        return _masked_mean(per_token, mask.float())
    return per_token.mean()


# ============================================================
# Reverse KL
# ============================================================

def reverse_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Reverse KL: KL(student || teacher) — mode-seeking，GKD/Generalized Knowledge Distillation。

    L = T^2 * KL(softmax(z_s/T) || softmax(z_t/T))
      = T^2 * sum p_s * (log p_s - log p_t)
    """
    T = temperature
    t_log_probs = F.log_softmax(teacher_logits.float() / T, dim=-1)
    s_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
    s_probs = s_log_probs.exp()
    per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
    per_token = per_token * (T * T)
    if mask is not None:
        return _masked_mean(per_token, mask.float())
    return per_token.mean()


# ============================================================
# JS Divergence
# ============================================================

def jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    JS divergence (skew-symmetric)。

    M = alpha * p_s + (1-alpha) * p_t
    JSD = alpha * KL(p_s || M) + (1-alpha) * KL(p_t || M)
    """
    T = temperature
    t_log_probs = F.log_softmax(teacher_logits.float() / T, dim=-1)
    s_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
    t_probs = t_log_probs.exp()
    s_probs = s_log_probs.exp()
    m_probs = alpha * s_probs + (1 - alpha) * t_probs
    m_log_probs = (m_probs.clamp(min=1e-12)).log()
    kl_s = (s_probs * (s_log_probs - m_log_probs)).sum(dim=-1)
    kl_t = (t_probs * (t_log_probs - m_log_probs)).sum(dim=-1)
    per_token = (alpha * kl_s + (1 - alpha) * kl_t) * (T * T)
    if mask is not None:
        return _masked_mean(per_token, mask.float())
    return per_token.mean()


# ============================================================
# Top-K KD
# ============================================================

def topk_kd_loss(
    student_logits: torch.Tensor,    # [B, T, V]
    teacher_topk_values: torch.Tensor,   # [B, T, K]  teacher 已选 top-K 的 logits
    teacher_topk_indices: torch.Tensor,  # [B, T, K]
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-K 蒸馏：teacher 只发送 top-K logits（节省带宽）。

    L = T^2 * KL( renorm(softmax(teacher_topk/T))  ||  softmax(student_topk_logits/T) )

    student 部分按 teacher 提供的 indices 取出对应 K 个 logits 再做 softmax。
    """
    T = temperature
    # 取 student 在相同 indices 上的 logits
    s_topk_logits = student_logits.gather(-1, teacher_topk_indices)  # [B, T, K]

    t_log_probs = F.log_softmax(teacher_topk_values.float() / T, dim=-1)
    s_log_probs = F.log_softmax(s_topk_logits.float() / T, dim=-1)
    t_probs = t_log_probs.exp()
    per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
    per_token = per_token * (T * T)
    if mask is not None:
        return _masked_mean(per_token, mask.float())
    return per_token.mean()
