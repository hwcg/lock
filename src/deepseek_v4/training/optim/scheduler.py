"""
LR Scheduler（从 0 实现）：

- ConstantWarmupScheduler
- LinearWarmupScheduler
- CosineWarmupScheduler
- PolynomialWarmupScheduler
- WSDScheduler (Warmup-Stable-Decay，MiniCPM/SmolLM 用法)

所有 scheduler 都按 step（不是 epoch）粒度推进。
统一接口：
    scheduler = build_scheduler(...)
    for step in range(total_steps):
        loss.backward()
        optimizer.step()
        scheduler.step()
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LRScheduler(LambdaLR):
    """通用基类：保存配置、便于 state_dict 序列化。"""

    def __init__(self, optimizer: Optimizer, lr_lambda, last_epoch: int = -1):
        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def _warmup(step: int, warmup_steps: int) -> float:
    """warmup factor ∈ [0, 1]。"""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


# ------------------------------------------------------------------------
# Cosine
# ------------------------------------------------------------------------

class CosineWarmupScheduler(LRScheduler):
    """
    Warmup → Cosine decay → 最低 lr_min。

    在 [0, warmup_steps] 线性升至 lr_max；
    在 (warmup_steps, total_steps] 余弦降至 lr_min。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        def fn(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step, warmup_steps)
            decay_steps = max(total_steps - warmup_steps, 1)
            progress = (step - warmup_steps) / decay_steps
            progress = min(progress, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cos

        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ------------------------------------------------------------------------
# Linear
# ------------------------------------------------------------------------

class LinearWarmupScheduler(LRScheduler):
    """Warmup → 线性降。"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step, warmup_steps)
            decay = max(total_steps - warmup_steps, 1)
            progress = (step - warmup_steps) / decay
            progress = min(progress, 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ------------------------------------------------------------------------
# Polynomial
# ------------------------------------------------------------------------

class PolynomialWarmupScheduler(LRScheduler):
    """Warmup → 多项式降。"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        def fn(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step, warmup_steps)
            decay = max(total_steps - warmup_steps, 1)
            progress = (step - warmup_steps) / decay
            progress = min(progress, 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * ((1.0 - progress) ** power)

        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ------------------------------------------------------------------------
# Constant with Warmup
# ------------------------------------------------------------------------

class ConstantWarmupScheduler(LRScheduler):
    """Warmup → 常数。"""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        def fn(step: int) -> float:
            return _warmup(step, warmup_steps)
        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ------------------------------------------------------------------------
# WSD（Warmup-Stable-Decay）
# ------------------------------------------------------------------------

class WSDScheduler(LRScheduler):
    """
    Warmup-Stable-Decay 调度（MiniCPM 提出）。

    阶段 1：[0, warmup_steps)              → 线性升 0 → lr_max
    阶段 2：[warmup_steps, decay_start)    → 保持 lr_max
    阶段 3：[decay_start, total_steps)     → 1 - sqrt 衰减到 lr_min

    优势：可在 stable 段任意截断而仅在最后做短 decay 收敛，复用早期 checkpoint。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_ratio: float = 0.1,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        decay_steps = int(total_steps * decay_ratio)
        decay_start = total_steps - decay_steps

        def fn(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step, warmup_steps)
            if step < decay_start:
                return 1.0
            # 1 - sqrt 衰减
            progress = (step - decay_start) / max(decay_steps, 1)
            progress = min(progress, 1.0)
            decay_factor = 1.0 - math.sqrt(progress)
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay_factor

        super().__init__(optimizer, lr_lambda=fn, last_epoch=last_epoch)


# ------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------

SCHEDULER_REGISTRY = {
    "constant":   ConstantWarmupScheduler,
    "linear":     LinearWarmupScheduler,
    "cosine":     CosineWarmupScheduler,
    "polynomial": PolynomialWarmupScheduler,
    "wsd":        WSDScheduler,
}


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    warmup_steps: int,
    total_steps: int,
    **kwargs: Any,
) -> LRScheduler:
    """统一工厂：name in {constant, linear, cosine, polynomial, wsd}。"""
    cls = SCHEDULER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(SCHEDULER_REGISTRY)}")
    if cls is ConstantWarmupScheduler:
        return cls(optimizer, warmup_steps=warmup_steps, **kwargs)
    return cls(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, **kwargs)
