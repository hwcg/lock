"""KL 控制器：固定 / 自适应。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FixedKLController:
    """固定系数。"""
    kl_coef: float = 0.1

    def update(self, current_kl: float, n_steps: int = 1) -> None:
        pass

    @property
    def value(self) -> float:
        return self.kl_coef


@dataclass
class AdaptiveKLController:
    """
    InstructGPT / TRL 风格自适应 KL 控制器。

    每次见到 current_kl 后，按比例缩放：
        e = clip(current_kl / target - 1, -0.2, 0.2)
        kl_coef *= 1 + e * (n_steps / horizon)
    """
    init_kl_coef: float = 0.1
    target_kl: float = 0.1
    horizon: int = 10000

    def __post_init__(self):
        self._value = self.init_kl_coef

    def update(self, current_kl: float, n_steps: int = 1) -> None:
        from numpy import clip
        proportional_error = clip(current_kl / max(self.target_kl, 1e-6) - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self._value = float(self._value * mult)

    @property
    def value(self) -> float:
        return self._value
