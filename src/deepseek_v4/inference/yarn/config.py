"""
YaRN 配置 + 三种 RoPE 缩放方法。

参考：
- Linear scaling (Chen et al. 2023, "Position Interpolation"):
      theta_i = theta_i / s
- NTK-aware (bloc97 2023):
      theta_i = (theta_i * base) -> 把 base 缩放为 base * s^(d/(d-2))
- YaRN (Peng et al. 2023):
      插值 + 外推 + 线性过渡 + temperature scaling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RopeScalingMethod(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    NTK = "ntk"
    YARN = "yarn"
    DYNAMIC_NTK = "dynamic_ntk"


@dataclass
class YarnConfig:
    """
    YaRN 配置。

    Args:
        method:                 缩放方法
        factor:                 总缩放因子（target / original）
        original_max_position:  训练时的原始最大长度（如 65536）
        target_max_position:    希望支持的目标长度（如 1048576）
        beta_fast:              快速变化频段的边界（YaRN 默认 32）
        beta_slow:              缓慢变化频段的边界（YaRN 默认 1）
        mscale:                 attention 输出 logit 的温度缩放（V4 默认关 = 1.0）
        mscale_all_dim:         是否对所有维度做 mscale（默认 False）
    """
    method: RopeScalingMethod = RopeScalingMethod.YARN
    factor: float = 4.0
    original_max_position: int = 65536
    target_max_position: int = 262144
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: bool = False

    def to_rope_scaling(self) -> Dict[str, Any]:
        """转为 modeling 中的 rope_scaling dict。"""
        if self.method == RopeScalingMethod.NONE:
            return {}
        d = {
            "type": self.method.value,
            "factor": self.factor,
            "original_max_position_embeddings": self.original_max_position,
        }
        if self.method == RopeScalingMethod.YARN:
            d.update({
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "mscale": self.mscale,
                "mscale_all_dim": self.mscale_all_dim,
            })
        return d


def build_rope_scaling(
    method: str = "yarn",
    factor: float = 4.0,
    original_max_position: int = 65536,
    **kwargs,
) -> Dict[str, Any]:
    """便捷构造函数。"""
    cfg = YarnConfig(
        method=RopeScalingMethod(method),
        factor=factor,
        original_max_position=original_max_position,
        **kwargs,
    )
    return cfg.to_rope_scaling()
