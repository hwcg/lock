"""LoRA 配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from deepseek_v4.utils.config import BaseConfig


@dataclass
class LoRAConfig(BaseConfig):
    """
    LoRA 训练配置。

    Args:
        r:               低秩维度
        lora_alpha:      缩放系数（等效 lr scale = alpha / r）
        lora_dropout:    LoRA 输入 dropout
        target_modules:  目标模块名 pattern 列表（substring 匹配 named_modules）
        modules_to_save: 这些模块完整训练（典型：lm_head, embed_tokens）
        bias:            "none" | "all" | "lora_only"
        use_dora:        是否启用 DoRA（Liu et al. 2024）
        use_rslora:      是否启用 rsLoRA 缩放（lora_alpha / sqrt(r)）
        init_lora_weights: 是否对 A 用 kaiming_uniform，B 用 zeros 初始化
        target_experts:  是否对 MoE experts 应用（默认 True）
        target_grouped:  是否对 GroupedLinear 应用（默认 True）
    """
    r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_a_proj", "q_b_proj", "kv_proj", "o_b_proj",
    ])
    modules_to_save: List[str] = field(default_factory=list)
    bias: str = "none"               # none | all | lora_only
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: bool = True
    target_experts: bool = False     # MoE experts 默认不动
    target_grouped: bool = True      # GroupedLinear 默认动
    target_router: bool = False      # router weight 默认不动

    @property
    def scaling(self) -> float:
        """lora 缩放因子。"""
        import math
        if self.use_rslora:
            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r
