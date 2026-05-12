"""
动态把 YaRN 配置注入已构造的模型（不重训）。

关键点：
- 修改 config 的 rope_scaling 与 max_position_embeddings
- 重新计算 RotaryEmbedding 的 inv_freq buffer（main / compress 两路）
- 兼容 V4 的 yarn / deepseek_yarn 实现（已在 modeling.compute_rope_inv_freq 中）
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from deepseek_v4.modeling.model import (
    DeepseekV4Config, DeepseekV4ForCausalLM, DeepseekV4Model,
    DeepseekV4RotaryEmbedding, compute_rope_inv_freq,
)
from deepseek_v4.inference.yarn.config import YarnConfig
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 1. 修改 config
# ============================================================

def apply_yarn_to_config(config: DeepseekV4Config, yarn: YarnConfig) -> DeepseekV4Config:
    """
    修改 config，使其使用 YaRN 缩放。

    返回同一个对象（in-place 修改）。
    """
    if yarn.method.value == "none":
        return config

    config.rope_scaling = yarn.to_rope_scaling()
    # 更新最大长度
    new_max = max(
        config.max_position_embeddings,
        int(yarn.original_max_position * yarn.factor),
        yarn.target_max_position,
    )
    config.max_position_embeddings = new_max

    # 重新构造 rope_parameters（main + compress）
    partial = config.qk_rope_head_dim / config.head_dim
    rope_extra = {k: v for k, v in config.rope_scaling.items() if k != "type"}
    rope_type = config.rope_scaling.get("type", "yarn")
    config.rope_parameters = {
        "main": {
            "rope_type": rope_type,
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": partial,
            **rope_extra,
        },
        "compress": {
            "rope_type": rope_type,
            "rope_theta": config.compress_rope_theta,
            "partial_rotary_factor": partial,
            **rope_extra,
        },
    }
    logger.info(
        f"[YaRN] config updated: method={yarn.method.value}, factor={yarn.factor}, "
        f"max_position={new_max}"
    )
    return config


# ============================================================
# 2. 重新计算所有 RotaryEmbedding 的 inv_freq buffers
# ============================================================

def recompute_inv_freq_buffers(model: nn.Module, config: DeepseekV4Config) -> int:
    """
    遍历所有 DeepseekV4RotaryEmbedding，重算其 inv_freq / attention_scaling。

    返回更新的模块数。
    """
    count = 0
    for module in model.modules():
        if isinstance(module, DeepseekV4RotaryEmbedding):
            module.config = config
            module.layer_types = [
                k for k, v in config.rope_parameters.items() if isinstance(v, dict)
            ]
            for lt in module.layer_types:
                params = config.rope_parameters[lt]
                module.rope_type[lt] = params["rope_type"]
                inv_freq, attn_scaling = compute_rope_inv_freq(config, layer_type=lt)
                buf_name = f"{lt}_inv_freq"
                existing = getattr(module, buf_name, None)
                if existing is not None:
                    inv_freq = inv_freq.to(existing.device, dtype=existing.dtype)
                module.register_buffer(buf_name, inv_freq, persistent=False)
                setattr(module, f"{lt}_attention_scaling", attn_scaling)
            count += 1
    return count


# ============================================================
# 3. 一站式 API
# ============================================================

def apply_yarn_to_model(
    model: nn.Module,
    yarn: YarnConfig,
) -> nn.Module:
    """
    给已加载好权重的 model 注入 YaRN（不动权重）。

    工作：
    1. 修改 config
    2. 重算 RotaryEmbedding inv_freq buffers
    """
    target_config: Optional[DeepseekV4Config] = None
    if isinstance(model, DeepseekV4ForCausalLM):
        target_config = model.config
    elif isinstance(model, DeepseekV4Model):
        target_config = model.config
    else:
        target_config = getattr(model, "config", None)

    if target_config is None:
        raise RuntimeError("Cannot locate DeepseekV4Config on model")

    apply_yarn_to_config(target_config, yarn)
    n_updated = recompute_inv_freq_buffers(model, target_config)
    logger.info(f"[YaRN] {n_updated} RotaryEmbedding(s) re-initialized")
    return model
