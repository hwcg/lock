"""
Gradient Checkpointing 工具。

通过 monkey-patch 把每一层 DecoderLayer.forward 替换为 checkpoint 版本，
以训练显存换取一倍激活值占用，标准做法。
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _make_checkpoint_forward(module: nn.Module, use_reentrant: bool = False) -> Callable:
    """生成一个对 module.forward 包了 checkpoint 的新 forward。"""
    original_forward = module.forward

    @wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        if not module.training:
            return original_forward(*args, **kwargs)

        if kwargs:
            # checkpoint 不支持 kwargs：闭包捕获
            def fn(*pargs):
                return original_forward(*pargs, **kwargs)
            return checkpoint(fn, *args, use_reentrant=use_reentrant)
        return checkpoint(original_forward, *args, use_reentrant=use_reentrant)

    return checkpointed_forward


def enable_gradient_checkpointing(
    model: nn.Module,
    layer_attr: str = "layers",
    skip_first_n: int = 0,
    skip_last_n: int = 0,
    use_reentrant: bool = False,
) -> int:
    """
    给 model.{layer_attr} 中的每一个 layer 启用 gradient checkpointing。

    Args:
        model:        顶层模型（如 DeepseekV4Model 或 ForCausalLM）
        layer_attr:   layer 容器属性名（V4 → "layers"）
        skip_first_n: 跳过前 N 层（第一层激活值小，CKPT 性价比低）
        skip_last_n:  跳过最后 N 层
        use_reentrant: PyTorch 推荐 False（更稳定）
    Returns:
        实际启用 checkpoint 的层数
    """
    # 兼容 ForCausalLM → .model.layers
    if not hasattr(model, layer_attr) and hasattr(model, "model"):
        return enable_gradient_checkpointing(
            model.model, layer_attr=layer_attr,
            skip_first_n=skip_first_n, skip_last_n=skip_last_n,
            use_reentrant=use_reentrant,
        )

    layers = getattr(model, layer_attr)
    n = len(layers)
    enabled = 0
    for idx, layer in enumerate(layers):
        if idx < skip_first_n or idx >= n - skip_last_n:
            continue
        layer.forward = _make_checkpoint_forward(layer, use_reentrant=use_reentrant)
        enabled += 1
    return enabled
