"""
Teacher 模型包装。

支持：
- 加载冻结的 teacher checkpoint
- 在线推理（每个 batch 即算）
- 离线缓存（预先把 teacher logits 算好存 .pt，训练时直接读）
- Top-K 截断（节省存储 / 通信）
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


class TeacherWrapper:
    """
    冻结的 teacher，提供 logits 接口。

    Args:
        teacher_model:   一个 nn.Module，forward(input_ids, attention_mask) -> {"logits": ...}
        topk:            如果 > 0，则只返回 top-K logits + indices
        temperature:     softmax 温度（仅影响 top-K 选取的概率，logits 仍原始返回）
        device:          强制设备
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        topk: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = teacher_model
        self.topk = topk
        self.device = device or next(teacher_model.parameters()).device
        self.dtype = dtype
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = out["logits"] if isinstance(out, dict) else out.logits
        if self.topk and self.topk > 0:
            topk_values, topk_indices = logits.topk(self.topk, dim=-1)
            return {
                "topk_values":  topk_values,
                "topk_indices": topk_indices,
            }
        return {"logits": logits}

    __call__ = forward
