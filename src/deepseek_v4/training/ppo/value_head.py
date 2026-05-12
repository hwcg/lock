"""
ValueHead：用于 PPO/Critic。

设计：
- 共享 backbone：policy 和 value 共享 transformer，最后接两个独立 head
  ★ 优点：显存省一半；缺点：学习信号互相干扰
- 独立 backbone：完全独立的 critic
  ★ 优点：训练更稳；缺点：显存翻倍

PolicyValueModel 提供统一接口给 PPOTrainer。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from deepseek_v4.modeling.model import DeepseekV4ForCausalLM, DeepseekV4Model


class ValueHead(nn.Module):
    """简单 MLP head：hidden_size → 1。"""

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.summary = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.summary.weight, std=1.0 / (hidden_size ** 0.5))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, S, H] → [B, S]
        return self.summary(self.dropout(hidden_states)).squeeze(-1)


class PolicyValueModel(nn.Module):
    """
    共享 backbone 的 Policy-Value 模型。

    forward(input_ids, ...) → {logits, values, hidden_states}
    """

    def __init__(
        self,
        policy_lm: DeepseekV4ForCausalLM,
        share_backbone: bool = True,
    ):
        super().__init__()
        self.share_backbone = share_backbone
        self.policy = policy_lm
        if share_backbone:
            self.value_head = ValueHead(policy_lm.config.hidden_size)
            self.value_backbone = None
        else:
            self.value_head = ValueHead(policy_lm.config.hidden_size)
            # 单独一份 backbone（深拷贝 policy.model）
            import copy
            self.value_backbone = copy.deepcopy(policy_lm.model)

    @property
    def config(self):
        return self.policy.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        return_values: bool = True,
        use_cache: bool = False,
        past_key_values=None,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, Any] = {}

        if self.share_backbone:
            # 跑一次 policy.model，复用 hidden 给 value
            hidden, past = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            if return_logits:
                out["logits"] = self.policy.lm_head(hidden)
            if return_values:
                out["values"] = self.value_head(hidden)
            out["past_key_values"] = past
        else:
            if return_logits:
                p_out = self.policy(
                    input_ids=input_ids, attention_mask=attention_mask,
                    past_key_values=past_key_values, use_cache=use_cache,
                )
                out["logits"] = p_out["logits"]
                out["past_key_values"] = p_out["past_key_values"]
            if return_values:
                v_hidden, _ = self.value_backbone(
                    input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                )
                out["values"] = self.value_head(v_hidden)
        return out

    # ---------- Adapter 让 generate() 可用 ----------
    def __call__(self, *args, **kwargs):
        # generate() 期望返回 {"logits": ..., "past_key_values": ...}
        # 默认只算 logits（不算 values）以加速
        kwargs.setdefault("return_values", False)
        return self.forward(*args, **kwargs)
