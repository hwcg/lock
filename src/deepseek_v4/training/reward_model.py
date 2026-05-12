"""
Reward Model（RM）。

模型架构：
    DeepseekV4ForCausalLM 主干 + scalar head
    取最后一个非 pad token 的 hidden state → Linear(d, 1) → 标量 reward

训练损失：
    Pairwise Bradley-Terry：
        L = -log σ(r(chosen) - r(rejected) - margin)
    可选 + LM reg loss（对 chosen 的语言建模损失）以防 RM 退化为常数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import DPOCollator
from deepseek_v4.data.dataset import DPODataset
from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM, DeepseekV4Model
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# RewardModel 主体
# ============================================================

class DeepseekV4RewardModel(nn.Module):
    """
    包装 DeepseekV4Model + scalar head。

    forward(input_ids, attention_mask) -> rewards: [B] or [B, S] (with chunking)
    """

    def __init__(self, base_model: nn.Module, hidden_size: int):
        super().__init__()
        self.base = base_model    # DeepseekV4Model（不带 LM head）
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=1.0 / (hidden_size ** 0.5))

    @classmethod
    def from_causal_lm(cls, lm: DeepseekV4ForCausalLM) -> "DeepseekV4RewardModel":
        """从 ForCausalLM 抽取 backbone 构造 RM。"""
        # 复用 lm.model（DeepseekV4Model）
        return cls(base_model=lm.model, hidden_size=lm.config.hidden_size)

    @classmethod
    def from_config(cls, config: DeepseekV4Config) -> "DeepseekV4RewardModel":
        base = DeepseekV4Model(config)
        return cls(base_model=base, hidden_size=config.hidden_size)

    # ---------- forward ----------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_per_token: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            return_per_token=False: [B]   (每序列一个 reward，取最后非 pad 位置)
            return_per_token=True:  [B,S] (每 token 都打分)
        """
        # base 返回 (hidden, past_kv)
        hidden, _ = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # [B, S, 1] → [B, S]
        rewards = self.value_head(hidden).squeeze(-1)
        if return_per_token:
            return rewards
        # 每序列取最后一个有效 token
        if attention_mask is None:
            return rewards[:, -1]
        # 找每个样本最后一个 attended 位置
        last_idx = attention_mask.long().sum(dim=1) - 1   # [B]
        last_idx = last_idx.clamp(min=0)
        return rewards.gather(1, last_idx[:, None]).squeeze(-1)


# ============================================================
# RewardConfig + RewardTrainer
# ============================================================

@dataclass
class RewardConfig(TrainerConfig):
    """RM 训练配置。"""
    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_paths: List[str] = field(default_factory=list)
    max_prompt_len: int = 1024
    max_seq_len: int = 2048
    cache_dir: Optional[str] = "cache/datasets"

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/sft/checkpoint-final"
    tokenizer_path: str = "checkpoints/tokenizer"

    # 损失
    margin: float = 0.0                     # BT 损失中的 margin
    lm_reg_weight: float = 0.0              # 对 chosen 的 LM 正则
    use_normalized: bool = True             # 标准化（chosen/rejected 共享 SoftPlus）

    # 默认覆盖（RM 需要更稳的训练）
    learning_rate: float = 5.0e-6
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_steps: int = 3000
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    metric_name: str = "eval_loss"


class RewardTrainer(BaseTrainer):
    """RM 训练器：pairwise margin loss。"""

    def __init__(self, config: RewardConfig, model: DeepseekV4RewardModel, tokenizer):
        super().__init__(config=config, model=model)
        self.config: RewardConfig = config
        self.tokenizer = tokenizer

    # ----- Dataset -----
    def get_train_dataset(self) -> Dataset:
        return DPODataset(
            paths=self.config.train_data_paths,
            tokenizer=self.tokenizer,
            max_prompt_len=self.config.max_prompt_len,
            max_seq_len=self.config.max_seq_len,
            cache_dir=self.config.cache_dir,
        )

    def get_eval_dataset(self) -> Optional[Dataset]:
        if not self.config.eval_data_paths:
            return None
        return DPODataset(
            paths=self.config.eval_data_paths,
            tokenizer=self.tokenizer,
            max_prompt_len=self.config.max_prompt_len,
            max_seq_len=self.config.max_seq_len,
            cache_dir=self.config.cache_dir,
        )

    def get_collator(self) -> Callable:
        return DPOCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            ignore_index=-100,
            pad_to_multiple_of=8,
        )

    # ----- Loss -----

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        chosen_ids = batch["chosen_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected_ids = batch["rejected_ids"]
        rejected_mask = batch["rejected_attention_mask"]

        # 把 chosen / rejected 拼成一个大 batch 一次前向（提速一倍）
        all_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)

        all_rewards = self.model(input_ids=all_ids, attention_mask=all_mask)
        B = chosen_ids.shape[0]
        r_chosen, r_rejected = all_rewards[:B], all_rewards[B:]

        # 主损失：-log σ(r_w - r_l - margin)
        diff = r_chosen - r_rejected - self.config.margin
        loss = -F.logsigmoid(diff).mean()

        # 准确率
        acc = (diff > 0).float().mean()

        out = {
            "loss": loss,
            "rm_loss": loss.detach(),
            "acc": acc.detach(),
            "reward_chosen_mean": r_chosen.detach().mean(),
            "reward_rejected_mean": r_rejected.detach().mean(),
            "reward_margin": (r_chosen - r_rejected).detach().mean(),
        }

        # 可选 LM reg：对 chosen 跑一次 LM 损失（避免 RM 学坏导致语言能力退化）
        if self.config.lm_reg_weight > 0:
            # 需要 lm_head；如果 RM base 是 DeepseekV4Model（无 head），跳过
            lm = getattr(self.model.base, "lm_head", None)
            if lm is not None:
                hidden, _ = self.model.base(
                    input_ids=chosen_ids,
                    attention_mask=chosen_mask,
                    use_cache=False,
                )
                logits = lm(hidden)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chosen_ids[..., 1:].contiguous().clone()
                # mask 掉 prompt 区域
                pad_mask = (chosen_mask[..., 1:] == 0)
                shift_labels.masked_fill_(pad_mask, -100)
                lm_loss = F.cross_entropy(
                    shift_logits.float().view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                out["loss"] = out["loss"] + self.config.lm_reg_weight * lm_loss
                out["lm_loss"] = lm_loss.detach()

        return out
