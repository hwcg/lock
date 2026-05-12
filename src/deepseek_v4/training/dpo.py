"""
DPO (Direct Preference Optimization) Trainer。

参考：Rafailov et al. 2023, NeurIPS.

核心损失：
    L = -E[log σ(β · (log π(y_w|x) - log π_ref(y_w|x)
                    - log π(y_l|x) + log π_ref(y_l|x)))]

支持变体：
- "dpo"   : 标准 DPO
- "ipo"   : Identity PO（无 log-σ，更稳）
- "dpoplus": DPO + λ·SFT(chosen) 防止退化
- "kto"   : KTO（Kahneman-Tversky），point-wise
- "rdpo"  : Robust DPO（Mitchell 2024，减小标签噪声）

实现要点：
1. 支持 reference model（独立加载或深拷贝当前 policy）
2. 用 forward chunking 显存友好（chosen / rejected concat 后 batch）
3. 仅在响应 token 上累加 log-prob（prompt mask 由 labels=-100 标记）
4. 数值稳定：FP32 计算 log-softmax
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import DPOCollator
from deepseek_v4.data.dataset import DPODataset
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class DPOConfig(TrainerConfig):
    """DPO 训练配置。"""

    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_paths: List[str] = field(default_factory=list)
    max_prompt_len: int = 1024
    max_seq_len: int = 2048
    cache_dir: Optional[str] = "cache/datasets"

    # 模型
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: str = "checkpoints/sft/checkpoint-final"
    reference_checkpoint: Optional[str] = None    # None = 与 init 相同
    tokenizer_path: str = "checkpoints/tokenizer"

    # DPO 损失
    dpo_variant: str = "dpo"                # dpo | ipo | dpoplus | kto | rdpo
    beta: float = 0.1                        # 温度
    label_smoothing: float = 0.0             # rDPO 用
    sft_weight: float = 0.0                  # DPO+ 时的 SFT 系数
    reference_free: bool = False             # 不用参考（SimPO/IPO 简化）

    # 默认覆盖
    learning_rate: float = 5.0e-7
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_steps: int = 2000
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    metric_name: str = "eval_loss"


# ============================================================
# 工具
# ============================================================

def _shift_logits_and_labels(
    logits: torch.Tensor, labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B, S, V] / [B, S] → shift 1 step。"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def _seq_log_prob(
    logits: torch.Tensor, labels: torch.Tensor, average: bool = False,
) -> torch.Tensor:
    """
    对每个序列累加 response token 的 log-prob。

    Args:
        logits: [B, S, V]
        labels: [B, S]，-100 表示忽略
        average: True → 返回 mean log-prob（每 token 平均），用于 IPO
    Returns:
        [B] —— 序列 log p
    """
    shift_logits, shift_labels = _shift_logits_and_labels(logits, labels)
    valid = (shift_labels != -100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    per_token = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    per_token = per_token * valid.float()
    if average:
        n = valid.sum(dim=1).clamp(min=1)
        return per_token.sum(dim=1) / n
    return per_token.sum(dim=1)


# ============================================================
# DPO Trainer
# ============================================================

class DPOTrainer(BaseTrainer):
    """从 0 实现的 DPO 训练器。"""

    def __init__(
        self,
        config: DPOConfig,
        model: DeepseekV4ForCausalLM,
        tokenizer,
        ref_model: Optional[DeepseekV4ForCausalLM] = None,
    ):
        super().__init__(config=config, model=model)
        self.config: DPOConfig = config
        self.tokenizer = tokenizer
        self.ref_model = ref_model

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

    # ----- Setup -----

    def setup(self) -> None:
        super().setup()
        # ref_model：默认深拷贝当前 policy 一份
        if self.ref_model is None and not self.config.reference_free:
            unwrap = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(unwrap, "_orig_mod"):
                unwrap = unwrap._orig_mod
            logger.info("[DPO] making frozen reference model (deep copy of policy)")
            self.ref_model = copy.deepcopy(unwrap)
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()
            self.ref_model = self.ref_model.to(self.device)

    # ----- Forward / Loss -----

    @staticmethod
    def _forward_pair(
        model: nn.Module,
        chosen_ids: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        average: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次 batch 跑 chosen+rejected。"""
        all_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        all_mask = torch.cat([chosen_mask, rejected_mask], dim=0)
        out = model(input_ids=all_ids, attention_mask=all_mask, use_cache=False)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        all_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
        all_log_probs = _seq_log_prob(logits, all_labels, average=average)
        B = chosen_ids.shape[0]
        return all_log_probs[:B], all_log_probs[B:]

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        chosen_ids = batch["chosen_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]
        rejected_ids = batch["rejected_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        average = (cfg.dpo_variant == "ipo")

        # ---- policy ----
        pi_chosen, pi_rejected = self._forward_pair(
            self.model,
            chosen_ids, chosen_mask, rejected_ids, rejected_mask,
            chosen_labels, rejected_labels,
            average=average,
        )

        # ---- reference ----
        if cfg.reference_free:
            ref_chosen = torch.zeros_like(pi_chosen)
            ref_rejected = torch.zeros_like(pi_rejected)
        else:
            with torch.no_grad():
                ref_chosen, ref_rejected = self._forward_pair(
                    self.ref_model,
                    chosen_ids, chosen_mask, rejected_ids, rejected_mask,
                    chosen_labels, rejected_labels,
                    average=average,
                )

        # ---- 损失 ----
        # 偏好 log-ratio（policy 相对 reference 的提升）
        chosen_logratio = pi_chosen - ref_chosen
        rejected_logratio = pi_rejected - ref_rejected
        diff = chosen_logratio - rejected_logratio   # h(x, y_w, y_l)

        if cfg.dpo_variant == "dpo":
            # 标准 DPO
            # = -log σ(β · diff)
            # rDPO with label smoothing:
            losses = -(
                (1 - cfg.label_smoothing) * F.logsigmoid(cfg.beta * diff)
                + cfg.label_smoothing * F.logsigmoid(-cfg.beta * diff)
            )
        elif cfg.dpo_variant == "ipo":
            # IPO: (h - 1/(2β))²
            losses = (diff - 0.5 / cfg.beta) ** 2
        elif cfg.dpo_variant == "dpoplus":
            # DPO + sft loss on chosen
            losses = -F.logsigmoid(cfg.beta * diff)
        elif cfg.dpo_variant == "rdpo":
            # 已包含在 dpo + label_smoothing
            losses = -(
                (1 - cfg.label_smoothing) * F.logsigmoid(cfg.beta * diff)
                + cfg.label_smoothing * F.logsigmoid(-cfg.beta * diff)
            )
        elif cfg.dpo_variant == "kto":
            # KTO point-wise（简化版）：
            # L = -log σ(β·(pi_chosen - ref_chosen)) + log σ(-β·(pi_rejected - ref_rejected))
            # = -log σ(β·chosen_logratio) - log σ(-β·rejected_logratio)
            # 此实现等价（不分 desirable/undesirable）
            losses = -F.logsigmoid(cfg.beta * chosen_logratio) - F.logsigmoid(-cfg.beta * rejected_logratio)
        else:
            raise ValueError(f"Unknown dpo_variant: {cfg.dpo_variant}")

        loss = losses.mean()

        # 准确率：偏好胜率
        with torch.no_grad():
            acc = (diff > 0).float().mean()
            chosen_reward = cfg.beta * chosen_logratio
            rejected_reward = cfg.beta * rejected_logratio
            margin = (chosen_reward - rejected_reward).mean()

        out = {
            "loss": loss,
            "dpo_loss": loss.detach(),
            "acc": acc.detach(),
            "chosen_reward": chosen_reward.mean().detach(),
            "rejected_reward": rejected_reward.mean().detach(),
            "reward_margin": margin.detach(),
            "policy_chosen_logp": pi_chosen.mean().detach(),
            "policy_rejected_logp": pi_rejected.mean().detach(),
            "ref_chosen_logp": ref_chosen.mean().detach() if not cfg.reference_free else torch.tensor(0.0),
            "ref_rejected_logp": ref_rejected.mean().detach() if not cfg.reference_free else torch.tensor(0.0),
        }

        # SFT 正则（DPO+）
        if cfg.dpo_variant == "dpoplus" and cfg.sft_weight > 0:
            # 对 chosen 的 LM CE
            policy_out = self.model(input_ids=chosen_ids, attention_mask=chosen_mask, use_cache=False)
            logits = policy_out["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chosen_labels[..., 1:].contiguous()
            sft = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = out["loss"] + cfg.sft_weight * sft
            out["sft_loss"] = sft.detach()

        return out
