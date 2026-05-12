"""
Distillation Trainer。

支持：
- on-policy 蒸馏（在线 teacher forward）
- 混合 CE + KD（alpha*CE + (1-alpha)*KD）
- forward KL / reverse KL / JSD / top-K
- 支持任意 (input_ids, labels) 风格数据集（直接复用 SFTDataset / PretrainDataset）
- 仅在 labels != ignore_index 的位置计算 KD（保持与 CE 一致）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from deepseek_v4.data.collator import SFTCollator
from deepseek_v4.data.dataset import PretrainDataset, SFTDataset
from deepseek_v4.modeling.model import DeepseekV4ForCausalLM
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.training.distill.losses import (
    jsd_loss, kd_loss, reverse_kd_loss, topk_kd_loss,
)
from deepseek_v4.training.distill.teacher import TeacherWrapper
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DistillConfig(TrainerConfig):
    """蒸馏配置。"""
    # 数据
    train_data_paths: List[str] = field(default_factory=list)
    eval_data_paths: List[str] = field(default_factory=list)
    dataset_type: str = "sft"                  # sft | pretrain
    max_seq_len: int = 4096
    cache_dir: Optional[str] = "cache/datasets"
    text_field: str = "text"

    # Student
    model_config_path: str = "configs/model/mini_2b.json"
    init_from_checkpoint: Optional[str] = None
    tokenizer_path: str = "checkpoints/tokenizer"

    # Teacher
    teacher_config_path: Optional[str] = None     # None: 用 student 同结构
    teacher_checkpoint: str = "checkpoints/teacher/model.safetensors"
    teacher_topk: int = 0                          # 0 = 全 vocab

    # 蒸馏
    kd_type: str = "forward_kl"     # forward_kl | reverse_kl | jsd | topk
    temperature: float = 2.0
    alpha_ce: float = 0.5                          # CE 权重；KD 权重 = 1 - alpha_ce
    jsd_alpha: float = 0.5

    # 默认覆盖
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    max_steps: int = 10000


class DistillTrainer(BaseTrainer):
    """蒸馏 Trainer。"""

    def __init__(
        self,
        config: DistillConfig,
        student: DeepseekV4ForCausalLM,
        teacher: TeacherWrapper,
        tokenizer,
    ):
        super().__init__(config=config, model=student)
        self.config: DistillConfig = config
        self.tokenizer = tokenizer
        self.teacher = teacher

    # ----- Dataset / Collator -----

    def get_train_dataset(self) -> Dataset:
        if self.config.dataset_type == "sft":
            return SFTDataset(
                paths=self.config.train_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                cache_dir=self.config.cache_dir,
            )
        return PretrainDataset(
            paths=self.config.train_data_paths,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            cache_dir=self.config.cache_dir,
            text_field=self.config.text_field,
        )

    def get_eval_dataset(self) -> Optional[Dataset]:
        if not self.config.eval_data_paths:
            return None
        if self.config.dataset_type == "sft":
            return SFTDataset(
                paths=self.config.eval_data_paths,
                tokenizer=self.tokenizer,
                max_seq_len=self.config.max_seq_len,
                cache_dir=self.config.cache_dir,
            )
        return PretrainDataset(
            paths=self.config.eval_data_paths,
            tokenizer=self.tokenizer,
            max_seq_len=self.config.max_seq_len,
            cache_dir=self.config.cache_dir,
            text_field=self.config.text_field,
        )

    def get_collator(self) -> Callable:
        return SFTCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            ignore_index=-100,
            pad_to_multiple_of=8,
        )

    # ----- Setup -----

    def setup(self):
        super().setup()
        # 把 teacher 也搬到当前 device
        self.teacher.model = self.teacher.model.to(self.device)
        self.teacher.device = self.device

    # ----- Loss -----

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cfg = self.config
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids.clone())
        attention_mask = batch.get("attention_mask")

        # Student forward
        s_out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        s_logits = s_out["logits"] if isinstance(s_out, dict) else s_out.logits

        # shift
        s_shift = s_logits[..., :-1, :].contiguous()
        labels_shift = labels[..., 1:].contiguous()

        # CE loss
        ce = F.cross_entropy(
            s_shift.float().view(-1, s_shift.size(-1)),
            labels_shift.view(-1),
            ignore_index=-100,
            reduction="mean",
        )

        out: Dict[str, torch.Tensor] = {"ce_loss": ce.detach()}
        loss = cfg.alpha_ce * ce

        # ===== KD loss =====
        if cfg.alpha_ce < 1.0:
            # Teacher forward
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)

            # 仅在 labels != -100 处计算 KD
            mask = (labels_shift != -100).float()

            if cfg.kd_type == "topk":
                # Teacher 提供 topk_values / topk_indices
                if "topk_values" not in t_out:
                    raise RuntimeError("teacher_topk 必须 > 0 才能用 topk 蒸馏")
                t_vals = t_out["topk_values"][..., :-1, :].contiguous()
                t_idx = t_out["topk_indices"][..., :-1, :].contiguous()
                kd = topk_kd_loss(
                    s_shift, t_vals, t_idx,
                    mask=mask, temperature=cfg.temperature,
                )
            else:
                t_logits = t_out["logits"][..., :-1, :].contiguous()
                if cfg.kd_type == "forward_kl":
                    kd = kd_loss(s_shift, t_logits, mask=mask, temperature=cfg.temperature)
                elif cfg.kd_type == "reverse_kl":
                    kd = reverse_kd_loss(s_shift, t_logits, mask=mask, temperature=cfg.temperature)
                elif cfg.kd_type == "jsd":
                    kd = jsd_loss(
                        s_shift, t_logits, mask=mask,
                        temperature=cfg.temperature, alpha=cfg.jsd_alpha,
                    )
                else:
                    raise ValueError(f"Unknown kd_type: {cfg.kd_type}")

            out["kd_loss"] = kd.detach()
            loss = loss + (1 - cfg.alpha_ce) * kd

        out["loss"] = loss

        # acc
        with torch.no_grad():
            preds = s_shift.argmax(dim=-1)
            valid = labels_shift != -100
            acc = ((preds == labels_shift) & valid).sum().float() / valid.sum().clamp(min=1.0)
            out["acc"] = acc.detach()

        return out
