"""
LoRA 训练器：
    1. 复用 SFTTrainer 的训练循环
    2. setup() 时注入 LoRA、冻结基础参数
    3. 保存只保存 adapter（占用极小）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn

from deepseek_v4.training.lora.apply import (
    apply_lora, get_lora_state_dict, print_trainable_parameters, save_lora,
)
from deepseek_v4.training.lora.config import LoRAConfig
from deepseek_v4.training.sft import SFTConfig, SFTTrainer
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LoRATrainerConfig(SFTConfig):
    """LoRA 训练配置：继承 SFTConfig，新增 lora 相关项。"""
    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    save_full_model: bool = False     # 是否同时保存全量 merged model（默认否，节省磁盘）
    save_adapter_only: bool = True     # 仅保存 adapter（推荐）

    # 一些默认覆盖
    learning_rate: float = 1.0e-4      # LoRA 用更大 lr
    weight_decay: float = 0.0


class LoRATrainer(SFTTrainer):
    """
    LoRA-SFT 训练器。

    主要差异：
    - setup 中调用 apply_lora
    - 保存时仅保存 adapter（除非 save_full_model）
    - 优化器仅作用于 trainable 参数（自动处理）
    """

    def __init__(self, config: LoRATrainerConfig, model, tokenizer):
        super().__init__(config=config, model=model, tokenizer=tokenizer)
        self.config: LoRATrainerConfig = config

    def setup(self) -> None:
        # 先注入 LoRA（在 _build_optim_engine 之前），让 optimizer 只看到 trainable params
        unwrap = self.model
        # 此时 model 还在 cpu / 还没 DDP wrap
        apply_lora(unwrap, self.config.lora)
        # 调用父类
        super().setup()
        # 打印
        print_trainable_parameters(self.model.module if hasattr(self.model, "module") else self.model)

    def _save_ckpt(self, metric: Optional[float], force: bool = False) -> None:
        """保存 adapter（默认）或全量。"""
        # 当只存 adapter 时，跳过 base trainer 的 full state_dict 保存，自己保存
        if self.config.save_adapter_only and not self.config.save_full_model:
            from deepseek_v4.distributed.utils import is_main_process
            if is_main_process():
                save_dir = (
                    self.ckpt_mgr.output_dir / f"checkpoint-{self.global_step}"
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                unwrap = self.model.module if hasattr(self.model, "module") else self.model
                save_lora(unwrap, save_dir, config=self.config.lora)
                # 元信息
                from deepseek_v4.utils.io import safe_save_json
                safe_save_json(save_dir / "trainer_state.json", {
                    "step": self.global_step,
                    "epoch": self.epoch,
                    "metric_name": self.config.metric_name,
                    "metric_value": metric,
                    "metric_mode": self.config.metric_mode,
                    "tokens_seen": self.tokens_seen,
                    "best_metric": self.best_metric,
                    "is_lora_only": True,
                })
                # 用 ckpt_mgr 的标准 API 维护轮转
                from deepseek_v4.training.checkpoint import CheckpointMeta
                meta = CheckpointMeta(
                    path=str(save_dir), step=self.global_step,
                    epoch=self.epoch, metric_value=metric,
                )
                self.ckpt_mgr._history.append(meta)
                self.ckpt_mgr._rotate()
                self.ckpt_mgr._save_index()
                logger.info(f"[LoRA] adapter saved at step {self.global_step}, metric={metric}")
            return
        # 否则走父类逻辑（存全量）
        super()._save_ckpt(metric=metric, force=force)
