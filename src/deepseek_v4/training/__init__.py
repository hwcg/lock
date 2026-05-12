"""训练子包：优化器 / 调度器 / Trainer / 检查点。"""
from deepseek_v4.training.optim import (
    AdamW, Muon, build_optimizer, build_scheduler, group_parameters,
    CosineWarmupScheduler, LinearWarmupScheduler, WSDScheduler,
)
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.training.pretrain import PretrainTrainer, PretrainConfig
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing

__all__ = [
    "AdamW", "Muon", "build_optimizer", "build_scheduler", "group_parameters",
    "CosineWarmupScheduler", "LinearWarmupScheduler", "WSDScheduler",
    "CheckpointManager",
    "BaseTrainer", "TrainerConfig",
    "PretrainTrainer", "PretrainConfig",
    "enable_gradient_checkpointing",
]
