"""模型蒸馏。"""
from deepseek_v4.training.distill.losses import (
    kd_loss, reverse_kd_loss, topk_kd_loss, jsd_loss,
)
from deepseek_v4.training.distill.teacher import TeacherWrapper
from deepseek_v4.training.distill.trainer import DistillConfig, DistillTrainer

__all__ = [
    "kd_loss", "reverse_kd_loss", "topk_kd_loss", "jsd_loss",
    "TeacherWrapper",
    "DistillConfig", "DistillTrainer",
]
