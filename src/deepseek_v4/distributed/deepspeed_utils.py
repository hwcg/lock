"""
DeepSpeed 集成（可选依赖）。

支持 ZeRO 1/2/3，含 CPU offload；
所有函数在 deepspeed 未安装时安全降级。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def deepspeed_available() -> bool:
    try:
        import deepspeed  # noqa: F401
        return True
    except ImportError:
        return False


def get_deepspeed_config(stage: int = 2, offload_optimizer: bool = False, offload_param: bool = False) -> Dict[str, Any]:
    """生成标准 ZeRO 配置（程序化），便于在 yaml 中直接覆盖。"""
    cfg: Dict[str, Any] = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 50,
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "wall_clock_breakdown": False,
    }
    if stage >= 2 and offload_optimizer:
        cfg["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu", "pin_memory": True,
        }
    if stage == 3 and offload_param:
        cfg["zero_optimization"]["offload_param"] = {
            "device": "cpu", "pin_memory": True,
        }
    if stage == 3:
        cfg["zero_optimization"].update({
            "sub_group_size": 1e9,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        })
    return cfg


def build_deepspeed_engine(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    model_parameters: Optional[Any] = None,
    lr_scheduler: Optional[Any] = None,
) -> Tuple[Any, Optional[torch.optim.Optimizer], Optional[Any]]:
    """
    用 deepspeed.initialize 构造 engine。

    Returns:
        (engine, optimizer, scheduler)
    """
    if not deepspeed_available():
        raise RuntimeError("deepspeed 未安装，请 pip install deepspeed")

    import deepspeed
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    engine, opt, _, sch = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
        model_parameters=model_parameters or [p for p in model.parameters() if p.requires_grad],
        lr_scheduler=lr_scheduler,
    )
    return engine, opt, sch


def save_deepspeed_checkpoint(engine: Any, save_dir: str, tag: str) -> None:
    """保存 deepspeed checkpoint（自动处理 ZeRO 分片）。"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(save_dir, tag=tag)


def load_deepspeed_checkpoint(engine: Any, load_dir: str, tag: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """加载 deepspeed checkpoint。返回 (loaded_tag, client_state)。"""
    return engine.load_checkpoint(load_dir, tag=tag)
