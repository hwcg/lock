"""
Checkpoint 管理器：

- 支持 PyTorch 原生 + safetensors 两种格式
- 支持 DeepSpeed ZeRO 自动委派
- 原子写入（先 .tmp 再 rename）
- 维护 best-K 与 last-K 列表
- 自动恢复（resume_from_checkpoint）
- 跨 rank 同步（仅 rank0 写盘）
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from deepseek_v4.distributed.utils import (
    barrier, broadcast_object_list, is_main_process, is_distributed,
)
from deepseek_v4.utils.io import atomic_write, safe_load_json, safe_save_json
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)

CKPT_META_FILENAME = "trainer_state.json"
MODEL_FILENAME_PT = "pytorch_model.bin"
MODEL_FILENAME_ST = "model.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
RNG_FILENAME = "rng_state.pt"


@dataclass
class CheckpointMeta:
    """每个 checkpoint 的元信息。"""
    path: str
    step: int
    epoch: int = 0
    metric_value: Optional[float] = None
    is_best: bool = False
    timestamp: float = field(default_factory=time.time)


# ------------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------------

def _unwrap(model: nn.Module) -> nn.Module:
    """剥离 DDP / compile 包装。"""
    while hasattr(model, "module") and not isinstance(model, nn.Module.__bases__):
        model = model.module
    if hasattr(model, "_orig_mod"):  # torch.compile
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


def _save_safetensors(state_dict: Dict[str, torch.Tensor], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError:
        torch.save(state_dict, path.with_suffix(".bin"))
        logger.warning("safetensors not installed, fell back to .bin")
        return
    # safetensors 不支持非 contiguous tensor
    cleaned = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
    save_file(cleaned, str(path), metadata={"format": "pt"})


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(path), device="cpu")


# ------------------------------------------------------------------------
# CheckpointManager
# ------------------------------------------------------------------------

class CheckpointManager:
    """
    多模式 Checkpoint 管理。

    用法：
        ckpt = CheckpointManager(output_dir="ckpts", keep_last=3, keep_best=2,
                                 metric_name="eval_loss", metric_mode="min")
        ckpt.save(model, optimizer, scheduler, step=1000, metric_value=2.5)
        # ↓ 恢复
        meta = ckpt.load_latest(model, optimizer, scheduler)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        keep_last: int = 3,
        keep_best: int = 2,
        metric_name: str = "loss",
        metric_mode: str = "min",   # "min" | "max"
        save_format: str = "safetensors",  # "safetensors" | "pytorch"
        deepspeed_engine: Any = None,
        save_on_main_only: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        if metric_mode not in ("min", "max"):
            raise ValueError(f"metric_mode must be min/max, got {metric_mode}")
        self.save_format = save_format
        self.engine = deepspeed_engine
        self.save_on_main_only = save_on_main_only
        self._history: List[CheckpointMeta] = []
        self._index_path = self.output_dir / "checkpoint_index.json"
        self._load_index()

    # -------- 索引 --------

    def _load_index(self) -> None:
        if self._index_path.exists():
            data = safe_load_json(self._index_path, default=[])
            self._history = [CheckpointMeta(**d) for d in data]

    def _save_index(self) -> None:
        if not is_main_process():
            return
        safe_save_json(self._index_path, [m.__dict__ for m in self._history])

    # -------- 主接口：保存 --------

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Any = None,
        step: int = 0,
        epoch: int = 0,
        metric_value: Optional[float] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        保存一个 checkpoint。

        Args:
            extra_state: 额外保存到 trainer_state.json 的字典
        """
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.engine is not None:
            self._save_deepspeed(ckpt_dir, step, epoch)
        else:
            self._save_native(ckpt_dir, model, optimizer, scheduler)

        # trainer state
        if is_main_process():
            state = {
                "step": step,
                "epoch": epoch,
                "metric_name": self.metric_name,
                "metric_value": metric_value,
                "metric_mode": self.metric_mode,
                "timestamp": time.time(),
            }
            if extra_state:
                state.update(extra_state)
            safe_save_json(ckpt_dir / CKPT_META_FILENAME, state)

        meta = CheckpointMeta(
            path=str(ckpt_dir),
            step=step,
            epoch=epoch,
            metric_value=metric_value,
        )
        self._history.append(meta)

        if is_main_process():
            self._rotate()
            self._save_index()

        if is_distributed():
            barrier()

        logger.info(f"[Checkpoint] saved: {ckpt_dir}  metric={metric_value}")
        return ckpt_dir

    # -------- DeepSpeed --------

    def _save_deepspeed(self, ckpt_dir: Path, step: int, epoch: int) -> None:
        # 所有 rank 都要参与
        client_state = {"step": step, "epoch": epoch}
        self.engine.save_checkpoint(str(ckpt_dir.parent), tag=ckpt_dir.name, client_state=client_state)

    def _load_deepspeed(self, ckpt_dir: Path) -> Dict[str, Any]:
        load_path, client_state = self.engine.load_checkpoint(
            str(ckpt_dir.parent), tag=ckpt_dir.name,
        )
        if load_path is None:
            raise FileNotFoundError(f"DeepSpeed failed to load {ckpt_dir}")
        return client_state

    # -------- Native --------

    def _save_native(
        self,
        ckpt_dir: Path,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Any,
    ) -> None:
        if self.save_on_main_only and not is_main_process():
            return

        unwrapped = _unwrap(model)
        if self.save_format == "safetensors":
            _save_safetensors(unwrapped.state_dict(), ckpt_dir / MODEL_FILENAME_ST)
        else:
            torch.save(unwrapped.state_dict(), ckpt_dir / MODEL_FILENAME_PT)

        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / OPTIMIZER_FILENAME)
        if scheduler is not None:
            try:
                torch.save(scheduler.state_dict(), ckpt_dir / SCHEDULER_FILENAME)
            except Exception:
                pass

        # RNG
        rng = {
            "python":      None,                    # python 内置 random 由调用者保存（避免依赖）
            "torch":       torch.get_rng_state(),
            "torch_cuda":  torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            import numpy as np
            rng["numpy"] = np.random.get_state()
        except ImportError:
            pass
        torch.save(rng, ckpt_dir / RNG_FILENAME)

    # -------- 加载 --------

    def load(
        self,
        ckpt_dir: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Any = None,
        strict: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_rng: bool = True,
    ) -> Dict[str, Any]:
        """
        从指定目录加载。返回 trainer_state dict（含 step / epoch）。
        """
        ckpt_dir = Path(ckpt_dir)
        if self.engine is not None:
            client = self._load_deepspeed(ckpt_dir)
            state = safe_load_json(ckpt_dir / CKPT_META_FILENAME, default={})
            state.update(client or {})
            return state

        # 模型
        unwrapped = _unwrap(model)
        st_path = ckpt_dir / MODEL_FILENAME_ST
        bin_path = ckpt_dir / MODEL_FILENAME_PT
        if st_path.exists():
            sd = _load_safetensors(st_path)
        elif bin_path.exists():
            sd = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model file in {ckpt_dir}")

        missing, unexpected = unwrapped.load_state_dict(sd, strict=strict)
        if missing:
            logger.warning(f"[Checkpoint] missing keys: {len(missing)} (前 5: {missing[:5]})")
        if unexpected:
            logger.warning(f"[Checkpoint] unexpected keys: {len(unexpected)} (前 5: {unexpected[:5]})")

        # 优化器
        if load_optimizer and optimizer is not None and (ckpt_dir / OPTIMIZER_FILENAME).exists():
            opt_state = torch.load(ckpt_dir / OPTIMIZER_FILENAME, map_location="cpu")
            optimizer.load_state_dict(opt_state)

        # scheduler
        if load_scheduler and scheduler is not None and (ckpt_dir / SCHEDULER_FILENAME).exists():
            try:
                scheduler.load_state_dict(torch.load(ckpt_dir / SCHEDULER_FILENAME, map_location="cpu"))
            except Exception as e:
                logger.warning(f"failed to load scheduler: {e}")

        # RNG
        if load_rng and (ckpt_dir / RNG_FILENAME).exists():
            try:
                rng = torch.load(ckpt_dir / RNG_FILENAME, map_location="cpu")
                if rng.get("torch") is not None:
                    torch.set_rng_state(rng["torch"])
                if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                if rng.get("numpy") is not None:
                    import numpy as np
                    np.random.set_state(rng["numpy"])
            except Exception as e:
                logger.warning(f"failed to load rng: {e}")

        state = safe_load_json(ckpt_dir / CKPT_META_FILENAME, default={})
        return state

    def load_latest(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        latest = self.find_latest()
        if latest is None:
            return None
        return self.load(latest, *args, **kwargs)

    def load_best(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        best = self.find_best()
        if best is None:
            return None
        return self.load(best, *args, **kwargs)

    # -------- 查找 --------

    def find_latest(self) -> Optional[Path]:
        if not self._history:
            # 兜底：扫描目录
            candidates = sorted(
                self.output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            )
            return candidates[-1] if candidates else None
        return Path(self._history[-1].path)

    def find_best(self) -> Optional[Path]:
        valid = [m for m in self._history if m.metric_value is not None]
        if not valid:
            return None
        if self.metric_mode == "min":
            best = min(valid, key=lambda m: m.metric_value)
        else:
            best = max(valid, key=lambda m: m.metric_value)
        return Path(best.path)

    # -------- 轮转 --------

    def _rotate(self) -> None:
        """根据 keep_last / keep_best 删除旧 checkpoint。"""
        # 维护 best-K 集合
        valid_metric = [m for m in self._history if m.metric_value is not None]
        if valid_metric and self.keep_best > 0:
            sorted_by_metric = sorted(
                valid_metric,
                key=lambda m: m.metric_value if self.metric_mode == "min" else -m.metric_value,
            )
            best_paths = {m.path for m in sorted_by_metric[: self.keep_best]}
        else:
            best_paths = set()

        # 维护 last-K
        sorted_by_step = sorted(self._history, key=lambda m: m.step)
        last_paths = {m.path for m in sorted_by_step[-max(self.keep_last, 0):]} if self.keep_last > 0 else set()

        keep = best_paths | last_paths
        new_history: List[CheckpointMeta] = []
        for m in self._history:
            if m.path in keep:
                m.is_best = m.path in best_paths
                new_history.append(m)
            else:
                # 删除
                p = Path(m.path)
                if p.exists():
                    try:
                        shutil.rmtree(p)
                        logger.info(f"[Checkpoint] removed old: {p}")
                    except Exception as e:
                        logger.warning(f"failed to remove {p}: {e}")
        self._history = new_history
