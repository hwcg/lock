"""
日志系统：

- get_logger：标准 logging（含 rich 美化）
- MetricLogger：移动平均的训练 metric 累计器
- WandBLogger / SwanLabLogger：双后端可视化
- MultiLogger：组合多个 backend
- 支持动态启停（环境变量 DS4_PAUSE 触发暂停）
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# 可选依赖
try:
    from rich.logging import RichHandler
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ============================================================
# Logging 基础
# ============================================================

_LOGGERS: Dict[str, logging.Logger] = {}


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    rich: bool = True,
    rank: int = 0,
) -> None:
    """全局 logging 设置。仅 rank=0 输出 INFO 及以上。"""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # 清理已有 handler
    for h in list(root.handlers):
        root.removeHandler(h)

    # rank > 0 只输出 WARNING+
    effective_level = level if rank == 0 else logging.WARNING
    root.setLevel(effective_level)

    fmt = f"[%(asctime)s] [rank{rank}] [%(name)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if rich and HAS_RICH and rank == 0:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt=datefmt))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.setLevel(effective_level)
    root.addHandler(handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        fh.setLevel(level)
        root.addHandler(fh)


def get_logger(name: str = "deepseek_v4") -> logging.Logger:
    """获取按名字缓存的 logger。"""
    if name not in _LOGGERS:
        _LOGGERS[name] = logging.getLogger(name)
    return _LOGGERS[name]


# ============================================================
# Metric Logger
# ============================================================

class MovingAverage:
    """带窗口的移动平均。"""
    def __init__(self, window: int = 100):
        self.window = window
        self.values: deque = deque(maxlen=window)
        self.total: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.values.append(value)
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return sum(self.values) / max(len(self.values), 1)

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def latest(self) -> float:
        return self.values[-1] if self.values else 0.0


class MetricLogger:
    """
    训练 metric 累计器（移动平均 + 全局平均）。

    用法：
        meter = MetricLogger(window=100)
        for batch in loader:
            loss = ...
            meter.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            if step % 10 == 0:
                logger.info(meter.summary(step))
    """
    def __init__(self, window: int = 100):
        self.window = window
        self.meters: Dict[str, MovingAverage] = {}
        self.start_time = time.time()

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = MovingAverage(window=self.window)
            self.meters[k].update(float(v))

    def get(self, key: str) -> Optional[float]:
        m = self.meters.get(key)
        return m.avg if m is not None else None

    def items(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}

    def summary(self, step: Optional[int] = None) -> str:
        elapsed = time.time() - self.start_time
        parts = []
        if step is not None:
            parts.append(f"step={step}")
        for k, m in self.meters.items():
            parts.append(f"{k}={m.avg:.4f}")
        parts.append(f"elapsed={elapsed:.0f}s")
        return "  ".join(parts)

    def reset(self) -> None:
        for m in self.meters.values():
            m.values.clear()
            m.total = 0.0
            m.count = 0


# ============================================================
# WandB / SwanLab Backend
# ============================================================

class _BaseTrackerBackend:
    """Tracker backend 接口。"""
    name: str = "base"

    def is_active(self) -> bool:
        return False

    def init(self, project: str, name: str, config: Dict[str, Any], **kwargs) -> None:
        pass

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str, name: str, type: str = "model") -> None:
        pass

    def finish(self) -> None:
        pass


class WandBLogger(_BaseTrackerBackend):
    """Weights & Biases 集成。"""
    name = "wandb"

    def __init__(self):
        try:
            import wandb
            self.wandb = wandb
            self._available = True
        except ImportError:
            self.wandb = None
            self._available = False
        self.run = None

    def is_active(self) -> bool:
        return self._available and self.run is not None

    def init(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not self._available:
            return
        self.run = self.wandb.init(
            project=project, name=name, config=config,
            entity=entity, tags=tags, notes=notes,
            reinit=True, resume="allow",
        )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.is_active():
            self.wandb.log(data, step=step)

    def log_artifact(self, path: str, name: str, type: str = "model") -> None:
        if not self.is_active():
            return
        art = self.wandb.Artifact(name=name, type=type)
        art.add_file(path) if os.path.isfile(path) else art.add_dir(path)
        self.run.log_artifact(art)

    def finish(self) -> None:
        if self.is_active():
            self.wandb.finish()
            self.run = None


class SwanLabLogger(_BaseTrackerBackend):
    """SwanLab 集成（国内可视化平台）。"""
    name = "swanlab"

    def __init__(self):
        try:
            import swanlab
            self.swanlab = swanlab
            self._available = True
        except ImportError:
            self.swanlab = None
            self._available = False
        self._inited = False

    def is_active(self) -> bool:
        return self._available and self._inited

    def init(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        workspace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        if not self._available:
            return
        self.swanlab.init(
            project=project, experiment_name=name,
            config=config, workspace=workspace, tags=tags,
            mode=os.environ.get("SWANLAB_MODE", "cloud"),
        )
        self._inited = True

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.is_active():
            self.swanlab.log(data, step=step)

    def finish(self) -> None:
        if self.is_active():
            self.swanlab.finish()
            self._inited = False


class JSONLLogger(_BaseTrackerBackend):
    """本地 jsonl 日志（用于无网络环境）。"""
    name = "jsonl"

    def __init__(self):
        self.fp = None
        self.path = None

    def is_active(self) -> bool:
        return self.fp is not None

    def init(self, project: str, name: str, config: Dict[str, Any], output_dir: str = "logs", **kwargs) -> None:
        d = Path(output_dir) / project
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{name}.jsonl"
        self.fp = open(self.path, "a", encoding="utf-8")
        self.fp.write(json.dumps({"_event": "init", "config": config}, ensure_ascii=False) + "\n")
        self.fp.flush()

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.is_active():
            return
        row = dict(data)
        if step is not None:
            row["_step"] = step
        row["_ts"] = time.time()
        self.fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self.fp.flush()

    def finish(self) -> None:
        if self.is_active():
            self.fp.close()
            self.fp = None


# ============================================================
# 动态启停（信号驱动）
# ============================================================

class _PauseController:
    """
    SIGUSR1 触发暂停 / 继续，SIGUSR2 触发立即保存退出。

    使用：
        ctrl = PAUSE_CONTROLLER
        if ctrl.should_save_and_exit():
            save_checkpoint(); sys.exit(0)
        ctrl.wait_if_paused()
    """
    def __init__(self):
        self.paused = False
        self.save_and_exit = False
        self._registered = False

    def _handle_pause(self, signum, frame):
        self.paused = not self.paused
        get_logger().warning(f"[PauseController] {'PAUSED' if self.paused else 'RESUMED'}")

    def _handle_save_exit(self, signum, frame):
        self.save_and_exit = True
        get_logger().warning(f"[PauseController] save-and-exit requested")

    def register(self) -> None:
        if self._registered:
            return
        # 仅在主线程注册（在 worker 进程中可能失败，忽略）
        try:
            signal.signal(signal.SIGUSR1, self._handle_pause)
            signal.signal(signal.SIGUSR2, self._handle_save_exit)
            self._registered = True
        except (ValueError, AttributeError, OSError):
            # 非主线程或不支持 signal 的平台（Windows）
            pass

    def wait_if_paused(self, check_interval: float = 1.0) -> None:
        while self.paused and not self.save_and_exit:
            time.sleep(check_interval)

    def should_save_and_exit(self) -> bool:
        return self.save_and_exit


PAUSE_CONTROLLER = _PauseController()


# ============================================================
# Multi Logger
# ============================================================

class MultiLogger:
    """
    统一对外接口：同时往多个 backend 写。

    用法：
        ml = MultiLogger(backends=["wandb", "jsonl"])
        ml.init(project="ds4", name="run1", config=cfg.to_dict())
        ml.log({"loss": 0.5, "lr": 1e-4}, step=100)
        ml.finish()
    """
    BACKEND_REGISTRY = {
        "wandb":    WandBLogger,
        "swanlab":  SwanLabLogger,
        "jsonl":    JSONLLogger,
    }

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        rank: int = 0,
        enable_signal_control: bool = True,
    ):
        self.rank = rank
        self.backends: List[_BaseTrackerBackend] = []
        backends = backends or []
        if rank == 0:
            for name in backends:
                if name not in self.BACKEND_REGISTRY:
                    get_logger().warning(f"Unknown logger backend: {name}, skipped")
                    continue
                self.backends.append(self.BACKEND_REGISTRY[name]())
        if enable_signal_control:
            PAUSE_CONTROLLER.register()

    def init(self, **kwargs) -> None:
        for b in self.backends:
            try:
                b.init(**kwargs)
                get_logger().info(f"[MultiLogger] {b.name} initialized")
            except Exception as e:
                get_logger().warning(f"[MultiLogger] failed to init {b.name}: {e}")

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        for b in self.backends:
            try:
                b.log(data, step=step)
            except Exception as e:
                get_logger().warning(f"[MultiLogger] {b.name} log failed: {e}")

    def log_artifact(self, path: str, name: str, type: str = "model") -> None:
        for b in self.backends:
            try:
                b.log_artifact(path, name, type)
            except Exception as e:
                get_logger().warning(f"[MultiLogger] {b.name} artifact failed: {e}")

    def finish(self) -> None:
        for b in self.backends:
            try:
                b.finish()
            except Exception:
                pass
