"""
通用训练循环抽象。

设计原则：
- 一份代码同时支持 DDP / DeepSpeed / 单卡
- 所有阶段（pretrain / sft / dpo / ppo / grpo / cispo / 蒸馏）共用主循环
- 子类只需实现 compute_loss
- 支持 grad_accum、AMP、grad clip、checkpoint、信号驱动暂停
"""
from __future__ import annotations

import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from deepseek_v4.distributed.deepspeed_utils import deepspeed_available, build_deepspeed_engine
from deepseek_v4.distributed.utils import (
    all_reduce_mean, barrier, get_rank, get_world_size, is_distributed,
    is_main_process, only_on_main, setup_distributed,
)
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing
from deepseek_v4.training.optim import build_optimizer, build_scheduler
from deepseek_v4.utils.config import BaseConfig
from deepseek_v4.utils.logger import (
    MetricLogger, MultiLogger, PAUSE_CONTROLLER, get_logger, setup_logging,
)
from deepseek_v4.utils.seed import set_seed
from deepseek_v4.utils.timer import Stopwatch, format_time

logger = get_logger(__name__)


# ============================================================
# Trainer 配置
# ============================================================

@dataclass
class TrainerConfig(BaseConfig):
    """通用训练配置。各阶段会继承。"""

    # 输出
    output_dir: str = "checkpoints/run"
    run_name: str = "run"
    project_name: str = "deepseek-v4-mini"

    # 训练
    seed: int = 42
    max_steps: int = 100000
    max_epochs: Optional[int] = None
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    eval_steps: int = 1000
    save_steps: int = 1000
    log_steps: int = 10
    eval_max_batches: int = 100

    # 优化器
    optimizer: str = "adamw"            # adamw | muon
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"           # constant | linear | cosine | polynomial | wsd
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    wsd_decay_ratio: float = 0.1

    # 精度
    precision: str = "bf16"             # fp32 | fp16 | bf16
    grad_scaler_enabled: Optional[bool] = None  # None=auto

    # Gradient checkpointing
    gradient_checkpointing: bool = True
    gc_skip_first_n: int = 0
    gc_skip_last_n: int = 0

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True

    # 分布式
    distributed_backend: str = "nccl"
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None

    # Checkpoint
    keep_last: int = 3
    keep_best: int = 2
    metric_name: str = "eval_loss"
    metric_mode: str = "min"
    save_format: str = "safetensors"
    resume_from_checkpoint: Optional[str] = None  # "auto" / 绝对路径 / None

    # Logging
    logger_backends: List[str] = field(default_factory=lambda: ["jsonl"])
    log_dir: Optional[str] = None      # 默认 output_dir/logs
    enable_signal_control: bool = True

    # 其他
    deterministic: bool = False
    compile_model: bool = False
    print_param_count: bool = True

    def precision_dtype(self) -> torch.dtype:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]


# ============================================================
# BaseTrainer
# ============================================================

class BaseTrainer:
    """
    通用训练器骨架。子类需要实现：
    - compute_loss(batch) → Dict[str, Tensor]，必须包含 "loss"
    - get_train_dataset() → Dataset
    - get_eval_dataset() → Optional[Dataset]
    - get_collator() → Callable
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        collator: Optional[Callable] = None,
        compute_loss_fn: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
    ):
        self.config = config
        self.model = model
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._collator = collator
        self._user_compute_loss = compute_loss_fn

        # 状态
        self.global_step: int = 0
        self.epoch: int = 0
        self.tokens_seen: int = 0
        self.best_metric: Optional[float] = None
        self.start_time: float = 0.0

        # 组件
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None
        self.engine: Optional[Any] = None       # DeepSpeed engine
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.ckpt_mgr: Optional[CheckpointManager] = None
        self.metric_logger = MetricLogger(window=100)
        self.tracker_logger: Optional[MultiLogger] = None
        self.timer = Stopwatch()

    # ------------------------------------------------------------------
    # 必须 / 可选由子类覆盖的钩子
    # ------------------------------------------------------------------

    def get_train_dataset(self) -> Dataset:
        if self._train_dataset is None:
            raise NotImplementedError("train_dataset not provided")
        return self._train_dataset

    def get_eval_dataset(self) -> Optional[Dataset]:
        return self._eval_dataset

    def get_collator(self) -> Optional[Callable]:
        return self._collator

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """子类 / closure 实现：返回必须含 'loss'。"""
        if self._user_compute_loss is not None:
            return self._user_compute_loss(self.model, batch)
        raise NotImplementedError

    def prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """把 batch 移到正确的 device。"""
        device = self.device
        return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def on_step_end(self, metrics: Dict[str, float]) -> None:
        """钩子：每完整 step 之后被调用。"""
        pass

    # ------------------------------------------------------------------
    # 设置阶段
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """初始化分布式 / logger / 优化器 / dataloader。"""
        # 分布式
        setup_distributed(backend=self.config.distributed_backend)

        # 设备
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # 日志
        log_dir = self.config.log_dir or os.path.join(self.config.output_dir, "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = os.path.join(log_dir, f"rank{get_rank()}.log")
        setup_logging(level="INFO", log_file=log_file, rank=get_rank())
        if is_main_process():
            logger.info("=" * 70)
            logger.info(f"  Trainer: {type(self).__name__}")
            logger.info(f"  output_dir: {self.config.output_dir}")
            logger.info(f"  world_size: {get_world_size()}")
            logger.info(f"  device: {self.device}")
            logger.info(f"  precision: {self.config.precision}")
            logger.info("=" * 70)

        # 随机种子
        set_seed(self.config.seed + get_rank(), deterministic=self.config.deterministic)

        # 模型移到 device
        self.model = self.model.to(self.device, dtype=self._model_dtype())

        # gradient checkpointing
        if self.config.gradient_checkpointing:
            n = enable_gradient_checkpointing(
                self.model,
                skip_first_n=self.config.gc_skip_first_n,
                skip_last_n=self.config.gc_skip_last_n,
            )
            if is_main_process():
                logger.info(f"[GC] enabled on {n} layers")

        # 参数统计
        if self.config.print_param_count and is_main_process():
            self._print_param_count()

        # 数据
        self._build_dataloaders()

        # 优化器 / scheduler / engine
        self._build_optim_engine()

        # Checkpoint mgr
        self.ckpt_mgr = CheckpointManager(
            output_dir=self.config.output_dir,
            keep_last=self.config.keep_last,
            keep_best=self.config.keep_best,
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            save_format=self.config.save_format,
            deepspeed_engine=self.engine,
        )

        # Tracker logger
        if is_main_process():
            self.tracker_logger = MultiLogger(
                backends=self.config.logger_backends,
                rank=0,
                enable_signal_control=self.config.enable_signal_control,
            )
            self.tracker_logger.init(
                project=self.config.project_name,
                name=self.config.run_name,
                config=self.config.to_dict(),
                output_dir=os.path.join(self.config.output_dir, "tracker"),
            )

        # 恢复
        self._maybe_resume()

    def _model_dtype(self) -> torch.dtype:
        # bf16 训练时模型本身保持 fp32（autocast 临时降精度），但 V4 的设计接受 bf16 主权重
        # 这里对单卡 / DDP 简单选 fp32（autocast 降精度），DeepSpeed 由配置决定
        return torch.float32

    def _print_param_count(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"[Params] total={total:,} ({total/1e9:.3f}B) trainable={trainable:,} ({trainable/1e9:.3f}B)")

    def _build_dataloaders(self) -> None:
        from deepseek_v4.data.dataset import build_dataloader

        train_ds = self.get_train_dataset()
        eval_ds = self.get_eval_dataset()
        collator = self.get_collator()

        self.train_loader = build_dataloader(
            train_ds,
            batch_size=self.config.micro_batch_size,
            collate_fn=collator,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            distributed=is_distributed(),
            seed=self.config.seed,
        )
        if eval_ds is not None:
            self.eval_loader = build_dataloader(
                eval_ds,
                batch_size=self.config.micro_batch_size,
                collate_fn=collator,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                distributed=is_distributed(),
                seed=self.config.seed,
            )

    def _build_optim_engine(self) -> None:
        """构造 optimizer / scheduler / DeepSpeed engine / DDP wrapper。"""
        cfg = self.config

        if cfg.use_deepspeed:
            if not deepspeed_available():
                raise RuntimeError("deepspeed not installed")
            ds_cfg = self._materialize_ds_config()
            self.engine, self.optimizer, self.scheduler = build_deepspeed_engine(
                model=self.model,
                config=ds_cfg,
            )
            return

        # ---- 非 DS：手工构造 ----
        self.optimizer = build_optimizer(
            self.model,
            name=cfg.optimizer,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            name=cfg.scheduler,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.max_steps,
            min_lr_ratio=cfg.min_lr_ratio,
            decay_ratio=cfg.wsd_decay_ratio if cfg.scheduler == "wsd" else None,
        ) if cfg.scheduler != "wsd" else build_scheduler(
            self.optimizer, name="wsd",
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.max_steps,
            decay_ratio=cfg.wsd_decay_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
        )

        # DDP 包装
        if is_distributed():
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                find_unused_parameters=False,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )

        # Compile（可选）
        if cfg.compile_model:
            try:
                self.model = torch.compile(self.model)
                if is_main_process():
                    logger.info("[Compile] torch.compile enabled")
            except Exception as e:
                if is_main_process():
                    logger.warning(f"torch.compile failed: {e}")

        # GradScaler
        if cfg.precision == "fp16":
            enabled = cfg.grad_scaler_enabled if cfg.grad_scaler_enabled is not None else True
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def _materialize_ds_config(self) -> Dict[str, Any]:
        """根据 TrainerConfig 自动渲染一份 DeepSpeed config。"""
        import json
        if self.config.deepspeed_config and Path(self.config.deepspeed_config).exists():
            with open(self.config.deepspeed_config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            from deepseek_v4.distributed.deepspeed_utils import get_deepspeed_config
            cfg = get_deepspeed_config(stage=2)
        # 替换 "auto"
        cfg["train_micro_batch_size_per_gpu"] = self.config.micro_batch_size
        cfg["gradient_accumulation_steps"] = self.config.gradient_accumulation_steps
        cfg["train_batch_size"] = (
            self.config.micro_batch_size *
            self.config.gradient_accumulation_steps *
            get_world_size()
        )
        cfg["gradient_clipping"] = self.config.max_grad_norm
        if self.config.precision == "bf16":
            cfg["bf16"] = {"enabled": True}
            cfg.pop("fp16", None)
        elif self.config.precision == "fp16":
            cfg["fp16"] = {"enabled": True}
            cfg.pop("bf16", None)
        return cfg

    # ------------------------------------------------------------------
    # 恢复
    # ------------------------------------------------------------------

    def _maybe_resume(self) -> None:
        path = self.config.resume_from_checkpoint
        if path is None:
            return
        if path == "auto":
            ck = self.ckpt_mgr.find_latest()
            if ck is None:
                logger.info("[Resume] no checkpoint found, starting fresh")
                return
            path = ck
        state = self.ckpt_mgr.load(
            path, model=self.model,
            optimizer=self.optimizer if self.engine is None else None,
            scheduler=self.scheduler if self.engine is None else None,
        )
        self.global_step = state.get("step", 0)
        self.epoch = state.get("epoch", 0)
        self.tokens_seen = state.get("tokens_seen", 0)
        self.best_metric = state.get("best_metric", None)
        if is_main_process():
            logger.info(f"[Resume] from {path}, step={self.global_step}, epoch={self.epoch}")

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.setup()
        self.start_time = time.time()
        cfg = self.config

        if is_main_process():
            logger.info(
                f"[Train] start: max_steps={cfg.max_steps}, "
                f"micro_bs={cfg.micro_batch_size}, "
                f"grad_accum={cfg.gradient_accumulation_steps}, "
                f"effective_bs={cfg.micro_batch_size * cfg.gradient_accumulation_steps * get_world_size()}"
            )

        train_iter = self._infinite_loader(self.train_loader)
        self.model.train()

        while self.global_step < cfg.max_steps:
            metrics = self._train_one_step(train_iter)
            self.global_step += 1

            if self.global_step % cfg.log_steps == 0 and is_main_process():
                self._log_step(metrics)

            # eval
            if cfg.eval_steps > 0 and self.global_step % cfg.eval_steps == 0:
                eval_metrics = self.evaluate()
                if is_main_process() and eval_metrics:
                    self._log_eval(eval_metrics)
                    metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})

            # 信号驱动
            PAUSE_CONTROLLER.wait_if_paused()
            if PAUSE_CONTROLLER.should_save_and_exit():
                self._save_ckpt(metric=None, force=True)
                if is_main_process():
                    logger.warning("[Trainer] save-and-exit signal received, exiting cleanly")
                break

            # 保存
            if cfg.save_steps > 0 and self.global_step % cfg.save_steps == 0:
                metric = metrics.get(f"eval/{cfg.metric_name}") or metrics.get(cfg.metric_name)
                self._save_ckpt(metric=metric)

            self.on_step_end(metrics)

        self._save_ckpt(metric=None, force=True)
        if self.tracker_logger is not None:
            self.tracker_logger.finish()
        if is_main_process():
            elapsed = time.time() - self.start_time
            logger.info(f"[Train] done in {format_time(elapsed)}, total_steps={self.global_step}")

    # ------------------------------------------------------------------
    # 单步逻辑
    # ------------------------------------------------------------------

    def _train_one_step(self, data_iter: Iterator) -> Dict[str, float]:
        cfg = self.config
        accum = cfg.gradient_accumulation_steps
        accumulated: Dict[str, float] = {}
        total_tokens = 0

        for micro in range(accum):
            with self.timer.track("data"):
                batch = next(data_iter)
                batch = self.prepare_inputs(batch)

            if "input_ids" in batch:
                total_tokens += batch["input_ids"].numel()

            with self.timer.track("forward"):
                if self.engine is not None:
                    outputs = self.compute_loss(batch)
                    loss = outputs["loss"]
                else:
                    autocast_ctx = self._autocast_ctx()
                    sync_ctx = self._ddp_sync_ctx(is_last_micro=(micro == accum - 1))
                    with sync_ctx, autocast_ctx:
                        outputs = self.compute_loss(batch)
                        loss = outputs["loss"] / accum

            with self.timer.track("backward"):
                if self.engine is not None:
                    self.engine.backward(loss)
                else:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            for k, v in outputs.items():
                if k == "loss":
                    accumulated[k] = accumulated.get(k, 0.0) + (v.detach().item() / accum)
                elif isinstance(v, torch.Tensor) and v.ndim == 0:
                    accumulated[k] = accumulated.get(k, 0.0) + (v.detach().item() / accum)

        # ---------- step ----------
        with self.timer.track("optim"):
            if self.engine is not None:
                self.engine.step()
                lr_now = self.engine.get_lr()[0] if hasattr(self.engine, "get_lr") else self.config.learning_rate
            else:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=cfg.max_grad_norm,
                )
                accumulated["grad_norm"] = float(grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                lr_now = self.scheduler.get_last_lr()[0]

        accumulated["lr"] = lr_now

        # tokens 全 rank 求和
        if is_distributed():
            t = torch.tensor(total_tokens, device=self.device, dtype=torch.long)
            from deepseek_v4.distributed.utils import all_reduce_sum
            t = all_reduce_sum(t)
            total_tokens = int(t.item())
        self.tokens_seen += total_tokens
        accumulated["tokens"] = total_tokens

        self.metric_logger.update(**{k: v for k, v in accumulated.items() if isinstance(v, (int, float))})
        return accumulated

    def _autocast_ctx(self):
        if self.config.precision in ("fp16", "bf16"):
            return torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=self.config.precision_dtype(),
            )
        return nullcontext()

    def _ddp_sync_ctx(self, is_last_micro: bool):
        """grad accumulation 时跳过非最后一步的 DDP 同步以减少通信。"""
        if not is_distributed() or is_last_micro or self.engine is not None:
            return nullcontext()
        return self.model.no_sync() if hasattr(self.model, "no_sync") else nullcontext()

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}
        self.model.eval()

        sums: Dict[str, float] = {}
        n = 0
        autocast_ctx = self._autocast_ctx()
        for i, batch in enumerate(self.eval_loader):
            if i >= self.config.eval_max_batches:
                break
            batch = self.prepare_inputs(batch)
            with autocast_ctx:
                outputs = self.compute_loss(batch)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    sums[k] = sums.get(k, 0.0) + float(v.detach().item())
            n += 1

        avg = {k: v / max(n, 1) for k, v in sums.items()}
        # 跨 rank 聚合
        if is_distributed():
            for k, v in list(avg.items()):
                t = torch.tensor(v, device=self.device)
                avg[k] = float(all_reduce_mean(t).item())
        self.model.train()
        return avg

    # ------------------------------------------------------------------
    # 日志 / 保存
    # ------------------------------------------------------------------

    @only_on_main
    def _log_step(self, metrics: Dict[str, float]) -> None:
        smoothed = self.metric_logger.items()
        msg = (
            f"step {self.global_step:>6d} "
            f"loss={smoothed.get('loss', 0):.4f}  "
            f"lr={smoothed.get('lr', 0):.2e}  "
            f"|g|={smoothed.get('grad_norm', 0):.2f}  "
            f"tok/s={self.tokens_seen / max(time.time() - self.start_time, 1):.0f}  "
            f"elapsed={format_time(time.time() - self.start_time)}"
        )
        logger.info(msg)
        if self.tracker_logger is not None:
            self.tracker_logger.log({**smoothed, "step": self.global_step}, step=self.global_step)

    @only_on_main
    def _log_eval(self, metrics: Dict[str, float]) -> None:
        msg = "[Eval] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(msg)
        if self.tracker_logger is not None:
            self.tracker_logger.log(
                {f"eval/{k}": v for k, v in metrics.items()}, step=self.global_step,
            )

    def _save_ckpt(self, metric: Optional[float], force: bool = False) -> None:
        # 评估当前 metric
        if metric is None:
            metric = self.metric_logger.get(self.config.metric_name)

        # 仅在新最好 / 间隔 / 强制时保存
        is_better = False
        if metric is not None:
            if self.best_metric is None:
                is_better = True
            elif self.config.metric_mode == "min" and metric < self.best_metric:
                is_better = True
            elif self.config.metric_mode == "max" and metric > self.best_metric:
                is_better = True
        if is_better:
            self.best_metric = metric

        self.ckpt_mgr.save(
            model=self.model,
            optimizer=self.optimizer if self.engine is None else None,
            scheduler=self.scheduler if self.engine is None else None,
            step=self.global_step,
            epoch=self.epoch,
            metric_value=metric,
            extra_state={
                "tokens_seen": self.tokens_seen,
                "best_metric": self.best_metric,
            },
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _infinite_loader(self, loader: DataLoader) -> Iterator:
        epoch = 0
        while True:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for batch in loader:
                yield batch
            epoch += 1
            self.epoch = epoch
            if self.config.max_epochs and epoch >= self.config.max_epochs:
                break
