"""训练模块单测：BaseTrainer / Optim / Scheduler / Checkpoint。"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from deepseek_v4.training.base_trainer import BaseTrainer, TrainerConfig
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.training.optim import (
    AdamW, CosineWarmupScheduler, LinearWarmupScheduler, Muon, WSDScheduler,
    build_optimizer, build_scheduler, group_parameters,
)
from deepseek_v4.training.grad_checkpoint import enable_gradient_checkpointing


# ============================================================
# 简单模型
# ============================================================

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 2)
        self.embed = nn.Embedding(100, 32)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        return {"logits": x, "loss": None}


# ============================================================
# TrainerConfig
# ============================================================

def test_trainer_config_defaults():
    cfg = TrainerConfig(output_dir="/tmp/test", run_name="test")
    assert cfg.learning_rate == 3e-4
    assert cfg.precision == "bf16"
    assert cfg.max_grad_norm == 1.0
    assert cfg.scheduler == "cosine"


def test_trainer_config_precision_dtype():
    cfg = TrainerConfig()
    assert cfg.precision_dtype() == torch.bfloat16
    cfg.precision = "fp32"
    assert cfg.precision_dtype() == torch.float32
    cfg.precision = "fp16"
    assert cfg.precision_dtype() == torch.float16


def test_trainer_config_to_dict():
    cfg = TrainerConfig(output_dir="/tmp/x")
    d = cfg.to_dict()
    assert d["output_dir"] == "/tmp/x"
    assert "learning_rate" in d


# ============================================================
# Optimizers
# ============================================================

def test_build_adamw():
    model = _SimpleModel()
    opt = build_optimizer(model, name="adamw", lr=1e-4, weight_decay=0.01)
    assert isinstance(opt, AdamW) or isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) >= 1


def test_build_muon():
    model = _SimpleModel()
    opt = build_optimizer(model, name="muon", lr=1e-4, weight_decay=0.01)
    assert isinstance(opt, torch.optim.Optimizer)


def test_group_parameters():
    model = _SimpleModel()
    groups = group_parameters(model, weight_decay=0.01, base_lr=1e-4)
    assert len(groups) >= 1


def test_adamw_importable():
    assert AdamW is not None
    opt = AdamW([{"params": [torch.nn.Parameter(torch.randn(3, 3))], "lr": 1e-3}], lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)


# ============================================================
# Schedulers
# ============================================================

def test_cosine_scheduler():
    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = CosineWarmupScheduler(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert sched.get_last_lr()[0] > 0
    for _ in range(20):
        opt.step()
        sched.step()
    lr = sched.get_last_lr()[0]
    assert lr > 0


def test_linear_scheduler():
    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = LinearWarmupScheduler(opt, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    for _ in range(20):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] > 0


def test_wsd_scheduler():
    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = WSDScheduler(opt, warmup_steps=10, total_steps=100, decay_ratio=0.1, min_lr_ratio=0.1)
    for _ in range(20):
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] > 0


def test_build_scheduler_cosine():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=1e-3)
    sched = build_scheduler(opt, name="cosine", warmup_steps=5, total_steps=50)
    assert sched is not None


# ============================================================
# Checkpoint
# ============================================================

def test_checkpoint_manager_save_load(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(
        output_dir=str(ckpt_dir),
        keep_last=3,
        keep_best=2,
        metric_name="loss",
        metric_mode="min",
        save_format="safetensors",
    )

    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters())

    path = mgr.save(model=model, optimizer=opt, step=100, metric_value=1.5)
    assert Path(path).exists()

    state = mgr.load(path, model=_SimpleModel(), optimizer=None)
    assert state["step"] == 100
    assert state["metric_value"] == 1.5


def test_checkpoint_manager_find_latest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(
        output_dir=str(ckpt_dir),
        keep_last=5,
        keep_best=2,
        metric_name="loss",
        metric_mode="min",
        save_format="safetensors",
    )

    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters())

    mgr.save(model=model, optimizer=opt, step=10, metric_value=3.0)
    mgr.save(model=model, optimizer=opt, step=20, metric_value=2.0)
    mgr.save(model=model, optimizer=opt, step=30, metric_value=1.0)

    latest = mgr.find_latest()
    assert latest is not None
    state = mgr.load(latest, model=_SimpleModel(), optimizer=None)
    assert state["step"] == 30


def test_checkpoint_manager_keep_last(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(
        output_dir=str(ckpt_dir),
        keep_last=2,
        keep_best=1,
        metric_name="loss",
        metric_mode="min",
        save_format="pytorch",
    )

    model = _SimpleModel()
    opt = torch.optim.AdamW(model.parameters())

    for s in range(10, 60, 10):
        mgr.save(model=model, optimizer=opt, step=s, metric_value=float(100 - s))

    # 应只保留最后 2 个（step=40, 50）+ 最好 1 个（step=50, 因为是 min）
    saved = list(ckpt_dir.glob("step_*"))
    assert len(saved) <= 3


# ============================================================
# Gradient Checkpointing
# ============================================================

def test_gradient_checkpointing_enable():
    model = _SimpleModel()

    class _DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, x):
            return self.linear(x)

    model.layers = nn.ModuleList([_DummyLayer() for _ in range(4)])
    n = enable_gradient_checkpointing(model, skip_first_n=0, skip_last_n=0)
    assert n == 4

    n2 = enable_gradient_checkpointing(model, skip_first_n=1, skip_last_n=1)
    assert n2 == 2


# ============================================================
# BaseTrainer smoke test
# ============================================================

class _SmokeTrainer(BaseTrainer):
    """最小可运行训练器。"""

    def get_train_dataset(self):
        class _DummyDS:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return {"input_ids": torch.randint(0, 50, (8,)), "labels": torch.randint(0, 50, (8,))}
        return _DummyDS()

    def compute_loss(self, batch):
        loss = torch.tensor(0.0, requires_grad=True)
        return {"loss": loss}

    def get_collator(self):
        return lambda x: {"input_ids": torch.stack([i["input_ids"] for i in x]),
                          "labels": torch.stack([i["labels"] for i in x])}


def test_base_trainer_smoke():
    model = _SimpleModel()
    cfg = TrainerConfig(
        output_dir="/tmp/test_smoke_trainer",
        max_steps=2,
        eval_steps=0,
        save_steps=0,
        log_steps=1,
        gradient_checkpointing=False,
        enable_signal_control=False,
        logger_backends=[],
        distributed_backend="gloo",
        print_param_count=False,
    )
    trainer = _SmokeTrainer(config=cfg, model=model)
    trainer.setup()
    # 跑 2 步
    for _ in range(2):
        batch = next(iter(trainer.train_loader))
        batch = trainer.prepare_inputs(batch)
        out = trainer.compute_loss(batch)
        assert "loss" in out
