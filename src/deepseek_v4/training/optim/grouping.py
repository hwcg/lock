"""
参数分组 + 优化器构造。

要点（与 LLaMA / GPT-NeoX / DeepSeek 一致）：
1. 不衰减：bias、LayerNorm/RMSNorm、Embedding、router weight、attention sinks、HC base/scale
2. 衰减：所有 nn.Linear / 矩阵参数
3. Muon 仅作用于 2D matrix；其他放 AdamW
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer

from deepseek_v4.training.optim.adamw import AdamW
from deepseek_v4.training.optim.muon import Muon
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# 不参与 weight decay 的参数名 pattern
NO_DECAY_PATTERNS: Tuple[str, ...] = (
    r"\.bias$",
    r"\.weight$.*norm",         # LayerNorm/RMSNorm.weight 走 weight match 已解决
    r"layernorm",
    r"\.norm\.",
    r"input_layernorm",
    r"post_attention_layernorm",
    r"\.sinks$",
    r"\.position_bias$",
    r"e_score_correction_bias",
    r"hc_base",
    r"hc_scale",
    r"\.scale$",
    r"\.base$",
    r"tid2eid",
)
# 默认不进入优化器的参数（如 buffer 误注册为 Parameter 的）
EXCLUDE_PATTERNS: Tuple[str, ...] = (
    r"tid2eid",   # hash router 查表 buffer
)
NO_DECAY_RE = re.compile("|".join(NO_DECAY_PATTERNS), re.IGNORECASE)
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS))


def _should_decay(name: str, param: torch.Tensor) -> bool:
    """根据名字 & 形状决定该参数是否参与 weight decay。"""
    if NO_DECAY_RE.search(name):
        return False
    # 1D 参数（norm.weight / bias / sinks / per-expert bias）默认不衰减
    if param.ndim < 2:
        return False
    # Embedding：常规做法不衰减
    if "embed_tokens" in name:
        return False
    return True


def _is_excluded(name: str) -> bool:
    return bool(EXCLUDE_RE.search(name))


# ------------------------------------------------------------------------
# 标准 AdamW 分组
# ------------------------------------------------------------------------

def group_parameters(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: Optional[float] = None,
    embedding_lr_mult: float = 1.0,
    log: bool = True,
) -> List[Dict[str, Any]]:
    """
    把模型参数分为两组：decay / no_decay。

    Args:
        embedding_lr_mult: embedding 参数的 lr 倍率（一些论文建议用更小 lr）
    """
    decay: List[nn.Parameter] = []
    decay_names: List[str] = []
    no_decay: List[nn.Parameter] = []
    no_decay_names: List[str] = []
    embed: List[nn.Parameter] = []
    embed_names: List[str] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_excluded(name):
            continue
        if "embed_tokens" in name and embedding_lr_mult != 1.0:
            embed.append(p)
            embed_names.append(name)
        elif _should_decay(name, p):
            decay.append(p)
            decay_names.append(name)
        else:
            no_decay.append(p)
            no_decay_names.append(name)

    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    n_embed = sum(p.numel() for p in embed)

    if log:
        logger.info(
            f"[ParamGroups] decay={n_decay:,} ({len(decay)} tensors), "
            f"no_decay={n_no_decay:,} ({len(no_decay)} tensors), "
            f"embedding={n_embed:,} ({len(embed)} tensors)"
        )

    groups = []
    if decay:
        g = {"params": decay, "weight_decay": weight_decay, "name": "decay"}
        if learning_rate is not None:
            g["lr"] = learning_rate
        groups.append(g)
    if no_decay:
        g = {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"}
        if learning_rate is not None:
            g["lr"] = learning_rate
        groups.append(g)
    if embed:
        g = {
            "params": embed,
            "weight_decay": 0.0,
            "name": "embedding",
            "lr": (learning_rate or 0.0) * embedding_lr_mult,
        }
        groups.append(g)
    return groups


# ------------------------------------------------------------------------
# Muon 分组（2D 矩阵走 Muon，其他走 AdamW）
# ------------------------------------------------------------------------

def _is_muon_eligible(name: str, param: torch.Tensor) -> bool:
    """决定一个参数是否适合用 Muon。"""
    if param.ndim != 2:
        return False
    # embedding / lm_head 通常用 AdamW 更稳
    if "embed_tokens" in name or "lm_head" in name:
        return False
    # router weight 也用 AdamW（防 collapse）
    if "gate.weight" in name and "mlp" in name:
        return False
    # 3D experts 由专门包装，跳过
    if "gate_up_proj" in name or "down_proj" in name:
        # experts.gate_up_proj 是 3D，由 _is_muon_eligible 之前的 ndim 检查过滤
        return False
    return True


def split_for_muon(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> Tuple[List[nn.Parameter], List[Dict[str, Any]]]:
    """
    返回 (muon_params, adamw_groups)。

    muon_params: 仅 2D 矩阵且非 embedding / router
    adamw_groups: 其余 + decay/no_decay 分组
    """
    muon_params: List[nn.Parameter] = []
    adamw_decay: List[nn.Parameter] = []
    adamw_no_decay: List[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad or _is_excluded(name):
            continue
        if _is_muon_eligible(name, p):
            muon_params.append(p)
        elif _should_decay(name, p):
            adamw_decay.append(p)
        else:
            adamw_no_decay.append(p)

    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": weight_decay, "name": "decay"})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0, "name": "no_decay"})

    logger.info(
        f"[Muon Split] muon={sum(p.numel() for p in muon_params):,} "
        f"adamw_decay={sum(p.numel() for p in adamw_decay):,} "
        f"adamw_no_decay={sum(p.numel() for p in adamw_no_decay):,}"
    )
    return muon_params, adamw_groups


# ------------------------------------------------------------------------
# Optimizer 工厂
# ------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    muon_lr_mult: float = 0.4,
    muon_momentum: float = 0.95,
) -> Optimizer:
    """
    优化器工厂。

    name:
        adamw       → 标准 AdamW
        muon        → Muon (2D) + AdamW (其他)，包装为 ChainedOptimizer
    """
    name = name.lower()

    if name == "adamw":
        groups = group_parameters(model, weight_decay=weight_decay, learning_rate=lr)
        return AdamW(groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    if name == "muon":
        muon_params, adamw_groups = split_for_muon(model, weight_decay=weight_decay)
        if not muon_params:
            logger.warning("No params eligible for Muon, falling back to AdamW")
            return build_optimizer(model, name="adamw", lr=lr, weight_decay=weight_decay,
                                   betas=betas, eps=eps)
        muon = Muon(muon_params, lr=lr * muon_lr_mult, momentum=muon_momentum)
        adamw = AdamW(adamw_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return ChainedOptimizer([muon, adamw])

    raise ValueError(f"Unknown optimizer: {name}")


# ------------------------------------------------------------------------
# Chained Optimizer：把多个 optimizer 组合在一起
# ------------------------------------------------------------------------

class ChainedOptimizer(Optimizer):
    """
    把多个 Optimizer 透明组合：step / zero_grad / state_dict 全部委派到所有 children。

    与 PyTorch 原生 ChainedScheduler 不同，这是 optimizer 级别的合并。
    """

    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers
        # 合并 param_groups（用于 LRScheduler）
        self.param_groups: List[Dict[str, Any]] = []
        self.defaults: Dict[str, Any] = {}
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    @property
    def state(self):
        merged: Dict[Any, Any] = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for opt in self.optimizers:
            opt.step()
        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f"opt_{i}"])

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        # 默认加到第一个 optimizer
        self.optimizers[0].add_param_group(param_group)
