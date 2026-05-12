"""
全模型 LoRA 注入与管理。

API：
    apply_lora(model, config)           → 修改 model in-place，返回 model
    set_lora_trainable(model, mode)     → 把 LoRA / bias / modules_to_save 设为可训练
    print_trainable_parameters(model)
    get_lora_state_dict(model)          → 仅 LoRA / bias / modules_to_save 的 state_dict
    set_lora_state_dict(model, sd)
    save_lora(model, output_dir)
    load_lora(model, adapter_dir)
    merge_and_unload(model)             → 合并 LoRA 后还原成普通 model
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from deepseek_v4.training.lora.config import LoRAConfig
from deepseek_v4.training.lora.layers import (
    LoRAExpertProj, LoRAGroupedLinear, LoRALinear,
)
from deepseek_v4.utils.io import safe_load_json, safe_save_json
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 辅助：找到模块 / 替换模块
# ============================================================

def _get_submodule(model: nn.Module, qualified_name: str) -> nn.Module:
    """根据点号路径取子模块（'a.b.0.c' → model.a.b[0].c）"""
    obj = model
    for p in qualified_name.split("."):
        if p == "":
            continue
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def _set_submodule(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.rsplit(".", 1)
    if len(parts) == 1:
        setattr(model, parts[0], new_module)
        return
    parent = _get_submodule(model, parts[0])
    if parts[1].isdigit():
        parent[int(parts[1])] = new_module
    else:
        setattr(parent, parts[1], new_module)


def _matches(name: str, patterns: List[str]) -> bool:
    return any(p in name for p in patterns)


# ============================================================
# 注入 LoRA
# ============================================================

def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    给 model 注入 LoRA。

    支持三类目标：
    - nn.Linear           → 替换为 LoRALinear
    - DeepseekV4GroupedLinear（V4 输出投影） → 替换为 LoRAGroupedLinear
    - DeepseekV4Experts.gate_up_proj/down_proj（3D nn.Parameter） → 注入 LoRAExpertProj
    """
    # 延迟 import 避免循环
    from deepseek_v4.modeling.model import (
        DeepseekV4Experts,
        DeepseekV4GroupedLinear,
    )

    # ---- 1. 冻结全部 ----
    for p in model.parameters():
        p.requires_grad = False

    targets = config.target_modules
    n_lora_linear = 0
    n_lora_grouped = 0
    n_lora_expert = 0

    # ---- 2. 替换 nn.Linear / GroupedLinear ----
    to_replace: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not _matches(name, targets):
            continue
        if isinstance(module, DeepseekV4GroupedLinear):
            if config.target_grouped:
                to_replace.append((name, module))
        elif isinstance(module, nn.Linear):
            # router weight 默认不加 LoRA
            if "gate.weight" in name and not config.target_router:
                continue
            to_replace.append((name, module))

    for name, module in to_replace:
        if isinstance(module, DeepseekV4GroupedLinear):
            new = LoRAGroupedLinear(
                module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                use_rslora=config.use_rslora,
                init_lora_weights=config.init_lora_weights,
            )
            _set_submodule(model, name, new)
            n_lora_grouped += 1
        elif isinstance(module, nn.Linear):
            new = LoRALinear(
                module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                use_dora=config.use_dora,
                use_rslora=config.use_rslora,
                init_lora_weights=config.init_lora_weights,
            )
            _set_submodule(model, name, new)
            n_lora_linear += 1
        else:
            # 非 nn.Linear，跳过
            continue

    # ---- 3. 处理 MoE Experts ----
    if config.target_experts:
        for name, module in model.named_modules():
            if isinstance(module, DeepseekV4Experts):
                # 给 gate_up_proj / down_proj 各挂一个 LoRAExpertProj
                if hasattr(module, "gate_up_proj"):
                    adapter = LoRAExpertProj(
                        module.gate_up_proj,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        use_rslora=config.use_rslora,
                        init_lora_weights=config.init_lora_weights,
                    )
                    module.add_module("_lora_gate_up", adapter)
                    n_lora_expert += 1
                if hasattr(module, "down_proj"):
                    adapter = LoRAExpertProj(
                        module.down_proj,
                        r=config.r,
                        lora_alpha=config.lora_alpha,
                        use_rslora=config.use_rslora,
                        init_lora_weights=config.init_lora_weights,
                    )
                    module.add_module("_lora_down", adapter)
                    n_lora_expert += 1
                # 替换 forward
                _patch_experts_forward(module)

    # ---- 4. modules_to_save ----
    if config.modules_to_save:
        for name, p in model.named_parameters():
            if any(m in name for m in config.modules_to_save):
                p.requires_grad = True

    # ---- 5. bias 处理 ----
    set_lora_trainable(model, mode=config.bias)

    logger.info(
        f"[LoRA] applied: linear={n_lora_linear}, grouped={n_lora_grouped}, "
        f"expert={n_lora_expert}"
    )
    print_trainable_parameters(model)
    return model


# ============================================================
# Patch Experts.forward 让它使用 LoRA 增量
# ============================================================

def _patch_experts_forward(experts_module: nn.Module) -> None:
    """
    给 DeepseekV4Experts 替换 forward：在调用 F.linear 时附加 LoRA 增量。

    原 forward 中：
        gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
        current = self._apply_gate(gate_up)
        current = F.linear(current, self.down_proj[expert_idx]) * weights

    替换后：在每个 expert_idx 上加上 _lora_gate_up.delta_weight(idx) 等。
    """
    from deepseek_v4.modeling.model import DeepseekV4Experts
    if not isinstance(experts_module, DeepseekV4Experts):
        return
    if getattr(experts_module, "_lora_patched", False):
        return

    import torch.nn.functional as F

    original_forward = experts_module.forward

    def patched_forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=experts_module.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_lora = getattr(experts_module, "_lora_gate_up", None)
        down_lora = getattr(experts_module, "_lora_down", None)

        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])

            # gate_up
            w_gu = experts_module.gate_up_proj[expert_idx]
            if gate_up_lora is not None and not gate_up_lora.merged:
                w_gu = w_gu + gate_up_lora.delta_weight(int(expert_idx)).to(w_gu.dtype)
            gate_up = F.linear(hidden_states[token_idx], w_gu)
            current = experts_module._apply_gate(gate_up)

            # down
            w_dn = experts_module.down_proj[expert_idx]
            if down_lora is not None and not down_lora.merged:
                w_dn = w_dn + down_lora.delta_weight(int(expert_idx)).to(w_dn.dtype)
            current = F.linear(current, w_dn) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    experts_module.forward = patched_forward
    experts_module._lora_patched = True


# ============================================================
# 设置 trainable
# ============================================================

def set_lora_trainable(model: nn.Module, mode: str = "none") -> None:
    """
    标记 LoRA / bias 为可训练。

    Args:
        mode: "none" | "all" | "lora_only"
    """
    # 1. LoRA params 永远 trainable
    for n, p in model.named_parameters():
        if any(s in n for s in (".lora_A", ".lora_B", ".dora_magnitude", "_lora_gate_up", "_lora_down")):
            p.requires_grad = True

    # 2. bias 依据 mode
    if mode == "none":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = False
    elif mode == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif mode == "lora_only":
        # 只让和 LoRA 同名 module 内的 bias trainable
        for name, module in model.named_modules():
            if isinstance(module, (LoRALinear, LoRAGroupedLinear)):
                if hasattr(module.base_layer, "bias") and module.base_layer.bias is not None:
                    module.base_layer.bias.requires_grad = True


def print_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    logger.info(
        f"[LoRA] trainable: {trainable:,} / {total:,} ({pct:.4f}%)"
    )
    return trainable, total


# ============================================================
# state_dict 工具
# ============================================================

def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """仅 LoRA / dora_magnitude / 标记 trainable 的 bias / modules_to_save。"""
    sd: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        sd[name] = p.detach().cpu()
    return sd


def set_lora_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """部分 load。返回 (missing, unexpected)。"""
    own = dict(model.named_parameters())
    missing = []
    unexpected = []
    for k in sd:
        if k not in own:
            unexpected.append(k)
    for k in own:
        if own[k].requires_grad and k not in sd:
            missing.append(k)
        if k in sd:
            own[k].data.copy_(sd[k].to(own[k].device, dtype=own[k].dtype))
    return missing, unexpected


# ============================================================
# 保存 / 加载
# ============================================================

def save_lora(model: nn.Module, output_dir: Union[str, Path], config: Optional[LoRAConfig] = None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sd = get_lora_state_dict(model)
    try:
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in sd.items()}, str(output_dir / "adapter_model.safetensors"))
    except ImportError:
        torch.save(sd, output_dir / "adapter_model.bin")
    if config is not None:
        safe_save_json(output_dir / "adapter_config.json", config.to_dict())
    logger.info(f"[LoRA] saved adapter ({len(sd)} tensors) to {output_dir}")


def load_lora(model: nn.Module, adapter_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
    adapter_dir = Path(adapter_dir)
    st = adapter_dir / "adapter_model.safetensors"
    bin_ = adapter_dir / "adapter_model.bin"
    if st.exists():
        from safetensors.torch import load_file
        sd = load_file(str(st))
    elif bin_.exists():
        sd = torch.load(str(bin_), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter found in {adapter_dir}")
    missing, unexpected = set_lora_state_dict(model, sd)
    logger.info(f"[LoRA] loaded adapter from {adapter_dir} "
                f"(missing={len(missing)} unexpected={len(unexpected)})")
    return missing, unexpected


# ============================================================
# merge_and_unload
# ============================================================

def merge_and_unload(model: nn.Module) -> nn.Module:
    """
    合并所有 LoRA 增量到 base 权重，并把 LoRA 模块替换回原模块。

    返回干净的 model（无 LoRA 包装），可以直接当作普通 model 推理 / 导出。
    """
    # 1. 合并所有 LoRA*Linear / Grouped
    to_unwrap: List[Tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if isinstance(m, (LoRALinear, LoRAGroupedLinear)):
            m.merge_weights()
            to_unwrap.append((name, m))

    for name, m in to_unwrap:
        _set_submodule(model, name, m.base_layer)

    # 2. 合并 MoE expert LoRA
    for name, m in model.named_modules():
        for sub in ("_lora_gate_up", "_lora_down"):
            if hasattr(m, sub):
                lora_mod: LoRAExpertProj = getattr(m, sub)
                lora_mod.merge_weights()
                # 解绑 forward 不可逆，只能让 _lora_* 不再生效
        # 删除 lora 子模块
        for sub in ("_lora_gate_up", "_lora_down"):
            if hasattr(m, sub):
                delattr(m, sub)
        # 还原 forward
        from deepseek_v4.modeling.model import DeepseekV4Experts
        if isinstance(m, DeepseekV4Experts) and getattr(m, "_lora_patched", False):
            # 因 forward 是 closure，最简办法：重新创建 forward
            del m.forward     # 退回 class 定义
            try:
                del m._lora_patched
            except Exception:
                pass

    # 3. 解冻全部参数
    for p in model.parameters():
        p.requires_grad = True

    logger.info("[LoRA] merge_and_unload done")
    return model
