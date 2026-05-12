"""
从 0 实现的 LoRA 层。

支持三种基础层：
1. nn.Linear           → LoRALinear
2. 3D Parameter (MoE) → LoRAExpertProj
3. V4 GroupedLinear    → LoRAGroupedLinear

每个适配器都：
- 保留原层引用，原 weight 冻结
- 增量 = scaling * dropout(x) @ A.T @ B.T   (对 nn.Linear)
- 支持 merge_weights(): 把 BA 合并回原 weight 后释放 LoRA 参数
- 支持 DoRA: 把 weight 的 magnitude 单独学习

数学（标准 LoRA）：
    y = x W^T + b + scaling · (x A^T) B^T
    A: [r, in_features]   初始 kaiming_uniform
    B: [out_features, r]  初始 0
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Linear → LoRA
# ============================================================

class LoRALinear(nn.Module):
    """
    nn.Linear 的 LoRA 适配（不支持 4-bit 量化基底；如需 QLoRA 见 layers_qlora.py）。
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_dora: bool = False,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0")
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.use_dora = use_dora
        self.merged = False

        # 冻结原参数
        for p in self.base_layer.parameters():
            p.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # LoRA 参数
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if init_lora_weights:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # B 已初始化为 0，保证 t=0 时 LoRA 无影响

        # DoRA：保留原 weight 列范数作为可学习的 magnitude
        if use_dora:
            with torch.no_grad():
                # 列范数（每个 output dim 的范数）
                col_norm = base_layer.weight.norm(dim=1, keepdim=True)  # [out, 1]
            self.dora_magnitude = nn.Parameter(col_norm.clone().to(dtype=dtype, device=device))

    # -------- forward --------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base_layer(x)

        base_out = self.base_layer(x)

        if self.r > 0:
            # ΔW = scaling · B @ A
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)   # [..., r]
            lora_out = F.linear(lora_out, self.lora_B) * self.scaling

            if not self.use_dora:
                return base_out + lora_out

            # DoRA：y = m * (W + ΔW) / ||W + ΔW||_col @ x
            # 实现技巧：复用 base_out + lora_out，再按列范数归一化乘 magnitude
            new_weight = self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A)
            new_norm = new_weight.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scale = self.dora_magnitude / new_norm
            # base_out 含原 W·x；lora_out 含 ΔW·x。两者之和 = (W+ΔW)·x
            full_out = base_out + lora_out
            # 按 output 维度逐列乘 scale
            return full_out * scale.squeeze(-1).unsqueeze(0).expand_as(full_out) \
                if full_out.dim() == 2 else \
                full_out * scale.view(*([1]*(full_out.dim()-1)), -1)
        return base_out

    # -------- merge / unmerge --------

    @torch.no_grad()
    def merge_weights(self) -> None:
        """把 LoRA 增量合并到 base weight。"""
        if self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        if self.use_dora:
            new_weight = self.base_layer.weight + delta
            new_norm = new_weight.norm(dim=1, keepdim=True).clamp(min=1e-6)
            scale = self.dora_magnitude / new_norm
            new_weight = new_weight * scale
            self.base_layer.weight.data.copy_(new_weight)
        else:
            self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge_weights(self) -> None:
        if not self.merged:
            return
        if self.use_dora:
            raise RuntimeError("DoRA does not support unmerge")
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base_layer.weight.data.sub_(delta.to(self.base_layer.weight.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, r={self.r}, "
            f"alpha={self.lora_alpha}, scaling={self.scaling:.3f}, "
            f"dora={self.use_dora}, merged={self.merged}"
        )


# ============================================================
# MoE Expert 3D Parameter → LoRA
# ============================================================

class LoRAExpertProj(nn.Module):
    """
    给 V4 的 DeepseekV4Experts.gate_up_proj / down_proj 这种 3D Parameter 加 LoRA。

    原参数：
        gate_up_proj: [num_experts, 2*intermediate, hidden]
        down_proj:    [num_experts, hidden, intermediate]

    LoRA 改为每 expert 独立的低秩分解：
        ΔW_e = scaling · B_e @ A_e
        A: [num_experts, r, in_dim]
        B: [num_experts, out_dim, r]
    """

    def __init__(
        self,
        base_param: nn.Parameter,           # [E, out_dim, in_dim]
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ):
        super().__init__()
        assert base_param.dim() == 3, f"expected 3D, got {base_param.shape}"
        self.base_param = base_param  # 保持引用（不在 self 注册以避免参数重复，外部仍持有）
        # base 必须冻结
        base_param.requires_grad = False

        E, out_dim, in_dim = base_param.shape
        self.num_experts = E
        self.out_features = out_dim
        self.in_features = in_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.merged = False

        device = base_param.device
        dtype = base_param.dtype

        self.lora_A = nn.Parameter(torch.empty(E, r, in_dim, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(E, out_dim, r, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if init_lora_weights:
            for e in range(E):
                nn.init.kaiming_uniform_(self.lora_A[e], a=math.sqrt(5))

    # -------- 单 expert 调用接口（提供给被 monkey-patch 的 forward） --------

    def delta_weight(self, expert_idx: int) -> torch.Tensor:
        """计算第 expert_idx 个 expert 的增量 weight。"""
        if self.merged:
            return torch.zeros_like(self.base_param[expert_idx])
        return self.scaling * (self.lora_B[expert_idx] @ self.lora_A[expert_idx])

    @torch.no_grad()
    def merge_weights(self) -> None:
        if self.merged:
            return
        for e in range(self.num_experts):
            self.base_param.data[e].add_(self.delta_weight(e).to(self.base_param.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge_weights(self) -> None:
        if not self.merged:
            return
        for e in range(self.num_experts):
            self.base_param.data[e].sub_(self.delta_weight(e).to(self.base_param.dtype))
        self.merged = False


# ============================================================
# GroupedLinear → LoRA（V4 输出投影 LoRA）
# ============================================================

class LoRAGroupedLinear(nn.Module):
    """
    适配 DeepseekV4GroupedLinear（块对角分组线性）。

    原层：
        weight: [n_groups * out_per_group, in_per_group]
        forward: x reshape [..., n_groups, in_per_group] → bmm

    LoRA：每个 group 一份独立的 (A, B)
        A: [n_groups, r, in_per_group]
        B: [n_groups, out_per_group, r]

    工作方式：替换原 forward，在 bmm 之后加 LoRA 增量。
    """

    def __init__(
        self,
        base_layer: nn.Module,         # DeepseekV4GroupedLinear
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        use_rslora: bool = False,
        init_lora_weights: bool = True,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.n_groups = base_layer.n_groups
        in_per_group = base_layer.in_features
        out_per_group = base_layer.out_features // self.n_groups
        self.in_per_group = in_per_group
        self.out_per_group = out_per_group
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / (math.sqrt(r) if use_rslora else r)
        self.merged = False

        for p in self.base_layer.parameters():
            p.requires_grad = False

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty(self.n_groups, r, in_per_group, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.n_groups, out_per_group, r, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if init_lora_weights:
            for g in range(self.n_groups):
                nn.init.kaiming_uniform_(self.lora_A[g], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_per_group]
        base_out = self.base_layer(x)
        if self.merged or self.r == 0:
            return base_out
        # 增量
        # 输入维度：(B*S, n_groups, in_per_group)
        original_shape = x.shape
        x_flat = self.lora_dropout(x).reshape(-1, self.n_groups, self.in_per_group).transpose(0, 1)  # [G, N, in]
        # A: [G, r, in]  → A^T: [G, in, r]
        z = torch.bmm(x_flat, self.lora_A.transpose(1, 2))   # [G, N, r]
        delta = torch.bmm(z, self.lora_B.transpose(1, 2))    # [G, N, out_per_group]
        delta = delta.transpose(0, 1)                        # [N, G, out]
        delta = delta.reshape(*original_shape[:-1], self.out_per_group) * self.scaling
        return base_out + delta

    @torch.no_grad()
    def merge_weights(self) -> None:
        if self.merged:
            return
        # 原 weight: [G * out_per_group, in_per_group] (实际 view 为 [G, out_per_group, in_per_group])
        for g in range(self.n_groups):
            delta = self.scaling * (self.lora_B[g] @ self.lora_A[g])  # [out, in]
            start = g * self.out_per_group
            end = start + self.out_per_group
            self.base_layer.weight.data[start:end].add_(delta.to(self.base_layer.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge_weights(self) -> None:
        if not self.merged:
            return
        for g in range(self.n_groups):
            delta = self.scaling * (self.lora_B[g] @ self.lora_A[g])
            start = g * self.out_per_group
            end = start + self.out_per_group
            self.base_layer.weight.data[start:end].sub_(delta.to(self.base_layer.weight.dtype))
        self.merged = False
