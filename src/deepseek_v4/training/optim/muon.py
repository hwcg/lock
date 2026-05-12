"""
Muon 优化器：Newton-Schulz 正交化梯度的动量方法。

适用于 2D 参数（矩阵权重），把动量方向通过 Newton-Schulz 迭代正交化，
显著改善大模型训练。1D 参数请用 AdamW。

参考：Keller Jordan, et al. (2024). Muon: An optimizer for hidden layers in neural networks.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


@torch.no_grad()
def newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    5-th order Newton-Schulz 迭代，将矩阵 G "推" 向其极分解中的正交因子。

    论文给出的定常系数 (a, b, c) = (3.4445, -4.7750, 2.0315) 在 5 步内
    把奇异值近似拉平到 1。

    Args:
        G:    任意 2D tensor，[m, n]
        steps: 迭代步数（默认 5）
        eps:   norm 归一的数值稳定项
    Returns:
        与 G 同 shape，但奇异值已正交化
    """
    assert G.ndim == 2, f"newton_schulz_5 requires 2D, got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.bfloat16()
    if X.size(0) > X.size(1):
        # 始终在"短边×长边"上计算，节省 FLOPs
        X = X.T
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon 优化器。

    每步：
        m_t = μ m_{t-1} + g_t                     # 标准动量
        u_t = m_t (Nesterov: g_t + μ m_t)
        u_t = NewtonSchulz(u_t)                    # 关键：正交化方向
        θ ← θ - lr · √max(m,n) · scale · u_t

    所有参数必须是 2D 矩阵；其他参数请放进 AdamW。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        update_scale: float = 0.2,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, update_scale=update_scale,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            scale = group["update_scale"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(
                        f"Muon only supports 2D params, got {tuple(grad.shape)}. "
                        "Use AdamW for non-matrix parameters."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                # Newton-Schulz 正交化
                update = newton_schulz_5(update, steps=ns_steps)

                # 缩放：补偿正交化后 update 的"幅度"
                fan = max(update.size(-2), update.size(-1))
                effective_lr = lr * math.sqrt(fan) * scale

                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.sub_(update, alpha=effective_lr)

        return loss
