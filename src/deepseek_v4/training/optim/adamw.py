"""
从 0 实现的 AdamW（解耦权重衰减）。

公式（Loshchilov & Hutter, 2019）：
    m_t = β1 m_{t-1} + (1-β1) g_t
    v_t = β2 v_{t-1} + (1-β2) g_t²
    m̂_t = m_t / (1 - β1^t)
    v̂_t = v_t / (1 - β2^t)
    θ_t = θ_{t-1} - lr · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})

要点：
- 权重衰减解耦：不经过 m/v，而是直接缩放参数。
- bias_correction 默认开启（与 PyTorch 一致）。
- 支持 fused/foreach 模式（性能优化）。
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    从 0 实现的 AdamW，行为与 torch.optim.AdamW 等价。

    Args:
        params:        参数（或 param groups）
        lr:            初始学习率
        betas:         动量系数 (β1, β2)
        eps:           分母数值稳定项
        weight_decay:  解耦权重衰减系数 λ
        amsgrad:       是否使用 AMSGrad 变体
        foreach:       是否启用 foreach 加速（多张量同时更新）
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        foreach: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            amsgrad=amsgrad, foreach=foreach,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[int] = []

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]
            foreach = group["foreach"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                state_steps.append(state["step"])

            if not params_with_grad:
                continue

            if foreach and torch.cuda.is_available():
                self._foreach_update(
                    params_with_grad, grads,
                    exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                    state_steps,
                    beta1=beta1, beta2=beta2, lr=lr, eps=eps,
                    weight_decay=wd, amsgrad=amsgrad,
                )
            else:
                self._single_update(
                    params_with_grad, grads,
                    exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                    state_steps,
                    beta1=beta1, beta2=beta2, lr=lr, eps=eps,
                    weight_decay=wd, amsgrad=amsgrad,
                )

        return loss

    # ------------------------------------------------------------------

    @staticmethod
    def _single_update(
        params: List[Tensor], grads: List[Tensor],
        exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *, beta1: float, beta2: float, lr: float, eps: float,
        weight_decay: float, amsgrad: bool,
    ) -> None:
        for i, p in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # decoupled weight decay
            if weight_decay != 0:
                p.mul_(1 - lr * weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            if amsgrad:
                max_avg_sq = max_exp_avg_sqs[i]
                torch.maximum(max_avg_sq, exp_avg_sq, out=max_avg_sq)
                denom = (max_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)

            step_size = lr / bc1
            p.addcdiv_(exp_avg, denom, value=-step_size)

    @staticmethod
    def _foreach_update(
        params: List[Tensor], grads: List[Tensor],
        exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *, beta1: float, beta2: float, lr: float, eps: float,
        weight_decay: float, amsgrad: bool,
    ) -> None:
        """foreach 版本：一次性更新所有 tensor，CUDA 上速度更快。"""
        # decoupled weight decay
        if weight_decay != 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        # m_t = β1 m_{t-1} + (1-β1) g_t
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        # v_t = β2 v_{t-1} + (1-β2) g_t²
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        bc1s = [1 - beta1 ** s for s in state_steps]
        bc2s = [1 - beta2 ** s for s in state_steps]

        if amsgrad:
            torch._foreach_maximum_(max_exp_avg_sqs, exp_avg_sqs)
            denom = torch._foreach_div(
                torch._foreach_sqrt(max_exp_avg_sqs),
                [math.sqrt(b) for b in bc2s],
            )
        else:
            denom = torch._foreach_div(
                torch._foreach_sqrt(exp_avg_sqs),
                [math.sqrt(b) for b in bc2s],
            )
        torch._foreach_add_(denom, eps)

        step_sizes = [-lr / b for b in bc1s]
        torch._foreach_addcdiv_(params, exp_avgs, denom, step_sizes)
