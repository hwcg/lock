"""Reward 函数基础抽象。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch


# ============================================================
# 抽象接口
# ============================================================

class RewardFunction:
    """
    Reward 函数协议。

    一个 RewardFunction 是 callable，签名：
        __call__(completions, references=None, prompts=None, **kwargs) -> List[float]
    """
    name: str = "reward"

    def __call__(
        self,
        completions: List[str],
        references: Optional[List[Any]] = None,
        prompts: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[float]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ============================================================
# 常见包装器
# ============================================================

class NamedReward(RewardFunction):
    """把任意 callable 包成 RewardFunction。"""

    def __init__(self, fn: Callable[..., List[float]], name: str):
        self.fn = fn
        self.name = name

    def __call__(self, completions, references=None, prompts=None, **kwargs):
        return self.fn(completions=completions, references=references, prompts=prompts, **kwargs)


class ConstantReward(RewardFunction):
    """返回常数（debug 用）。"""

    def __init__(self, value: float = 0.0, name: str = "constant"):
        self.value = value
        self.name = name

    def __call__(self, completions, references=None, prompts=None, **kwargs):
        return [float(self.value)] * len(completions)


class CompositeReward(RewardFunction):
    """
    线性组合多个 reward：
        score_i = Σ_k w_k · r_k(sample_i)

    返回总分（按需也返回各子项 via return_detail）。
    """

    def __init__(
        self,
        rewards: List[RewardFunction],
        weights: Optional[List[float]] = None,
        name: str = "composite",
    ):
        if weights is None:
            weights = [1.0] * len(rewards)
        if len(weights) != len(rewards):
            raise ValueError("rewards / weights 长度不一致")
        self.rewards = rewards
        self.weights = weights
        self.name = name

    def __call__(
        self,
        completions,
        references=None,
        prompts=None,
        return_detail: bool = False,
        **kwargs,
    ) -> Union[List[float], Dict[str, List[float]]]:
        details: Dict[str, List[float]] = {}
        n = len(completions)
        totals = [0.0] * n
        for w, fn in zip(self.weights, self.rewards):
            scores = fn(completions=completions, references=references, prompts=prompts, **kwargs)
            assert len(scores) == n, f"{fn.name} returned {len(scores)} != {n}"
            details[fn.name] = scores
            for i, s in enumerate(scores):
                totals[i] += w * float(s)
        if return_detail:
            details["_total"] = totals
            return details
        return totals


# ============================================================
# 配置驱动构造
# ============================================================

def build_reward_from_config(
    cfg_list: List[Dict[str, Any]],
    registry: Optional[Dict[str, Callable[..., RewardFunction]]] = None,
) -> CompositeReward:
    """
    从配置列表构造 CompositeReward。

    每项 dict 形如：
        { name: "math_correctness", weight: 1.0, params: { ... } }
    """
    if registry is None:
        from deepseek_v4.training.rewards.format import (
            length_reward, repetition_penalty_reward, format_reward,
            thinking_format_reward, tool_call_format_reward,
            json_format_reward, regex_reward,
        )
        from deepseek_v4.training.rewards.math_ import (
            math_correctness_reward, boxed_reward, gsm8k_answer_reward,
        )
        from deepseek_v4.training.rewards.code import (
            code_python_reward, code_execute_reward,
        )
        registry = {
            "length":           length_reward,
            "repetition":       repetition_penalty_reward,
            "format":           format_reward,
            "thinking_format":  thinking_format_reward,
            "tool_call_format": tool_call_format_reward,
            "json_format":      json_format_reward,
            "regex":            regex_reward,
            "math_correctness": math_correctness_reward,
            "boxed":            boxed_reward,
            "gsm8k":            gsm8k_answer_reward,
            "code_python":      code_python_reward,
            "code_execute":     code_execute_reward,
            "constant":         lambda **kw: ConstantReward(**kw),
        }

    rewards: List[RewardFunction] = []
    weights: List[float] = []
    for item in cfg_list:
        name = item["name"]
        weight = float(item.get("weight", 1.0))
        params = item.get("params", {}) or {}
        builder = registry.get(name)
        if builder is None:
            raise KeyError(f"Unknown reward: {name}")
        fn = builder(**params)
        if not isinstance(fn, RewardFunction):
            fn = NamedReward(fn, name=name)
        rewards.append(fn)
        weights.append(weight)
    return CompositeReward(rewards=rewards, weights=weights, name="composite")
