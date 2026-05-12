"""
Mode Router：决定走 thinking 还是 chat 模式。

两种模式：
- "fixed":   全部按用户指定模式
- "auto":    用 DifficultyEstimator 决定（hard → thinking, easy → chat）
- "learned": 用一个小模型预测（占位实现）
"""
from __future__ import annotations

from typing import List, Literal, Optional

from deepseek_v4.training.adaptive_thinking.difficulty import (
    DifficultyEstimator, RuleBasedDifficulty,
)


class ModeRouter:
    """
    决定每个 prompt 的 thinking_mode。

    Usage:
        router = ModeRouter(strategy="auto", estimator=RuleBasedDifficulty())
        modes = router.route(prompts)   # ["thinking", "chat", ...]
    """
    def __init__(
        self,
        strategy: Literal["fixed", "auto", "learned"] = "auto",
        default_mode: str = "chat",
        threshold: float = 0.5,
        estimator: Optional[DifficultyEstimator] = None,
    ):
        self.strategy = strategy
        self.default_mode = default_mode
        self.threshold = threshold
        self.estimator = estimator if estimator is not None else RuleBasedDifficulty()

    def route(self, prompts: List[str], **kwargs) -> List[str]:
        if self.strategy == "fixed":
            return [self.default_mode] * len(prompts)
        if self.strategy == "auto":
            hard = self.estimator.is_hard(prompts, threshold=self.threshold, **kwargs)
            return ["thinking" if h else "chat" for h in hard]
        if self.strategy == "learned":
            # 未来：调用小型分类模型
            raise NotImplementedError("learned routing not implemented")
        raise ValueError(self.strategy)
