"""
难度估计器：决定每个 prompt 应该走 thinking 还是 chat 模式。

策略：
- RuleBasedDifficulty：基于 prompt 关键词（数学/代码/复杂推理）
- GroupVarianceDifficulty：在 RL 阶段，根据组内 reward 方差判定（高方差 = 难）
- LearnedDifficulty：未来扩展，由小模型预测难度（此处仅占位）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class DifficultyEstimator:
    """难度估计器协议。"""
    def estimate(self, prompts: List[str], **kwargs) -> List[float]:
        """返回每个 prompt 的难度评分 ∈ [0, 1]。"""
        raise NotImplementedError

    def is_hard(self, prompts: List[str], threshold: float = 0.5, **kwargs) -> List[bool]:
        return [s >= threshold for s in self.estimate(prompts, **kwargs)]


# ============================================================
# Rule-based
# ============================================================

@dataclass
class RuleBasedDifficulty(DifficultyEstimator):
    """
    基于关键词与长度的规则难度判定。

    简单启发式：
    - 含数学符号、code 关键字 / 多步推理标志 → 高难度
    - 长 prompt 通常更难
    """
    math_keywords: Tuple[str, ...] = (
        "compute", "solve", "calculate", "prove",
        "求解", "计算", "证明", "推导",
        "integral", "derivative", "积分", "导数",
    )
    code_keywords: Tuple[str, ...] = (
        "function", "implement", "algorithm", "complexity",
        "实现", "算法", "调试", "优化",
    )
    reasoning_keywords: Tuple[str, ...] = (
        "explain why", "step by step", "reasoning", "为什么", "推理",
        "分析", "比较", "为何",
    )
    has_math_symbols: bool = True
    long_prompt_threshold: int = 200

    def _score_one(self, prompt: str) -> float:
        text = prompt.lower()
        score = 0.0
        if any(k in text for k in self.math_keywords):
            score += 0.4
        if any(k in text for k in self.code_keywords):
            score += 0.3
        if any(k in text for k in self.reasoning_keywords):
            score += 0.3
        if self.has_math_symbols and re.search(r"[=+\-*/^∫∑√≤≥≠∈∀∃]|\\frac|\\sqrt", text):
            score += 0.3
        # 数字密集
        n_digits = sum(c.isdigit() for c in text)
        if n_digits / max(len(text), 1) > 0.05:
            score += 0.1
        # 长度
        if len(prompt) > self.long_prompt_threshold:
            score += 0.1
        return min(score, 1.0)

    def estimate(self, prompts: List[str], **kwargs) -> List[float]:
        return [self._score_one(p) for p in prompts]


# ============================================================
# Group-variance based（RL 阶段用）
# ============================================================

@dataclass
class GroupVarianceDifficulty(DifficultyEstimator):
    """
    根据 group 内 reward 方差判定。

    使用场景：在 GRPO 风格 RL 中，每个 prompt 采样 G 次，
    若不同采样的 reward 方差大 → 模型对该题没把握 → 标记为难。

    输入：rewards [N]，group_ids [N]
    输出：每个 group 的难度 ∈ [0, 1]（用 std 归一化）
    """
    eps: float = 1e-6

    def estimate(self, prompts: List[str], **kwargs) -> List[float]:
        rewards = kwargs.get("rewards")
        group_ids = kwargs.get("group_ids")
        if rewards is None or group_ids is None:
            raise ValueError("GroupVarianceDifficulty needs rewards & group_ids")

        rewards = [float(r) for r in rewards]
        group_ids = [int(g) for g in group_ids]

        # 每个 group 的 std
        groups: Dict[int, List[float]] = {}
        for r, g in zip(rewards, group_ids):
            groups.setdefault(g, []).append(r)
        stds: Dict[int, float] = {}
        for g, rs in groups.items():
            if len(rs) <= 1:
                stds[g] = 0.0
            else:
                m = sum(rs) / len(rs)
                stds[g] = (sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) ** 0.5

        max_std = max(stds.values()) if stds else 1.0
        norm = max(max_std, self.eps)
        # 把 group-level 难度广播到 sample-level
        per_group = {g: s / norm for g, s in stds.items()}
        return [per_group[g] for g in group_ids]
