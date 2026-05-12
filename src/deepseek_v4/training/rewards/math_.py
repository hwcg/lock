"""
数学正确性 reward。

支持：
- \\boxed{...} 答案抽取（OBQA / MATH 风格）
- 最后一个数字抽取（GSM8K 风格 "#### 42"）
- 答案归一化（分数、负号、千分位逗号）
- 大小写不敏感、忽略尾随空白
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, List, Optional, Union

from deepseek_v4.training.rewards.base import NamedReward, RewardFunction


# ============================================================
# 抽取
# ============================================================

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_GSM8K_ANSWER_RE = re.compile(r"####\s*(-?\d[\d,\.\s]*)")
_FINAL_NUMBER_RE = re.compile(r"(-?\d[\d,\.]*)\s*\.?\s*$")


def extract_boxed(text: str) -> Optional[str]:
    """抽取 \\boxed{...} 中最后一个匹配。"""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_number(text: str) -> Optional[str]:
    """抽取文本最后一个数字（GSM8K 风格）。"""
    # 优先 #### 标记
    m = _GSM8K_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    # 否则取最后一个数字
    matches = re.findall(r"-?\d[\d,\.]*", text)
    if not matches:
        return None
    return matches[-1].strip()


# ============================================================
# 归一化
# ============================================================

def normalize_answer(s: str) -> str:
    """
    把答案归一化为可比较的字符串。

    步骤：
    - 去逗号、空格、$、引号
    - 把 \\frac{a}{b} → a/b
    - 把 a/b → 化简后的最简分数
    - 把 100% → 1 / 0.5 → 1/2 等不做（保留原表示）
    - 去尾随 .0
    """
    if s is None:
        return ""
    s = s.strip()
    # 去掉常见装饰
    s = re.sub(r"[\$,\"\']", "", s)
    s = re.sub(r"\\(?:left|right)", "", s)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\dfrac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = s.replace(" ", "")
    # 去末尾 .0
    if "." in s and "/" not in s:
        try:
            f = float(s)
            if f == int(f):
                s = str(int(f))
            else:
                # 规整：去多余 0
                s = ("%g" % f)
        except ValueError:
            pass
    # 化简 a/b
    if re.fullmatch(r"-?\d+/-?\d+", s):
        try:
            frac = Fraction(s)
            if frac.denominator == 1:
                s = str(frac.numerator)
            else:
                s = f"{frac.numerator}/{frac.denominator}"
        except Exception:
            pass
    return s.lower()


def is_equiv(pred: str, gold: str) -> bool:
    """
    判断 pred 与 gold 是否等价。

    宽松规则：
    - 归一化后字符串一致
    - 或浮点数差 < 1e-4
    """
    if pred is None or gold is None:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    try:
        # 分数 / 浮点
        fp = _to_float(p)
        fg = _to_float(g)
        if fp is not None and fg is not None and abs(fp - fg) < 1e-4:
            return True
    except Exception:
        pass
    return False


def _to_float(s: str) -> Optional[float]:
    """字符串 → 浮点数（支持分数）。"""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if "/" in s:
        try:
            return float(Fraction(s))
        except Exception:
            return None
    return None


# ============================================================
# Reward 工厂
# ============================================================

def boxed_reward(
    match_reward: float = 1.0,
    miss_reward: float = 0.0,
    require_boxed: bool = True,
) -> RewardFunction:
    """根据是否存在 \\boxed{...} 给奖励（不验证内容）。"""
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            ans = extract_boxed(c)
            out.append(match_reward if ans else miss_reward)
        return out
    return NamedReward(fn, name="boxed_present")


def math_correctness_reward(
    extractor: str = "auto",         # auto | boxed | last_number
    correct_reward: float = 1.0,
    wrong_reward: float = 0.0,
    no_answer_reward: float = -0.5,
) -> RewardFunction:
    """
    数学正确性 reward。

    需要 references = List[str]（标准答案）。

    extractor:
        auto:        先 boxed，失败回退 last_number
        boxed:       仅 \\boxed{}
        last_number: 仅最后一个数字
    """
    def _extract(text: str) -> Optional[str]:
        if extractor == "boxed":
            return extract_boxed(text)
        if extractor == "last_number":
            return extract_last_number(text)
        # auto
        x = extract_boxed(text)
        if x is not None:
            return x
        return extract_last_number(text)

    def fn(completions, references=None, prompts=None, **kwargs):
        if references is None:
            raise ValueError("math_correctness_reward requires references")
        assert len(completions) == len(references), "length mismatch"
        out = []
        for c, r in zip(completions, references):
            ans = _extract(c)
            if ans is None:
                out.append(no_answer_reward)
                continue
            gold = str(r) if not isinstance(r, str) else r
            # 如果 gold 自身也是 boxed/带 ####，做一次抽取
            gold_clean = extract_boxed(gold) or extract_last_number(gold) or gold
            out.append(correct_reward if is_equiv(ans, gold_clean) else wrong_reward)
        return out

    return NamedReward(fn, name="math_correctness")


def gsm8k_answer_reward(**kwargs) -> RewardFunction:
    """GSM8K 专用：last_number 抽取。"""
    return math_correctness_reward(extractor="last_number", **kwargs)
