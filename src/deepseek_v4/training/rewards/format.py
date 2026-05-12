"""
形式（非语义）类 reward：长度、重复、格式正确。

所有 builder 都返回 RewardFunction，便于与 CompositeReward 组合。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from deepseek_v4.training.rewards.base import NamedReward, RewardFunction


# ============================================================
# 长度
# ============================================================

def length_reward(
    target_len: int = 256,
    tolerance: int = 64,
    min_reward: float = 0.0,
    max_reward: float = 1.0,
) -> RewardFunction:
    """
    距离 target_len 越近 reward 越高（三角形）。
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            l = len(c.split())
            d = abs(l - target_len)
            if d <= tolerance:
                # 三角形从 max 到 min
                r = max_reward - (max_reward - min_reward) * (d / max(tolerance, 1))
            else:
                r = min_reward
            out.append(float(r))
        return out

    return NamedReward(fn, name="length")


# ============================================================
# 重复
# ============================================================

def repetition_penalty_reward(
    ngram: int = 4,
    max_repeat_ratio: float = 0.2,
    penalty: float = -1.0,
) -> RewardFunction:
    """
    检测 n-gram 重复，超过阈值给负奖励。

    重复率 = 重复 n-gram 数 / 总 n-gram 数
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            tokens = c.split()
            if len(tokens) < ngram + 1:
                out.append(0.0)
                continue
            grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
            cnt = Counter(grams)
            dup = sum(v - 1 for v in cnt.values() if v > 1)
            ratio = dup / len(grams)
            r = penalty if ratio > max_repeat_ratio else 0.0
            out.append(float(r))
        return out

    return NamedReward(fn, name="repetition")


# ============================================================
# 一般格式（任意正则）
# ============================================================

def regex_reward(
    pattern: str,
    flags: int = re.DOTALL,
    match_reward: float = 1.0,
    miss_reward: float = 0.0,
) -> RewardFunction:
    """匹配 pattern 给 match_reward，否则 miss_reward。"""
    rx: Pattern[str] = re.compile(pattern, flags)
    def fn(completions, references=None, prompts=None, **kwargs):
        return [match_reward if rx.search(c) else miss_reward for c in completions]
    fn.__name__ = f"regex({pattern[:20]}...)"
    return NamedReward(fn, name="regex")


# ============================================================
# 通用 "包含若干必需 token" 格式
# ============================================================

def format_reward(
    required_substrings: List[str],
    forbidden_substrings: Optional[List[str]] = None,
    match_reward: float = 1.0,
    miss_reward: float = 0.0,
) -> RewardFunction:
    """
    要求 completion 同时包含所有 required，不包含任何 forbidden。
    """
    forbidden_substrings = forbidden_substrings or []
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            ok = all(s in c for s in required_substrings) and \
                 not any(s in c for s in forbidden_substrings)
            out.append(match_reward if ok else miss_reward)
        return out
    return NamedReward(fn, name="format")


# ============================================================
# 思考格式 <think>...</think>
# ============================================================

_THINK_PATTERN = re.compile(
    r"^<think>(.*?)</think>(.*)$", re.DOTALL,
)


def thinking_format_reward(
    require_think: bool = True,
    min_think_len: int = 10,
    max_think_ratio: float = 0.95,
    match_reward: float = 1.0,
    partial_reward: float = 0.3,
    miss_reward: float = -0.5,
) -> RewardFunction:
    """
    严格的思考格式：
    - completion 必须以 <think> 开头
    - 必须含 </think>
    - </think> 之后还要有正文
    - think 内容不能太短，也不能占总长 95%+（异常情况）
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            c_stripped = c.strip()
            m = _THINK_PATTERN.match(c_stripped)
            if not m:
                # 没有正确的 think block
                if require_think:
                    out.append(miss_reward)
                else:
                    out.append(0.0)
                continue
            think = m.group(1).strip()
            rest = m.group(2).strip()
            if len(think) < min_think_len:
                out.append(partial_reward)
                continue
            if not rest:
                # 全是思考没有最终回答
                out.append(partial_reward)
                continue
            ratio = len(think) / max(len(c_stripped), 1)
            if ratio > max_think_ratio:
                out.append(partial_reward)
                continue
            out.append(match_reward)
        return out
    return NamedReward(fn, name="thinking_format")


# ============================================================
# DSML 工具调用格式
# ============================================================

_TC_OPEN = "<｜DSML｜tool_calls>"
_TC_CLOSE = "</｜DSML｜tool_calls>"
_INVOKE_OPEN = "<｜DSML｜invoke"
_PARAM_OPEN = "<｜DSML｜parameter"


def tool_call_format_reward(
    require_tool_call: bool = False,
    well_formed_reward: float = 1.0,
    malformed_reward: float = -1.0,
    no_call_reward: float = 0.0,
) -> RewardFunction:
    """
    检查 tool_call 块是否良好闭合。

    良好定义：
        <DSML|tool_calls> ... <DSML|invoke ...> ... <DSML|parameter ...>VAL</...parameter> ... </...invoke> ... </tool_calls>
    """
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            has_open = _TC_OPEN in c
            if not has_open:
                out.append(malformed_reward if require_tool_call else no_call_reward)
                continue
            # 检查闭合
            open_count = c.count(_TC_OPEN)
            close_count = c.count(_TC_CLOSE)
            if open_count != close_count:
                out.append(malformed_reward)
                continue
            invoke_open = c.count(_INVOKE_OPEN)
            invoke_close = c.count("</｜DSML｜invoke>")
            param_open = c.count(_PARAM_OPEN)
            param_close = c.count("</｜DSML｜parameter>")
            if invoke_open != invoke_close or param_open != param_close:
                out.append(malformed_reward)
                continue
            out.append(well_formed_reward)
        return out
    return NamedReward(fn, name="tool_call_format")


# ============================================================
# JSON 格式
# ============================================================

def json_format_reward(
    strict: bool = True,
    must_have_keys: Optional[List[str]] = None,
    valid_reward: float = 1.0,
    invalid_reward: float = -0.5,
) -> RewardFunction:
    """
    检查 completion 是否为合法 JSON。

    strict=True 时：completion 必须能整体被 json.loads 解析
    strict=False 时：抽出第一个 {...} / [...] 子串解析
    """
    must = must_have_keys or []
    def fn(completions, references=None, prompts=None, **kwargs):
        out = []
        for c in completions:
            text = c.strip()
            try:
                if strict:
                    obj = json.loads(text)
                else:
                    # 抽第一个 JSON 子串
                    m = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
                    if not m:
                        out.append(invalid_reward)
                        continue
                    obj = json.loads(m.group(1))
                if must and isinstance(obj, dict):
                    if not all(k in obj for k in must):
                        out.append(invalid_reward / 2)
                        continue
                out.append(valid_reward)
            except Exception:
                out.append(invalid_reward)
        return out
    return NamedReward(fn, name="json_format")
