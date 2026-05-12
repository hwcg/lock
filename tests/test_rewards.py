"""Reward 函数单测。"""
import json
import pytest

from deepseek_v4.training.rewards.base import (
    CompositeReward, ConstantReward, NamedReward, RewardFunction,
    build_reward_from_config,
)
from deepseek_v4.training.rewards.format import (
    format_reward, json_format_reward, length_reward, regex_reward,
    repetition_penalty_reward, thinking_format_reward, tool_call_format_reward,
)
from deepseek_v4.training.rewards.math_ import (
    boxed_reward, extract_boxed, extract_last_number, gsm8k_answer_reward,
    is_equiv, math_correctness_reward, normalize_answer,
)
from deepseek_v4.training.rewards.code import code_python_reward, code_execute_reward


# ============================================================
# Base
# ============================================================

def test_constant_reward():
    r = ConstantReward(value=0.5)
    scores = r(completions=["a", "b", "c"])
    assert scores == [0.5, 0.5, 0.5]


def test_named_reward():
    def fn(completions, **kwargs):
        return [1.0] * len(completions)
    r = NamedReward(fn, name="test")
    scores = r(completions=["x", "y"])
    assert scores == [1.0, 1.0]


def test_composite_reward():
    r1 = ConstantReward(value=0.5, name="a")
    r2 = ConstantReward(value=1.0, name="b")
    comp = CompositeReward(rewards=[r1, r2], weights=[1.0, 2.0])
    scores = comp(completions=["x", "y", "z"])
    # 0.5*1 + 1.0*2 = 2.5 per sample
    assert scores == pytest.approx([2.5, 2.5, 2.5])


def test_composite_reward_detail():
    r1 = ConstantReward(value=0.5, name="a")
    comp = CompositeReward(rewards=[r1], weights=[1.0])
    detail = comp(completions=["x"], return_detail=True)
    assert "_total" in detail
    assert "a" in detail
    assert detail["_total"] == [0.5]


def test_build_reward_from_config():
    config = [
        {"name": "constant", "weight": 1.0, "params": {"value": 2.0}},
    ]
    comp = build_reward_from_config(config)
    scores = comp(completions=["a", "b"])
    assert scores == [2.0, 2.0]


# ============================================================
# Format Rewards
# ============================================================

def test_length_reward():
    r = length_reward(target_len=5, tolerance=3)
    # "one two three four five" = 5 words → max reward
    s1 = r(completions=["one two three four five"])
    assert s1[0] >= 0.9

    # 1 word → min reward
    s2 = r(completions=["hello"])
    assert s2[0] < 0.5


def test_repetition_penalty_reward():
    r = repetition_penalty_reward(ngram=3, max_repeat_ratio=0.2, penalty=-1.0)
    # no repetition
    s1 = r(completions=["a b c d e f g h i j"])
    assert s1[0] == 0.0

    # heavy repetition
    s2 = r(completions=["a b c " * 20])
    assert s2[0] == -1.0


def test_regex_reward():
    r = regex_reward(pattern=r"\d+")
    s1 = r(completions=["the answer is 42"])
    assert s1[0] == 1.0
    s2 = r(completions=["no number here"])
    assert s2[0] == 0.0


def test_format_reward():
    r = format_reward(required_substrings=["answer"], forbidden_substrings=["???"])
    s1 = r(completions=["the answer is 42"])
    assert s1[0] == 1.0
    s2 = r(completions=["what??? hello"])
    assert s2[0] == 0.0


def test_thinking_format_reward():
    r = thinking_format_reward(require_think=True)
    # valid think
    s1 = r(completions=["<think>hmm let me think...</think>final answer 42"])
    assert s1[0] > 0

    # no think block
    s2 = r(completions=["just answer 42"])
    assert s2[0] < 0


def test_tool_call_format_reward():
    r = tool_call_format_reward(require_tool_call=False)
    # well-formed tool call
    s1 = r(completions=[
        "Sure.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    ])
    assert s1[0] == 1.0

    # no tool call
    s2 = r(completions=["plain text"])
    assert s2[0] == 0.0


def test_json_format_reward():
    r = json_format_reward(strict=True)
    s1 = r(completions=['{"key": "value"}'])
    assert s1[0] == 1.0
    s2 = r(completions=["not json"])
    assert s2[0] == -0.5


# ============================================================
# Math Rewards
# ============================================================

def test_extract_boxed():
    assert extract_boxed(r"the answer is \boxed{42}") == "42"
    assert extract_boxed(r"no box") is None
    # nested braces
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_last_number():
    assert extract_last_number("#### 42\n##") == "42"
    # no #### marker
    assert extract_last_number("the answer is 99.") == "99"


def test_normalize_answer():
    assert normalize_answer(" 42 ") == "42"
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("$ 42 $") == "42"
    # fraction
    result = normalize_answer("1/2")
    assert result in ("1/2", "0.5")


def test_is_equiv():
    assert is_equiv("42", "42") is True
    assert is_equiv("42", "42.0") is True
    assert is_equiv("1/2", "0.5") is True
    assert is_equiv("42", "99") is False


def test_boxed_reward():
    r = boxed_reward(match_reward=1.0, miss_reward=0.0)
    s1 = r(completions=[r"\boxed{42}"])
    assert s1[0] == 1.0
    s2 = r(completions=["no box"])
    assert s2[0] == 0.0


def test_math_correctness_reward():
    r = math_correctness_reward(extractor="boxed")
    scores = r(
        completions=[r"\boxed{42}"],
        references=["42"],
    )
    assert scores[0] == 1.0

    scores2 = r(
        completions=[r"\boxed{99}"],
        references=["42"],
    )
    assert scores2[0] == 0.0


def test_math_correctness_no_answer():
    r = math_correctness_reward(no_answer_reward=-0.5)
    scores = r(
        completions=["no answer found here"],
        references=["42"],
    )
    assert scores[0] == -0.5


def test_gsm8k_answer_reward():
    r = gsm8k_answer_reward()
    scores = r(
        completions=["#### 42"],
        references=["42"],
    )
    assert scores[0] == 1.0


# ============================================================
# Code Rewards
# ============================================================

def test_code_execute_reward():
    r = code_execute_reward()
    scores = r(
        completions=["The answer is ```python\nprint(1+1)\n```"],
        references=["2\n"],
    )
    assert len(scores) == 1


def test_code_python_reward_basic():
    r = code_python_reward()
    scores = r(completions=["```python\nprint(42)\n```"])
    assert len(scores) == 1
