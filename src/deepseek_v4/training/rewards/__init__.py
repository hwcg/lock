"""
可组合的 reward 函数库。

设计：
- 所有 reward 函数都接收：
    completions: List[str]    模型生成的文本
    references:  Optional[List[Any]]  对应的参考（如 GSM8K 标准答案、单元测试）
    **kwargs                   其它上下文
  并返回：List[float]，每个样本一个标量。

- 多个 reward 可通过 CompositeReward 组合，并附权重。
- 同步支持 reward model 与规则 reward 混合（在 RL trainer 中调用）。

内置：
- length_reward / repetition_penalty_reward / format_reward
- math_correctness_reward    （GSM8K 风格答案匹配）
- boxed_reward               （\\boxed{...} 抽取）
- regex_reward               （任意正则）
- code_python_reward         （沙盒执行）
- json_format_reward
- thinking_format_reward     （<think>...</think>...）
- tool_call_format_reward    （DSML 调用格式）
- rm_reward                  （封装 RewardModel）
"""
from deepseek_v4.training.rewards.base import (
    RewardFunction, CompositeReward, ConstantReward, NamedReward,
    build_reward_from_config,
)
from deepseek_v4.training.rewards.format import (
    length_reward, repetition_penalty_reward, format_reward,
    thinking_format_reward, tool_call_format_reward, json_format_reward,
    regex_reward,
)
from deepseek_v4.training.rewards.math_ import (
    math_correctness_reward, boxed_reward, gsm8k_answer_reward,
    extract_boxed, extract_last_number, normalize_answer,
)
from deepseek_v4.training.rewards.code import (
    code_python_reward, run_python_sandboxed, code_execute_reward,
)
from deepseek_v4.training.rewards.model_reward import RewardModelReward

__all__ = [
    "RewardFunction", "CompositeReward", "ConstantReward", "NamedReward",
    "build_reward_from_config",
    "length_reward", "repetition_penalty_reward", "format_reward",
    "thinking_format_reward", "tool_call_format_reward", "json_format_reward",
    "regex_reward",
    "math_correctness_reward", "boxed_reward", "gsm8k_answer_reward",
    "extract_boxed", "extract_last_number", "normalize_answer",
    "code_python_reward", "run_python_sandboxed", "code_execute_reward",
    "RewardModelReward",
]
