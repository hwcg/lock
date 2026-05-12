"""
DeepSeek-V4-Mini: 工业级 DeepSeek-V4 训练与推理框架。

子包：
    modeling     模型核心
    tokenizer    分词与对话编码
    data         数据流水线
    training     训练算法（pretrain / SFT / LoRA / DPO / PPO / GRPO / CISPO / 蒸馏 / Agentic RL）
    inference    推理与服务
    evaluation   评测
    distributed  分布式
    utils        工具
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "DeepSeek-V4 Mini Team"

# 暴露最常用的接口
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.tokenizer.special_tokens import SpecialTokens

__all__ = [
    "__version__",
    "DeepseekV4Tokenizer",
    "SpecialTokens",
]
