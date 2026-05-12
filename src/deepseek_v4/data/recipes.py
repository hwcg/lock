"""
数据 recipe：每个阶段（pretrain / sft / dpo / ...）的标准数据组合。

每条 recipe 是一个 list[dict]，dict 字段：
    source:      "huggingface" | "modelscope" | "url" | "local"
    name:        数据集名
    config:      子集（可选）
    split:       默认 "train"
    weight:      混合权重
    sample_size: 采样上限
    field_map:   字段映射（用于统一不同数据集的列名）
    notes:       说明
"""
from __future__ import annotations

from typing import Any, Dict, List


# ============================================================
# Pretrain 阶段
# ============================================================
PRETRAIN_RECIPE: List[Dict[str, Any]] = [
    # 中文
    {
        "source": "huggingface",
        "name": "wikipedia",
        "config": "20231101.zh",
        "split": "train",
        "weight": 1.0,
        "sample_size": 1_000_000,
        "field_map": {"text": "text"},
        "notes": "中文维基百科",
    },
    {
        "source": "huggingface",
        "name": "BelleGroup/train_3.5M_CN",
        "config": None,
        "split": "train",
        "weight": 0.8,
        "sample_size": 500_000,
        "field_map": {"text": ["instruction", "input", "output"]},
        "notes": "中文通用指令（拼接 instruction+input+output 用作 PT）",
    },
    {
        "source": "huggingface",
        "name": "ranWang/zh-pretrain",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 500_000,
        "field_map": {"text": "text"},
        "notes": "中文预训练通用语料",
    },
    # 英文
    {
        "source": "huggingface",
        "name": "wikipedia",
        "config": "20231101.en",
        "split": "train",
        "weight": 1.0,
        "sample_size": 1_000_000,
        "field_map": {"text": "text"},
        "notes": "英文维基百科",
    },
    {
        "source": "huggingface",
        "name": "openwebtext",
        "config": None,
        "split": "train",
        "weight": 0.8,
        "sample_size": 500_000,
        "field_map": {"text": "text"},
        "notes": "英文网页文本",
    },
    # 代码
    {
        "source": "huggingface",
        "name": "code_search_net",
        "config": "python",
        "split": "train",
        "weight": 0.5,
        "sample_size": 200_000,
        "field_map": {"text": "whole_func_string"},
        "notes": "Python 代码",
    },
    {
        "source": "huggingface",
        "name": "bigcode/the-stack-smol",
        "config": "data/python",
        "split": "train",
        "weight": 0.4,
        "sample_size": 100_000,
        "field_map": {"text": "content"},
        "notes": "小型 The Stack（多语言代码）",
    },
    # 数学 / 推理
    {
        "source": "huggingface",
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "weight": 0.3,
        "sample_size": None,
        "field_map": {"text": ["question", "answer"]},
        "notes": "GSM8K 小学数学",
    },
]


# ============================================================
# SFT 阶段
# ============================================================
SFT_RECIPE: List[Dict[str, Any]] = [
    {
        "source": "huggingface",
        "name": "shibing624/alpaca-zh",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 50_000,
        "field_map": {
            "messages": [
                {"role": "user", "content": "$instruction\n\n$input"},
                {"role": "assistant", "content": "$output"},
            ],
        },
        "notes": "中文 Alpaca",
    },
    {
        "source": "huggingface",
        "name": "yahma/alpaca-cleaned",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 50_000,
        "field_map": {
            "messages": [
                {"role": "user", "content": "$instruction\n\n$input"},
                {"role": "assistant", "content": "$output"},
            ],
        },
        "notes": "英文 Alpaca cleaned",
    },
    {
        "source": "huggingface",
        "name": "BelleGroup/multiturn_chat_0.8M",
        "config": None,
        "split": "train",
        "weight": 0.8,
        "sample_size": 100_000,
        "field_map": {"messages": "conversations"},
        "notes": "Belle 多轮对话",
    },
    {
        "source": "huggingface",
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "weight": 0.6,
        "sample_size": None,
        "field_map": {
            "messages": [
                {"role": "user", "content": "$question"},
                {"role": "assistant", "content": "$answer"},
            ],
        },
        "notes": "GSM8K 转 SFT",
    },
    {
        "source": "huggingface",
        "name": "TIGER-Lab/MathInstruct",
        "config": None,
        "split": "train",
        "weight": 0.4,
        "sample_size": 100_000,
        "field_map": {
            "messages": [
                {"role": "user", "content": "$instruction"},
                {"role": "assistant", "content": "$output"},
            ],
        },
        "notes": "数学指令",
    },
]


# ============================================================
# DPO 阶段
# ============================================================
DPO_RECIPE: List[Dict[str, Any]] = [
    {
        "source": "huggingface",
        "name": "Anthropic/hh-rlhf",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 50_000,
        "field_map": {"chosen": "chosen", "rejected": "rejected"},
        "notes": "Anthropic 偏好数据",
    },
    {
        "source": "huggingface",
        "name": "argilla/distilabel-intel-orca-dpo-pairs",
        "config": None,
        "split": "train",
        "weight": 0.8,
        "sample_size": 10_000,
        "field_map": {
            "prompt": "input",
            "chosen": "chosen",
            "rejected": "rejected",
        },
        "notes": "DistilOrca DPO",
    },
]


# ============================================================
# Tool Use
# ============================================================
TOOL_RECIPE: List[Dict[str, Any]] = [
    {
        "source": "huggingface",
        "name": "Salesforce/xlam-function-calling-60k",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 20_000,
        "field_map": {
            "messages": "query",
            "tools": "tools",
            "tool_calls": "answers",
        },
        "notes": "xLAM function-calling 数据",
    },
    {
        "source": "huggingface",
        "name": "glaiveai/glaive-function-calling-v2",
        "config": None,
        "split": "train",
        "weight": 0.8,
        "sample_size": 20_000,
        "field_map": {"raw": "chat"},
        "notes": "Glaive function-calling",
    },
]


# ============================================================
# 推理 / 思考
# ============================================================
THINKING_RECIPE: List[Dict[str, Any]] = [
    {
        "source": "huggingface",
        "name": "open-thoughts/OpenThoughts-114k",
        "config": None,
        "split": "train",
        "weight": 1.0,
        "sample_size": 30_000,
        "field_map": {
            "messages": "conversations",
            "thinking": True,
        },
        "notes": "OpenThoughts 推理数据",
    },
    {
        "source": "huggingface",
        "name": "PrimeIntellect/SYNTHETIC-1",
        "config": None,
        "split": "train",
        "weight": 0.5,
        "sample_size": 20_000,
        "field_map": {"messages": "conversations", "thinking": True},
        "notes": "SYNTHETIC-1 推理数据",
    },
]


DATA_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "pretrain": PRETRAIN_RECIPE,
    "sft":      SFT_RECIPE,
    "dpo":      DPO_RECIPE,
    "tool":     TOOL_RECIPE,
    "thinking": THINKING_RECIPE,
}


def get_recipe(stage: str) -> List[Dict[str, Any]]:
    if stage not in DATA_RECIPES:
        raise KeyError(f"Unknown stage: {stage}. Available: {list(DATA_RECIPES.keys())}")
    return DATA_RECIPES[stage]
