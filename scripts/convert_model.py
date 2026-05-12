#!/usr/bin/env python
"""
统一格式转换 CLI。

用法：
    # 1. 训练目录 → HF 兼容
    python scripts/convert_model.py to-hf \
        --state_dict_dir checkpoints/sft/checkpoint-final \
        --model_config configs/model/mini_2b.json \
        --tokenizer_dir checkpoints/tokenizer \
        --output_dir exports/hf

    # 2. → vLLM
    python scripts/convert_model.py to-vllm \
        --state_dict_dir checkpoints/sft/checkpoint-final \
        --model_config configs/model/mini_2b.json \
        --tokenizer_dir checkpoints/tokenizer \
        --output_dir exports/vllm

    # 3. → GGUF（需 llama.cpp）
    python scripts/convert_model.py to-gguf \
        --state_dict_dir checkpoints/sft/checkpoint-final \
        --model_config configs/model/mini_2b.json \
        --tokenizer_dir checkpoints/tokenizer \
        --output_dir exports/gguf \
        --llama_cpp_dir /path/to/llama.cpp \
        --quantization q4_K_M

    # 4. → Ollama
    python scripts/convert_model.py to-ollama \
        --gguf_path exports/gguf/model-q4_K_M.gguf \
        --output_dir exports/ollama \
        --model_name deepseek-v4-mini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from deepseek_v4.inference.convert import (
    export_to_gguf, export_to_hf, export_to_ollama, export_to_vllm,
)
from deepseek_v4.utils.logger import setup_logging


def _dtype(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- to-hf ----
    p = sub.add_parser("to-hf")
    p.add_argument("--state_dict_dir", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_shard_size", default="5GB")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])

    # ---- to-vllm ----
    p = sub.add_parser("to-vllm")
    p.add_argument("--state_dict_dir", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_shard_size", default="5GB")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])

    # ---- to-gguf ----
    p = sub.add_parser("to-gguf")
    p.add_argument("--state_dict_dir", required=True)
    p.add_argument("--model_config", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--quantization", default="q4_K_M",
                   help="f32 | f16 | q8_0 | q5_K_M | q4_K_M | q4_0 ...")
    p.add_argument("--llama_cpp_dir", default=None)
    p.add_argument("--convert_script", default=None)
    p.add_argument("--keep_intermediate_hf", action="store_true")

    # ---- to-ollama ----
    p = sub.add_parser("to-ollama")
    p.add_argument("--gguf_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="deepseek-v4-mini")
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num_ctx", type=int, default=4096)
    p.add_argument("--no_register", action="store_true", help="不调用 ollama create")

    args = parser.parse_args()
    setup_logging(level="INFO")

    if args.cmd == "to-hf":
        export_to_hf(
            state_dict_dir=args.state_dict_dir,
            model_config_path=args.model_config,
            tokenizer_dir=args.tokenizer_dir,
            output_dir=args.output_dir,
            max_shard_size=args.max_shard_size,
            dtype=_dtype(args.dtype),
        )
    elif args.cmd == "to-vllm":
        export_to_vllm(
            state_dict_dir=args.state_dict_dir,
            model_config_path=args.model_config,
            tokenizer_dir=args.tokenizer_dir,
            output_dir=args.output_dir,
            max_shard_size=args.max_shard_size,
            dtype=_dtype(args.dtype),
        )
    elif args.cmd == "to-gguf":
        export_to_gguf(
            state_dict_dir=args.state_dict_dir,
            model_config_path=args.model_config,
            tokenizer_dir=args.tokenizer_dir,
            output_dir=args.output_dir,
            quantization=args.quantization,
            llama_cpp_dir=args.llama_cpp_dir,
            convert_script_path=args.convert_script,
            keep_intermediate_hf=args.keep_intermediate_hf,
        )
    elif args.cmd == "to-ollama":
        export_to_ollama(
            gguf_path=args.gguf_path,
            output_dir=args.output_dir,
            model_name=args.model_name,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_ctx=args.num_ctx,
            run_create=(not args.no_register),
        )


if __name__ == "__main__":
    main()
