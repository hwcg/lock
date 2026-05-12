#!/usr/bin/env python
"""
Needle-in-a-Haystack 长文本压力测试入口。

例：
    python scripts/run_needle.py \
        --model_path checkpoints/sft/checkpoint-final \
        --tokenizer_path checkpoints/tokenizer \
        --yarn_factor 16 \
        --context_lengths 4000,16000,65000,262000 \
        --output_dir needle_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from deepseek_v4.evaluation.engine import LocalEngine, OpenAIEngine, VLLMEngine
from deepseek_v4.inference.yarn import (
    NeedleConfig, YarnConfig, apply_yarn_to_model, run_needle_test,
)
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.utils.logger import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="local", choices=["local", "openai", "vllm"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=300_000)
    parser.add_argument("--openai_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--openai_api_key", default="EMPTY")
    parser.add_argument("--openai_model", default="deepseek-v4-mini")
    parser.add_argument("--yarn_factor", type=float, default=16.0, help="YaRN 扩展因子")
    parser.add_argument("--context_lengths", default="4000,16000,65000,262000",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--needle", default="The secret treasure is hidden under the old oak tree in the middle of the forest near the abandoned castle.")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--output_dir", default="needle_results")
    args = parser.parse_args()

    setup_logging(level="INFO")

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    needle_cfg = NeedleConfig(
        needle=args.needle,
        context_lengths=context_lengths,
        num_trials=args.num_trials,
        output_dir=str(out),
    )

    if args.backend == "local":
        from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
        tokenizer = DeepseekV4Tokenizer.from_pretrained(args.tokenizer_path)
        # Load model
        import json
        ckpt_dir = Path(args.model_path)
        config_file = ckpt_dir.parent / "config.json" if ckpt_dir.is_dir() else None
        if config_file and config_file.exists():
            with open(config_file, "r") as f:
                model_cfg = DeepseekV4Config.from_dict(json.load(f))
        else:
            model_cfg = DeepseekV4Config()
        model = DeepseekV4ForCausalLM(model_cfg)
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_dir / "model.safetensors")) if ckpt_dir.is_dir() else load_file(str(args.model_path))
        model.load_state_dict(sd, strict=False)

        # Apply YaRN scaling
        yarn_cfg = YarnConfig(factor=args.yarn_factor, max_seq_len=args.max_seq_len)
        model = apply_yarn_to_model(model, yarn_cfg)

        engine = LocalEngine(model=model, tokenizer=tokenizer, device=args.device, dtype=args.dtype)
        run_needle_test(engine, tokenizer, needle_cfg)
    elif args.backend == "openai":
        engine = OpenAIEngine(base_url=args.openai_base_url, api_key=args.openai_api_key, model=args.openai_model)
        tokenizer = None
        run_needle_test(engine, tokenizer, needle_cfg)
    elif args.backend == "vllm":
        engine = VLLMEngine(model_path=args.model_path, max_seq_len=args.max_seq_len)
        tokenizer = None
        run_needle_test(engine, tokenizer, needle_cfg)


if __name__ == "__main__":
    main()
