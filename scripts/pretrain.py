#!/usr/bin/env python
"""
预训练入口。

单卡：
    python scripts/pretrain.py --config configs/training/pretrain.yaml

单机多卡（DDP）：
    torchrun --standalone --nproc_per_node 8 \
        scripts/pretrain.py --config configs/training/pretrain.yaml

DeepSpeed：
    torchrun --standalone --nproc_per_node 8 \
        scripts/pretrain.py --config configs/training/pretrain.yaml \
        use_deepspeed=true deepspeed_config=configs/deepspeed/zero2.json

支持命令行 override：
    python scripts/pretrain.py --config configs/training/pretrain.yaml \
        learning_rate=1e-4 max_steps=50000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.training.pretrain import PretrainConfig, PretrainTrainer
from deepseek_v4.utils.config import load_config_with_overrides, parse_overrides
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("pretrain")


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini Pretraining")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    # 加载配置
    config = load_config_with_overrides(PretrainConfig, args.config, args.overrides)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.to_yaml(Path(config.output_dir) / "trainer_config.yaml")

    # 仅在主进程提前 setup logging（BaseTrainer.setup 会再覆盖一次，但提前打印更友好）
    setup_logging(level="INFO")

    # ----- 1. 加载 tokenizer -----
    logger.info(f"Loading tokenizer from {config.tokenizer_path}")
    tokenizer = DeepseekV4Tokenizer.from_pretrained(config.tokenizer_path)
    logger.info(f"  vocab_size = {tokenizer.vocab_size}")

    # ----- 2. 构造模型 -----
    logger.info(f"Loading model config from {config.model_config_path}")
    with open(config.model_config_path, "r", encoding="utf-8") as f:
        model_cfg_dict = json.load(f)
    model_cfg = DeepseekV4Config.from_dict(model_cfg_dict)

    # 把 tokenizer 的 pad_token_id 注入 model config
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.bos_token_id = tokenizer.bos_token_id
    model_cfg.eos_token_id = tokenizer.eos_token_id

    logger.info("Building model...")
    model = DeepseekV4ForCausalLM(model_cfg)

    # 初始化或加载
    if config.init_from_checkpoint:
        logger.info(f"Loading initial weights from {config.init_from_checkpoint}")
        from safetensors.torch import load_file
        sd = load_file(config.init_from_checkpoint)
        model.load_state_dict(sd, strict=False)
    else:
        model.init_weights()

    # ----- 3. 训练 -----
    trainer = PretrainTrainer(config=config, model=model, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
