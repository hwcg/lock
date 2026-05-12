#!/usr/bin/env python
"""
CISPO 训练入口。

用法：
    python scripts/cispo.py --config configs/training/cispo.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from safetensors.torch import load_file

from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.training.cispo import CISPOConfig, CISPOTrainer
from deepseek_v4.training.rewards import build_reward_from_config
from deepseek_v4.utils.config import load_config_with_overrides
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("cispo")


def _load_sd(p: Path):
    if p.is_dir():
        for q in [p / "model.safetensors", p / "pytorch_model.bin"]:
            if q.exists():
                return load_file(str(q)) if q.suffix == ".safetensors" else torch.load(str(q), map_location="cpu")
        raise FileNotFoundError(p)
    return load_file(str(p)) if str(p).endswith(".safetensors") else torch.load(str(p), map_location="cpu")


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini CISPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config_with_overrides(CISPOConfig, args.config, args.overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.output_dir) / "trainer_config.yaml")
    setup_logging(level="INFO")

    tokenizer = DeepseekV4Tokenizer.from_pretrained(cfg.tokenizer_path)
    with open(cfg.model_config_path, "r", encoding="utf-8") as f:
        model_cfg = DeepseekV4Config.from_dict(json.load(f))
    model_cfg.pad_token_id = tokenizer.pad_token_id

    policy = DeepseekV4ForCausalLM(model_cfg)
    if cfg.init_from_checkpoint:
        policy.load_state_dict(_load_sd(Path(cfg.init_from_checkpoint)), strict=False)

    rewards = [build_reward_from_config(r) for r in cfg.rewards] if cfg.rewards else []

    trainer = CISPOTrainer(config=cfg, policy=policy, tokenizer=tokenizer, reward_fns=rewards)
    trainer.train()


if __name__ == "__main__":
    main()
