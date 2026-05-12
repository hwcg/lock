#!/usr/bin/env python
"""
LoRA-SFT 训练入口。

用法：
    python scripts/lora.py --config configs/training/lora.yaml
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
from deepseek_v4.training.lora_trainer import LoRATrainer, LoRATrainerConfig
from deepseek_v4.utils.config import load_config_with_overrides
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("lora")


def _load_pretrained(model, ckpt_path: str):
    p = Path(ckpt_path)
    if p.is_dir():
        st = p / "model.safetensors"
        bin_ = p / "pytorch_model.bin"
        if st.exists():
            from safetensors.torch import load_file
            sd = load_file(str(st))
        elif bin_.exists():
            sd = torch.load(str(bin_), map_location="cpu")
        else:
            raise FileNotFoundError(f"no model file in {p}")
    else:
        if str(p).endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(str(p))
        else:
            sd = torch.load(str(p), map_location="cpu")
    model.load_state_dict(sd, strict=False)


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini LoRA-SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config_with_overrides(LoRATrainerConfig, args.config, args.overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.output_dir) / "trainer_config.yaml")
    setup_logging(level="INFO")

    tokenizer = DeepseekV4Tokenizer.from_pretrained(cfg.tokenizer_path)
    with open(cfg.model_config_path, "r", encoding="utf-8") as f:
        model_cfg = DeepseekV4Config.from_dict(json.load(f))
    model_cfg.pad_token_id = tokenizer.pad_token_id

    model = DeepseekV4ForCausalLM(model_cfg)
    if cfg.init_from_checkpoint:
        logger.info(f"Loading weights from {cfg.init_from_checkpoint}")
        _load_pretrained(model, cfg.init_from_checkpoint)
    else:
        model.init_weights()

    trainer = LoRATrainer(config=cfg, model=model, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
