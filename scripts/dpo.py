#!/usr/bin/env python
"""
DPO 训练入口。

用法：
    python scripts/dpo.py --config configs/training/dpo.yaml
    torchrun --nproc_per_node 8 scripts/dpo.py --config configs/training/dpo.yaml
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
from deepseek_v4.training.dpo import DPOConfig, DPOTrainer
from deepseek_v4.utils.config import load_config_with_overrides
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("dpo")


def _load_state_dict(p: Path):
    if p.is_dir():
        st = p / "model.safetensors"
        bin_ = p / "pytorch_model.bin"
        if st.exists():
            return load_file(str(st))
        if bin_.exists():
            return torch.load(str(bin_), map_location="cpu")
        raise FileNotFoundError(p)
    if str(p).endswith(".safetensors"):
        return load_file(str(p))
    return torch.load(str(p), map_location="cpu")


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini DPO")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config_with_overrides(DPOConfig, args.config, args.overrides)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.output_dir) / "trainer_config.yaml")
    setup_logging(level="INFO")

    tokenizer = DeepseekV4Tokenizer.from_pretrained(cfg.tokenizer_path)
    with open(cfg.model_config_path, "r", encoding="utf-8") as f:
        model_cfg = DeepseekV4Config.from_dict(json.load(f))
    model_cfg.pad_token_id = tokenizer.pad_token_id

    model = DeepseekV4ForCausalLM(model_cfg)
    if cfg.init_from_checkpoint:
        model.load_state_dict(_load_state_dict(Path(cfg.init_from_checkpoint)), strict=False)
    else:
        model.init_weights()

    # Load reference model if specified
    ref_model = None
    if cfg.reference_checkpoint:
        ref_model = DeepseekV4ForCausalLM(model_cfg)
        ref_model.load_state_dict(_load_state_dict(Path(cfg.reference_checkpoint)), strict=False)

    trainer = DPOTrainer(config=cfg, model=model, ref_model=ref_model, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
