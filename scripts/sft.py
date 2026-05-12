#!/usr/bin/env python
"""
SFT 训练入口。

单卡：
    python scripts/sft.py --config configs/training/sft.yaml

多卡 DDP：
    bash scripts/launch.sh sft configs/training/sft.yaml

DeepSpeed：
    bash scripts/launch.sh sft configs/training/sft.yaml use_deepspeed=true
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
from deepseek_v4.training.sft import SFTConfig, SFTTrainer
from deepseek_v4.training.checkpoint import CheckpointManager
from deepseek_v4.utils.config import load_config_with_overrides
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("sft")


def _try_load_pretrained(model: DeepseekV4ForCausalLM, ckpt_path: str) -> None:
    """从 pretrain checkpoint 加载权重。"""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"init checkpoint not found: {p}")
    if p.is_dir():
        # 优先 safetensors
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

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"  loaded: missing={len(missing)} unexpected={len(unexpected)}")


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 Mini SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config_with_overrides(SFTConfig, args.config, args.overrides)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.to_yaml(Path(config.output_dir) / "trainer_config.yaml")
    setup_logging(level="INFO")

    # ----- Tokenizer -----
    logger.info(f"Loading tokenizer: {config.tokenizer_path}")
    tokenizer = DeepseekV4Tokenizer.from_pretrained(config.tokenizer_path)

    # ----- Model -----
    logger.info(f"Loading model config: {config.model_config_path}")
    with open(config.model_config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    model_cfg = DeepseekV4Config.from_dict(cfg_dict)
    model_cfg.pad_token_id = tokenizer.pad_token_id
    model_cfg.bos_token_id = tokenizer.bos_token_id
    model_cfg.eos_token_id = tokenizer.eos_token_id

    model = DeepseekV4ForCausalLM(model_cfg)
    if config.init_from_checkpoint:
        logger.info(f"Loading initial weights from {config.init_from_checkpoint}")
        _try_load_pretrained(model, config.init_from_checkpoint)
    else:
        logger.warning("No init_from_checkpoint; SFT 从随机初始化开始（不推荐）")
        model.init_weights()

    # ----- Train -----
    trainer = SFTTrainer(config=config, model=model, tokenizer=tokenizer)
    trainer.train()


if __name__ == "__main__":
    main()
