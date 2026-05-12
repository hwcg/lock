#!/usr/bin/env python
"""
Tokenizer 训练入口。

用法：
    python scripts/train_tokenizer.py \
        --config configs/tokenizer/train_config.yaml \
        --output_dir checkpoints/tokenizer
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse
import yaml

from deepseek_v4.tokenizer.bpe import BPETrainer, BPETrainerConfig
from deepseek_v4.tokenizer.special_tokens import ALL_SPECIAL_TOKENS
from deepseek_v4.tokenizer.encoding import DeepseekV4Tokenizer


def main():
    parser = argparse.ArgumentParser("DeepSeek-V4 BPE Tokenizer Training")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--text_file", default=None, help="Override text file from config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)

    cfg = BPETrainerConfig(
        vocab_size=raw["vocab_size"],
        min_frequency=raw.get("min_frequency", 2),
        num_workers=raw.get("num_workers", 8),
        special_tokens=ALL_SPECIAL_TOKENS,
    )

    text_files = args.text_file or raw.get("text_file", None)
    if text_files is None:
        text_files = raw["text_files"]
    if isinstance(text_files, str):
        text_files = [text_files]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = BPETrainer(cfg)
    trainer.train(text_files)
    tokenizer = trainer.build_tokenizer()

    tokenizer.save_pretrained(str(out))
    print(f"Tokenizer saved to {out}")


if __name__ == "__main__":
    main()
