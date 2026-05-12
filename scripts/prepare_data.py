#!/usr/bin/env python
"""
完整数据准备流程：

    # 准备 SFT 数据
    python scripts/prepare_data.py \
        --stage sft \
        --output_dir data/processed \
        --dedup_method minhash

流程：
    1. 按 recipe 下载多源数据到 data/raw/
    2. 各文件独立清洗 → data/cleaned/
    3. 跨文件全局去重 → data/processed/{stage}.jsonl
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deepseek_v4.data.cleaning import (
    LanguageFilter, QualityFilter, SafetyFilter, TextCleaner, clean_pipeline,
)
from deepseek_v4.data.dedup import dedup_pipeline
from deepseek_v4.data.download import DownloadConfig, download_dataset
from deepseek_v4.data.recipes import DATA_RECIPES
from deepseek_v4.utils.logger import get_logger, setup_logging

logger = get_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser("Prepare data pipeline")
    parser.add_argument("--stage", required=True, choices=list(DATA_RECIPES.keys()),
                        help="数据准备阶段")
    parser.add_argument("--output_dir", default="data", help="数据根目录")
    parser.add_argument("--cache_dir", default="cache/hf_datasets",
                        help="HuggingFace 缓存目录")
    parser.add_argument("--dedup_method", default="minhash",
                        choices=["exact", "minhash", "simhash"])
    parser.add_argument("--dedup_threshold", type=float, default=0.85)
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--skip_dedup", action="store_true")
    parser.add_argument("--allowed_languages", nargs="+", default=["zh", "en"])
    args = parser.parse_args()

    setup_logging(level="INFO")

    root = Path(args.output_dir)
    raw_dir = root / "raw"
    cleaned_dir = root / "cleaned"
    processed_dir = root / "processed"
    for d in (raw_dir, cleaned_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ----- 1. Download -----
    recipe = DATA_RECIPES[args.stage]
    logger.info(f"[Prepare] stage={args.stage}, recipe size={len(recipe)}")
    raw_paths = []
    for entry in recipe:
        cfg = DownloadConfig(
            source=entry["source"],
            name=entry["name"],
            config=entry.get("config"),
            split=entry.get("split", "train"),
            sample_size=entry.get("sample_size"),
            field_map=entry.get("field_map", {}),
            cache_dir=args.cache_dir,
        )
        try:
            p = download_dataset(cfg, output_dir=str(raw_dir))
            raw_paths.append(p)
        except Exception as e:
            logger.warning(f"download failed: {cfg.name} - {e}")

    # ----- 2. Clean -----
    text_field = "text" if args.stage == "pretrain" else "messages"
    if args.skip_clean or args.stage != "pretrain":
        # 非 pretrain 阶段（messages 类型）跳过文本清洗
        cleaned_paths = raw_paths
    else:
        cleaner = TextCleaner(strip_urls=False, max_consecutive_newlines=2)
        quality = QualityFilter()
        lang = LanguageFilter(allowed_languages=args.allowed_languages)
        safety = SafetyFilter()
        cleaned_paths = []
        for p in raw_paths:
            cleaned_p = cleaned_dir / Path(p).name
            clean_pipeline(
                input_path=p, output_path=cleaned_p,
                text_field=text_field,
                cleaner=cleaner, quality_filter=quality,
                language_filter=lang, safety_filter=safety,
            )
            cleaned_paths.append(str(cleaned_p))

    # ----- 3. Dedup -----
    out_path = processed_dir / f"{args.stage}.jsonl"
    if args.skip_dedup:
        # 直接合并
        from deepseek_v4.utils.io import write_jsonl, read_jsonl
        def _iter():
            for p in cleaned_paths:
                yield from read_jsonl(p)
        write_jsonl(out_path, _iter())
    else:
        dedup_pipeline(
            input_paths=cleaned_paths,
            output_path=out_path,
            text_field=text_field,
            method=args.dedup_method,
            threshold=args.dedup_threshold,
        )

    logger.info(f"[Prepare] done: {out_path}")


if __name__ == "__main__":
    main()
