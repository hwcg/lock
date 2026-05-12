"""
数据下载器：

支持源：
- huggingface（datasets 库）
- modelscope（中国镜像加速）
- url（HTTP / HTTPS）
- local（本地文件）

输出统一为 .jsonl 文件，便于后续清洗 / 去重 / dataset 加载。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from tqdm import tqdm

from deepseek_v4.utils.io import write_jsonl, read_jsonl
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置
# ============================================================

@dataclass
class DownloadConfig:
    source: str                  # "huggingface" | "modelscope" | "url" | "local"
    name: str                    # 数据集名 / URL / 路径
    config: Optional[str] = None
    split: str = "train"
    sample_size: Optional[int] = None
    seed: int = 42
    field_map: Dict[str, Any] = field(default_factory=dict)
    output_path: Optional[str] = None  # 不指定则按 dataset 名生成
    streaming: bool = True
    cache_dir: Optional[str] = None

    @property
    def cache_key(self) -> str:
        """用于缓存目录命名。"""
        s = f"{self.source}/{self.name}/{self.config}/{self.split}/{self.sample_size}"
        return hashlib.md5(s.encode()).hexdigest()[:12]


# ============================================================
# Field Map 解析
# ============================================================

def _resolve_template(template: str, row: Dict[str, Any]) -> str:
    """
    解析 "$field1\n\n$field2" 模板。

    支持嵌套 dot 访问：$conversations.0.value
    """
    import re

    def replace(m):
        path = m.group(1).split(".")
        v: Any = row
        for p in path:
            if isinstance(v, list):
                try:
                    v = v[int(p)]
                except (ValueError, IndexError):
                    return ""
            elif isinstance(v, dict):
                v = v.get(p, "")
            else:
                return ""
        return str(v) if v is not None else ""

    return re.sub(r"\$(\w+(?:\.\w+)*)", replace, template)


def _apply_field_map(row: Dict[str, Any], field_map: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    应用 field_map 进行字段映射 / 转换。

    field_map 语法：
        {"text": "content"}                  → output["text"] = row["content"]
        {"text": ["a", "b"]}                 → output["text"] = "\n".join(row["a"], row["b"])
        {"text": "$instruction\n\n$input"}   → 模板插值
        {"messages": [{"role": "user", "content": "$q"}, ...]}  → 静态消息列表（带插值）
    """
    if not field_map:
        return dict(row)

    out: Dict[str, Any] = {}
    for target_key, source_spec in field_map.items():
        if isinstance(source_spec, str):
            if "$" in source_spec:
                out[target_key] = _resolve_template(source_spec, row)
            else:
                # 简单字段名
                if source_spec not in row:
                    return None
                out[target_key] = row[source_spec]
        elif isinstance(source_spec, list):
            # 列表：可能是字段拼接，也可能是消息模板
            if source_spec and isinstance(source_spec[0], dict):
                # 消息模板
                msgs = []
                for item in source_spec:
                    msg = {
                        k: (_resolve_template(v, row) if isinstance(v, str) else v)
                        for k, v in item.items()
                    }
                    msgs.append(msg)
                out[target_key] = msgs
            else:
                # 字符串列表 → 拼接对应字段
                parts = []
                for k in source_spec:
                    v = row.get(k, "")
                    if v:
                        parts.append(str(v))
                out[target_key] = "\n\n".join(parts)
        elif source_spec is True:
            # 标记位（如 thinking=True）
            out[target_key] = True
        else:
            out[target_key] = source_spec
    return out


# ============================================================
# Downloader 实现
# ============================================================

def _download_huggingface(cfg: DownloadConfig) -> Iterator[Dict[str, Any]]:
    """从 HuggingFace Datasets 流式下载。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先 pip install datasets")

    ds = load_dataset(
        cfg.name, cfg.config,
        split=cfg.split,
        streaming=cfg.streaming,
        cache_dir=cfg.cache_dir,
    )
    if not cfg.streaming and cfg.sample_size:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.sample_size, len(ds))))

    count = 0
    for row in ds:
        yield row
        count += 1
        if cfg.sample_size and count >= cfg.sample_size:
            break


def _download_modelscope(cfg: DownloadConfig) -> Iterator[Dict[str, Any]]:
    """从 ModelScope 流式下载（用于中国大陆加速）。"""
    try:
        from modelscope.msdatasets import MsDataset
    except ImportError:
        raise ImportError("请先 pip install modelscope")

    ds = MsDataset.load(cfg.name, subset_name=cfg.config, split=cfg.split, cache_dir=cfg.cache_dir)
    count = 0
    for row in ds:
        if isinstance(row, dict):
            yield row
        else:
            yield {"text": str(row)}
        count += 1
        if cfg.sample_size and count >= cfg.sample_size:
            break


def _download_url(cfg: DownloadConfig) -> Iterator[Dict[str, Any]]:
    """从 URL 下载 jsonl/txt 文件。"""
    cache_dir = Path(cfg.cache_dir or "cache/downloads")
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_file = cache_dir / (cfg.cache_key + Path(cfg.name).suffix or ".dat")

    if not local_file.exists():
        logger.info(f"Downloading {cfg.name} → {local_file}")
        with urllib.request.urlopen(cfg.name) as resp, open(local_file, "wb") as f:
            shutil.copyfileobj(resp, f)

    suffix = local_file.suffix
    count = 0
    if suffix in (".jsonl", ".json"):
        for row in read_jsonl(local_file):
            yield row
            count += 1
            if cfg.sample_size and count >= cfg.sample_size:
                break
    else:
        with open(local_file, "r", encoding="utf-8") as f:
            for line in f:
                yield {"text": line.rstrip("\n")}
                count += 1
                if cfg.sample_size and count >= cfg.sample_size:
                    break


def _download_local(cfg: DownloadConfig) -> Iterator[Dict[str, Any]]:
    """从本地文件加载。支持 jsonl / json / txt。"""
    p = Path(cfg.name)
    if not p.exists():
        raise FileNotFoundError(p)
    count = 0
    if p.suffix == ".jsonl":
        for row in read_jsonl(p):
            yield row
            count += 1
            if cfg.sample_size and count >= cfg.sample_size:
                break
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for row in data:
                yield row
                count += 1
                if cfg.sample_size and count >= cfg.sample_size:
                    break
        else:
            yield data
    else:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                yield {"text": line.rstrip("\n")}
                count += 1
                if cfg.sample_size and count >= cfg.sample_size:
                    break


DOWNLOAD_HANDLERS: Dict[str, Callable[[DownloadConfig], Iterator[Dict[str, Any]]]] = {
    "huggingface": _download_huggingface,
    "modelscope":  _download_modelscope,
    "url":         _download_url,
    "local":       _download_local,
}


# ============================================================
# 主入口
# ============================================================

def download_dataset(cfg: DownloadConfig, output_dir: str = "data/raw") -> str:
    """
    下载单个数据集并保存为统一的 jsonl 格式。
    返回 output_path。
    """
    handler = DOWNLOAD_HANDLERS.get(cfg.source)
    if handler is None:
        raise ValueError(f"Unsupported source: {cfg.source}")

    # 决定 output_path
    if cfg.output_path:
        out_path = Path(cfg.output_path)
    else:
        safe_name = cfg.name.replace("/", "__")
        suffix = f"__{cfg.config}" if cfg.config else ""
        out_path = Path(output_dir) / f"{cfg.source}__{safe_name}{suffix}__{cfg.split}.jsonl"

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"[Download] cached: {out_path}")
        return str(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Download] {cfg.source}://{cfg.name} → {out_path}")

    written = 0
    t0 = time.time()

    def _row_iter():
        nonlocal written
        for raw_row in handler(cfg):
            mapped = _apply_field_map(raw_row, cfg.field_map)
            if mapped is None:
                continue
            yield mapped
            written += 1

    pbar = tqdm(_row_iter(), desc=f"downloading {cfg.name}", unit="row")
    write_jsonl(out_path, pbar)
    pbar.close()

    elapsed = time.time() - t0
    logger.info(f"[Download] done: {written:,} rows in {elapsed:.1f}s → {out_path}")
    return str(out_path)


def download_recipe(
    recipe: List[Dict[str, Any]],
    output_dir: str = "data/raw",
    cache_dir: Optional[str] = None,
) -> List[str]:
    """下载一个完整 recipe 的所有数据集，返回所有输出 jsonl 路径列表。"""
    paths = []
    for entry in recipe:
        cfg = DownloadConfig(
            source=entry["source"],
            name=entry["name"],
            config=entry.get("config"),
            split=entry.get("split", "train"),
            sample_size=entry.get("sample_size"),
            field_map=entry.get("field_map", {}),
            cache_dir=cache_dir,
        )
        try:
            p = download_dataset(cfg, output_dir=output_dir)
            paths.append(p)
        except Exception as e:
            logger.warning(f"[Download] failed: {cfg.name}: {e}")
    return paths


def list_supported_datasets() -> List[str]:
    return list(DOWNLOAD_HANDLERS.keys())
