"""
分片 safetensors 读写工具。

格式与 HuggingFace 一致：
    model.safetensors.index.json
        { "metadata": {"total_size": ...},
          "weight_map": { "param.name": "model-00001-of-00003.safetensors", ... } }
    model-00001-of-NNNNN.safetensors
    ...
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch


_SHARD_RE = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")
_FORBID_ZERO_BYTES = True


def _parse_size(size: Union[int, str]) -> int:
    """'5GB' / '500MB' / int → bytes。"""
    if isinstance(size, int):
        return size
    s = str(size).upper().strip()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for u in ["TB", "GB", "MB", "KB", "B"]:
        if s.endswith(u):
            return int(float(s[: -len(u)]) * units[u])
    return int(s)


def save_sharded_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Union[str, Path],
    max_shard_size: Union[int, str] = "5GB",
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    把 state_dict 写到 output_dir 下的多个 safetensors 分片。

    Returns:
        weight_map: param_name → shard_filename
    """
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _parse_size(max_shard_size)

    # 数据 dtype 化 + 强制 contiguous + 移到 cpu
    cleaned: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().contiguous().cpu()
        if dtype is not None:
            t = t.to(dtype=dtype)
        cleaned[k] = t

    # 切片
    shards: List["OrderedDict[str, torch.Tensor]"] = []
    cur: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    cur_size = 0
    for k, v in cleaned.items():
        sz = v.numel() * v.element_size()
        if cur and cur_size + sz > max_bytes:
            shards.append(cur)
            cur = OrderedDict()
            cur_size = 0
        cur[k] = v
        cur_size += sz
    if cur:
        shards.append(cur)

    n = len(shards)
    weight_map: Dict[str, str] = {}
    md = dict(metadata or {})
    md.setdefault("format", "pt")

    total_size = 0
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        save_file(dict(shard), str(output_dir / fname), metadata=md)
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += v.numel() * v.element_size()

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return weight_map


def load_sharded_safetensors(
    directory: Union[str, Path],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    读取 HF 风格的分片 safetensors（或单文件）。
    """
    from safetensors.torch import load_file

    directory = Path(directory)
    idx_file = directory / "model.safetensors.index.json"
    state_dict: Dict[str, torch.Tensor] = {}

    if idx_file.exists():
        with open(idx_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        for fn in files:
            state_dict.update(load_file(str(directory / fn), device=device))
    else:
        single = directory / "model.safetensors"
        if single.exists():
            state_dict.update(load_file(str(single), device=device))
        else:
            # 退化：扫所有 .safetensors
            for p in sorted(directory.glob("*.safetensors")):
                state_dict.update(load_file(str(p), device=device))
    return state_dict
