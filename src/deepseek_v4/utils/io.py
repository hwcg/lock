"""安全文件 IO 工具：原子写入、jsonl 流式读写、压缩支持。"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union


# ---------------- 原子文件操作 ----------------

def atomic_write(path: Union[str, Path], content: Union[str, bytes], mode: str = "w") -> None:
    """
    原子写入：先写临时文件，再 rename。
    保证写入不会出现半成品文件（即便进程崩溃）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 与目标文件同一目录的临时文件保证 rename 在同一文件系统
    suffix = ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, mode) as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------- JSON ----------------

def safe_load_json(path: Union[str, Path], default: Any = None) -> Any:
    """加载 json，文件不存在时返回 default。"""
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_save_json(path: Union[str, Path], obj: Any, indent: int = 2) -> None:
    """原子写入 json。"""
    text = json.dumps(obj, ensure_ascii=False, indent=indent)
    atomic_write(path, text, mode="w")


# ---------------- JSONL ----------------

def _open_smart(path: Union[str, Path], mode: str):
    """支持 .jsonl / .jsonl.gz 自动识别。"""
    path = Path(path)
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t" if "b" not in mode else mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def read_jsonl(path: Union[str, Path], skip_invalid: bool = True) -> Iterator[Dict[str, Any]]:
    """流式读取 jsonl / jsonl.gz。"""
    with _open_smart(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                if not skip_invalid:
                    raise
                # 静默跳过坏行
                continue


def write_jsonl(
    path: Union[str, Path],
    rows: Iterable[Dict[str, Any]],
    mode: str = "w",
    flush_every: int = 1000,
) -> int:
    """流式写 jsonl，返回写入的行数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with _open_smart(path, mode) as f:
        for i, row in enumerate(rows):
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
            if (i + 1) % flush_every == 0:
                f.flush()
    return count


# ---------------- 通用 ----------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """复制文件（自动创建目标目录）。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def file_size(path: Union[str, Path]) -> int:
    """文件大小（字节）。"""
    return Path(path).stat().st_size


def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
    """列出目录下匹配 pattern 的文件。"""
    d = Path(directory)
    if not d.exists():
        return []
    if recursive:
        return sorted(d.rglob(pattern))
    return sorted(d.glob(pattern))
