"""格式转换工具单测。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from deepseek_v4.inference.convert.safetensors_utils import (
    _parse_size, load_sharded_safetensors, save_sharded_safetensors,
)
from deepseek_v4.inference.convert.to_ollama import build_ollama_modelfile


# ============ _parse_size ============

def test_parse_size_int():
    assert _parse_size(1024) == 1024


def test_parse_size_kb_mb_gb():
    assert _parse_size("1KB") == 1024
    assert _parse_size("1MB") == 1024 ** 2
    assert _parse_size("5GB") == 5 * (1024 ** 3)
    assert _parse_size("0.5GB") == int(0.5 * 1024 ** 3)


def test_parse_size_default_bytes():
    assert _parse_size("12345") == 12345


# ============ shard save / load roundtrip ============

def test_save_load_sharded_roundtrip(tmp_path):
    sd = {
        "embed.weight": torch.randn(100, 64),
        "layer.0.q.weight": torch.randn(64, 64),
        "layer.0.k.weight": torch.randn(64, 64),
        "layer.1.q.weight": torch.randn(64, 64),
        "head.bias": torch.zeros(100),
    }
    weight_map = save_sharded_safetensors(sd, tmp_path, max_shard_size="50KB")
    # 索引存在
    idx_file = tmp_path / "model.safetensors.index.json"
    assert idx_file.exists()
    with open(idx_file) as f:
        idx = json.load(f)
    assert "weight_map" in idx
    # 加载
    loaded = load_sharded_safetensors(tmp_path)
    assert set(loaded.keys()) == set(sd.keys())
    for k in sd:
        torch.testing.assert_close(loaded[k], sd[k].contiguous().cpu())


def test_save_sharded_with_dtype(tmp_path):
    sd = {"x": torch.randn(10, 10, dtype=torch.float32)}
    save_sharded_safetensors(sd, tmp_path, dtype=torch.float16)
    loaded = load_sharded_safetensors(tmp_path)
    assert loaded["x"].dtype == torch.float16


def test_save_sharded_creates_multiple_shards(tmp_path):
    # 强制 5KB 分片，每个 tensor ~256B (8x8 float32=256B)
    sd = {f"p{i}": torch.randn(8, 8) for i in range(50)}
    save_sharded_safetensors(sd, tmp_path, max_shard_size="5KB")
    n_shards = len(list(tmp_path.glob("model-*-of-*.safetensors")))
    assert n_shards > 1


# ============ Ollama Modelfile ============

def test_ollama_modelfile_contains_from(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"dummy")
    mf = build_ollama_modelfile(gguf_path=gguf, temperature=0.5, top_p=0.9, num_ctx=8192)
    assert f"FROM {gguf.resolve()}" in mf
    assert "PARAMETER temperature 0.5" in mf
    assert "PARAMETER top_p 0.9" in mf
    assert "PARAMETER num_ctx 8192" in mf
    assert 'TEMPLATE """' in mf


def test_ollama_modelfile_with_system(tmp_path):
    gguf = tmp_path / "m.gguf"; gguf.write_bytes(b"")
    mf = build_ollama_modelfile(gguf_path=gguf, system_prompt="You are helpful.")
    assert 'SYSTEM """You are helpful."""' in mf


def test_ollama_modelfile_stops(tmp_path):
    gguf = tmp_path / "m.gguf"; gguf.write_bytes(b"")
    mf = build_ollama_modelfile(gguf_path=gguf, stop_strings=["</s>", "User:"])
    assert 'PARAMETER stop "</s>"' in mf
    assert 'PARAMETER stop "User:"' in mf
