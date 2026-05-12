"""数据流水线单测。"""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from deepseek_v4.data.cleaning import (
    LanguageFilter, QualityFilter, TextCleaner,
)
from deepseek_v4.data.collator import DPOCollator, PadCollator
from deepseek_v4.data.dataset import DPODataset, PretrainDataset, SFTDataset
from deepseek_v4.data.dedup import (
    ExactDeduper, MinHashDeduper, SimHashDeduper, get_shingles,
)
from deepseek_v4.utils.io import write_jsonl


# ---------- Cleaner ----------

def test_text_cleaner():
    c = TextCleaner()
    assert c("<p>hello</p>   world\n\n\n\n!") == "hello world\n\n!"
    assert c("") == ""


def test_quality_filter():
    qf = QualityFilter(min_length=10)
    assert qf("a") is False
    assert qf("this is a normal sentence and should pass") is True
    # 重复
    assert qf("same\n" * 50) is False


def test_language_filter():
    lf = LanguageFilter(allowed_languages=["zh"])
    assert lf("你好世界 这是中文测试 包含足够字符") is True
    assert lf("hello world this is english text") is False


# ---------- Dedup ----------

def test_exact_deduper():
    d = ExactDeduper()
    assert d.is_duplicate("hello") is False
    assert d.is_duplicate("hello") is True
    assert d.is_duplicate("world") is False


def test_shingles():
    s = get_shingles("hello world this is a test", n=3)
    assert len(s) > 0


def test_minhash_deduper():
    d = MinHashDeduper(num_perm=64, threshold=0.7)
    a = "the quick brown fox jumps over the lazy dog every day in the morning"
    b = "the quick brown fox jumps over the lazy dog every day in the afternoon"
    c = "completely unrelated content about machine learning and language models"
    assert d.is_duplicate(a) is False
    assert d.is_duplicate(b) is True   # 与 a 高度相似
    assert d.is_duplicate(c) is False


def test_simhash_deduper():
    d = SimHashDeduper(hamming_threshold=3)
    a = "this is a sample document about deepseek v4 model training framework"
    b = "this is a sample document about deepseek v4 model training framework!"  # 几乎相同
    c = "completely different content about quantum computing and physics theory"
    assert d.is_duplicate(a) is False
    assert d.is_duplicate(b) is True
    assert d.is_duplicate(c) is False


# ---------- Dataset (需要 tokenizer，这里 mock) ----------

class _MockTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 100

    def encode(self, text):
        # 一个字符一个 id
        return [(ord(c) % 90) + 10 for c in text][:200]


@pytest.fixture
def tok():
    return _MockTokenizer()


@pytest.fixture
def pretrain_file(tmp_path):
    p = tmp_path / "pretrain.jsonl"
    write_jsonl(p, [{"text": f"hello world {i}"} for i in range(20)])
    return str(p)


def test_pretrain_dataset(tok, pretrain_file, tmp_path):
    ds = PretrainDataset([pretrain_file], tok, max_seq_len=64, cache_dir=str(tmp_path / "cache"))
    assert len(ds) == 20
    item = ds[0]
    assert "input_ids" in item and "labels" in item


def test_collator():
    coll = PadCollator(pad_token_id=2, ignore_index=-100)
    batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5])},
    ]
    out = coll(batch)
    assert out["input_ids"].shape == (2, 3)
    assert "attention_mask" in out


def test_sft_dataset(tok, tmp_path):
    p = tmp_path / "sft.jsonl"
    write_jsonl(p, [
        {"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]},
    ])
    ds = SFTDataset([str(p)], tok, max_seq_len=128, cache_dir=str(tmp_path / "cache"))
    assert len(ds) == 1
    item = ds[0]
    # 至少有部分 label 非 ignore_index
    assert (item["labels"] != -100).sum() > 0


def test_dpo_dataset(tok, tmp_path):
    p = tmp_path / "dpo.jsonl"
    write_jsonl(p, [
        {
            "prompt": "What is 1+1?",
            "chosen": "1+1 equals 2.",
            "rejected": "I don't know.",
        }
    ])
    ds = DPODataset([str(p)], tok, max_prompt_len=32, max_seq_len=64,
                    cache_dir=str(tmp_path / "cache"))
    assert len(ds) == 1
    item = ds[0]
    assert "chosen_ids" in item and "rejected_ids" in item
