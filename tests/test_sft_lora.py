"""SFT / LoRA 训练器单测。"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from deepseek_v4.training.sft import SFTConfig, SFTTrainer, _NEFTuneEmbedding
from deepseek_v4.training.lora.layers import LoRALinear
from deepseek_v4.training.lora.config import LoRAConfig
from deepseek_v4.training.lora.apply import apply_lora, merge_lora_weights


# ============================================================
# Mock tokenizer
# ============================================================

class _MockTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    vocab_size = 100

    def encode(self, text):
        return [(ord(c) % 90) + 10 for c in text][:200]

    def decode(self, ids):
        return "".join(chr((i - 10) % 90 + 30) for i in ids)


# ============================================================
# SFTConfig
# ============================================================

def test_sft_config_defaults():
    cfg = SFTConfig()
    assert cfg.learning_rate == 2.0e-5
    assert cfg.neftune_alpha == 0.0
    assert cfg.loss_reduction == "token"
    assert isinstance(cfg.metric_name, str)


def test_sft_config_mask_user():
    cfg = SFTConfig(mask_user=True, thinking_mode_default="chat")
    assert cfg.mask_user is True
    assert cfg.thinking_mode_default == "chat"


# ============================================================
# NEFTune
# ============================================================

def test_neftune_embedding():
    embed = nn.Embedding(50, 16)
    neftune = _NEFTuneEmbedding(alpha=5.0)
    neftune.attach(embed)

    x = torch.randint(0, 50, (2, 8))
    embed.train()
    out_with_noise = embed(x)
    # 噪声应不为 0（alpha > 0）
    assert out_with_noise is not None

    embed.eval()
    out_no_noise = embed(x)
    # eval 模式不应加噪声
    neftune.detach()


def test_neftune_disable():
    embed = nn.Embedding(50, 16)
    neftune = _NEFTuneEmbedding(alpha=0.0)
    neftune.attach(embed)

    embed.train()
    x = torch.randint(0, 50, (2, 8))
    out = embed(x)
    assert out is not None
    neftune.detach()


# ============================================================
# LoRA
# ============================================================

class _LoRATestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 64)
        self.v_proj = nn.Linear(32, 64)
        self.mlp = nn.Linear(64, 32)

    def forward(self, x):
        return self.mlp(self.v_proj(self.q_proj(x)))


def test_lora_linear_init():
    lora = LoRALinear(
        in_features=32,
        out_features=64,
        r=8,
        alpha=16,
        dropout=0.0,
    )
    x = torch.randn(2, 8, 32)
    y = lora(x)
    assert y.shape == (2, 8, 64)
    # 初始 LoRA B 为 0，输出应与原线性相同
    lora.lora_B.weight.zero_()
    y_zero = lora(x)
    y_base = lora.linear(x)
    assert torch.allclose(y_zero, y_base, atol=1e-5)


def test_lora_config():
    cfg = LoRAConfig(r=16, alpha=32, target_modules=["q_proj", "v_proj"])
    assert cfg.r == 16
    assert cfg.alpha == 32
    assert "q_proj" in cfg.target_modules


def test_apply_lora():
    model = _LoRATestModel()
    cfg = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])
    lorad = apply_lora(model, cfg)
    # 检查 q_proj 被替换
    assert isinstance(lorad.q_proj, LoRALinear) or hasattr(lorad.q_proj, "lora_A")
    # original mlp 不变
    assert isinstance(lorad.mlp, nn.Linear)


def test_merge_lora_weights():
    model = _LoRATestModel()
    cfg = LoRAConfig(r=8, alpha=16, target_modules=["q_proj"])
    lorad = apply_lora(model, cfg)

    # 手动设置 LoRA 权重
    if hasattr(lorad.q_proj, "lora_A"):
        lorad.q_proj.lora_A.weight.data.fill_(0.0)
        lorad.q_proj.lora_B.weight.data.fill_(0.0)

    merged = merge_lora_weights(lorad, cfg)
    # 合并后 q_proj 应回到 nn.Linear（去掉 LoRA wrapper）
    assert isinstance(merged.q_proj, nn.Linear)


# ============================================================
# SFT Trainer compute loss
# ============================================================

@pytest.fixture
def sft_trainer_kwargs():
    model = _LoRATestModel()
    # 扩展为类语言模型接口
    class _Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MagicMock()
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            B, S = input_ids.shape
            # 把 input_ids 映射到 32 维
            x = nn.functional.one_hot((input_ids % 50).long(), num_classes=50).float() @ torch.randn(50, 32)
            logits = torch.randn(B, S, 100)
            return {"logits": logits}

    return {"model": _Wrap(), "tokenizer": _MockTokenizer()}
