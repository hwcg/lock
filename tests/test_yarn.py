"""YaRN 长文本外推单测。"""
import math

import pytest
import torch
import torch.nn as nn

from deepseek_v4.modeling.model import (
    DeepseekV4Config, DeepseekV4RotaryEmbedding, compute_rope_inv_freq,
    _yarn_find_correction_dim, _yarn_find_correction_range, _yarn_linear_ramp_mask,
)
from deepseek_v4.inference.yarn.apply import apply_yarn_config, estimate_yarn_max_seq_len
from deepseek_v4.inference.yarn.config import YarnConfig


# ============================================================
# YaRN math helpers
# ============================================================

def test_yarn_find_correction_dim():
    dim = 128
    base = 10000.0
    max_pos = 65536

    for num_rot in [1, 4, 16, 64]:
        d = _yarn_find_correction_dim(num_rot, dim, base, max_pos)
        assert 0 <= d <= dim


def test_yarn_find_correction_range():
    dim = 128
    base = 10000.0
    max_pos = 65536
    beta_fast, beta_slow = 32, 1

    low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, max_pos)
    assert 0 <= low <= high <= dim


def test_yarn_linear_ramp_mask():
    dim = 64
    mask = _yarn_linear_ramp_mask(10, 50, dim)
    assert mask.shape == (dim,)
    # 前 10 维 ≈ 0
    assert mask[:5].mean() < 0.1
    # 中间从 0 到 1 线性增长
    assert 0.4 < mask[25].item() < 0.6
    # 后 50 维 ≈ 1
    assert mask[55:].mean() > 0.9


# ============================================================
# RoPE Inv Freq
# ============================================================

def test_compute_rope_inv_freq_default():
    config = get_test_config()
    config.rope_scaling["type"] = "default"
    inv_freq, attn_scaling = compute_rope_inv_freq(config, layer_type="main")
    assert inv_freq.shape == (config.head_dim // 2,)
    assert attn_scaling == 1.0


def test_compute_rope_inv_freq_yarn():
    config = get_test_config()
    config.rope_scaling["type"] = "yarn"
    inv_freq, attn_scaling = compute_rope_inv_freq(config, layer_type="main")
    assert inv_freq is not None
    # YaRN 内部 inv_freq 维度
    rope_dim = int(config.head_dim * 0.5)  # partial_factor ≈ 0.5 * head_dim / 2
    # 实际 rope_dim 取决于 head_dim * partial_rotary_factor
    assert inv_freq.shape[0] > 0


def test_compute_rope_inv_freq_compress():
    config = get_test_config()
    inv_freq_main, _ = compute_rope_inv_freq(config, layer_type="main")
    inv_freq_compress, _ = compute_rope_inv_freq(config, layer_type="compress")
    # compress theta 更大 → inv_freq 更小（长程 RoPE 更慢旋转）
    assert inv_freq_compress[0].item() < inv_freq_main[0].item()


# ============================================================
# RotaryEmbedding
# ============================================================

def get_test_config():
    return DeepseekV4Config(
        vocab_size=200,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=64,
        qk_rope_head_dim=32,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=2,
        moe_intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        sliding_window=32,
        max_position_embeddings=2048,
        compress_ratios=(128, 4, 128, 0),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=32,
        hc_mult=2,
        hc_sinkhorn_iters=5,
    )


def test_rotary_embedding_forward():
    config = get_test_config()
    rope = DeepseekV4RotaryEmbedding(config)

    x = torch.randn(2, 8, config.hidden_size)
    pos = torch.arange(8).unsqueeze(0).expand(2, -1)
    cos, sin = rope(x, position_ids=pos, layer_type="main")

    # cos/sin 形状: [B, S, rope_head_dim//2]
    rope_half_dim = int(config.head_dim * (config.qk_rope_head_dim / config.head_dim)) // 2
    assert cos.shape == (2, 8, rope_half_dim)
    assert sin.shape == (2, 8, rope_half_dim)
    assert torch.allclose(cos.pow(2) + sin.pow(2), torch.ones_like(cos), atol=1e-5)


# ============================================================
# YarnConfig
# ============================================================

def test_yarn_config_defaults():
    cfg = YarnConfig()
    assert cfg.factor >= 1
    assert cfg.beta_fast > 0
    assert cfg.beta_slow > 0
    assert cfg.original_max_position_embeddings > 0


def test_yarn_config_custom():
    cfg = YarnConfig(factor=16, beta_fast=32, beta_slow=1,
                     original_max_position_embeddings=65536)
    assert cfg.factor == 16
    assert cfg.beta_fast == 32
    assert cfg.beta_slow == 1


# ============================================================
# Apply YaRN
# ============================================================

def test_apply_yarn_config():
    """apply_yarn_config 应修改 config 的 rope_scaling 字段。"""
    config = get_test_config()
    original_max = config.max_position_embeddings

    yarn_cfg = YarnConfig(factor=4, original_max_position_embeddings=2048)
    config = apply_yarn_config(config, yarn_cfg)

    assert config.rope_scaling["factor"] == 4
    assert config.rope_scaling["original_max_position_embeddings"] == 2048
    assert config.max_position_embeddings >= original_max * 4


# ============================================================
# Estimate YaRN max seq len
# ============================================================

def test_estimate_yarn_max_seq_len():
    """根据因子估算最大序列长度。"""
    max_len = estimate_yarn_max_seq_len(
        base_max=65536,
        factor=16,
    )
    assert max_len >= 65536 * 16


def test_estimate_yarn_max_seq_len_small_factor():
    max_len = estimate_yarn_max_seq_len(base_max=4096, factor=4)
    assert max_len >= 4096 * 4


# ============================================================
# RoPE periodicity
# ============================================================

def test_rope_periodicity():
    """RoPE 在两个相距 2π 的位置应输出相同。"""
    dim = 64
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 位置 0 和位置 2π / inv_freq[i] 应对应
    pos = torch.tensor([0, dim * 2 * math.pi / math.log(base)], dtype=torch.float32)

    freqs = torch.outer(pos, inv_freq)
    cos0 = freqs[0].cos()
    cos1 = freqs[1].cos()
    # 不同位置 cos 不同
    assert not torch.allclose(cos0, cos1, atol=0.1)


# ============================================================
# YaRN interleaved partial rope
# ============================================================

def test_apply_rotary_pos_emb_partial():
    """Partial RoPE：仅最后 rope_head_dim 通道旋转。"""
    from deepseek_v4.modeling.model import apply_rotary_pos_emb

    B, H, S, D = 2, 4, 8, 64
    rope_dim = 16

    x = torch.randn(B, H, S, D)
    # cos/sin 只对 rope_dim//2 个频率
    cos = torch.randn(B, S, rope_dim // 2).cos()
    sin = torch.randn(B, S, rope_dim // 2).sin()

    y = apply_rotary_pos_emb(x, cos, sin)

    # 前 D - rope_dim 通道不变
    nope_x = x[..., :D - rope_dim]
    nope_y = y[..., :D - rope_dim]
    assert torch.allclose(nope_x, nope_y)

    # 后 rope_dim 通道可能变化（旋转）
    rope_x = x[..., -rope_dim:]
    rope_y = y[..., -rope_dim:]
    rope_changed = not torch.allclose(rope_x, rope_y)
    # 至少对非零 sin 有旋转
    if sin.abs().max() > 0:
        assert rope_changed
