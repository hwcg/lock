"""集成测试通用 fixture。"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from deepseek_v4.modeling.model import DeepseekV4Config, DeepseekV4ForCausalLM
from deepseek_v4.tokenizer.bpe import BPETokenizer, BPETrainer, BPETrainerConfig
from deepseek_v4.tokenizer.special_tokens import ALL_SPECIAL_TOKENS
from deepseek_v4.tokenizer.tokenizer import DeepseekV4Tokenizer
from deepseek_v4.utils.io import write_jsonl


# ------------------------------------------------------------------
# 极小模型配置（快到能在 CI 上跑完）
# ------------------------------------------------------------------

def _tiny_model_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        max_position_embeddings=256,
        q_lora_rank=32, o_lora_rank=16, o_groups=2,
        sliding_window=16,
        compress_ratios=(128, 0),                 # 1 HCA + 1 sliding
        compress_rope_theta=160000.0,
        index_n_heads=2, index_head_dim=16, index_topk=8,
        moe_intermediate_size=64,
        n_routed_experts=4, n_shared_experts=1,
        num_experts_per_tok=2, num_hash_layers=0,
        hc_mult=2, hc_sinkhorn_iters=2,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn", "factor": 2,
            "original_max_position_embeddings": 128,
            "beta_fast": 32, "beta_slow": 1,
        },
    )


@pytest.fixture(scope="session")
def tmp_workspace():
    """整套集成测试共享 workspace。"""
    d = Path(tempfile.mkdtemp(prefix="ds4_e2e_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_workspace):
    """训练一个极小 tokenizer。"""
    out = tmp_workspace / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    texts = (
        ["hello world"] * 30 +
        ["the quick brown fox jumps over the lazy dog"] * 20 +
        ["你好世界"] * 30 +
        ["1+1=2", "2+2=4", "3+3=6"] * 10 +
        ["question: what is x? answer: y"] * 10
    )
    cfg = BPETrainerConfig(
        vocab_size=512, min_frequency=1,
        special_tokens=ALL_SPECIAL_TOKENS, show_progress=False,
    )
    trainer = BPETrainer(cfg)
    trainer.feed(iter(texts))
    vocab, merges = trainer.train()
    bpe = BPETokenizer(vocab=vocab, merges=merges, special_tokens=ALL_SPECIAL_TOKENS)
    tok = DeepseekV4Tokenizer(bpe_tokenizer=bpe, model_max_length=256)
    tok.save_pretrained(str(out))
    return tok, str(out)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_workspace, tiny_tokenizer):
    """构造并保存一个随机初始化的 tiny 模型 + config。"""
    tok, _tok_path = tiny_tokenizer
    cfg = _tiny_model_config()
    cfg.vocab_size = tok.vocab_size
    cfg.pad_token_id = tok.pad_token_id
    cfg.bos_token_id = tok.bos_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = DeepseekV4ForCausalLM(cfg)
    model.init_weights()

    out = tmp_workspace / "tiny_model"
    out.mkdir(parents=True, exist_ok=True)
    # 保存 config + state_dict
    from safetensors.torch import save_file
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return str(out)


@pytest.fixture(scope="session")
def tiny_model_config_path(tmp_workspace):
    """保存 tiny config 为 JSON 文件，供训练脚本使用。"""
    p = tmp_workspace / "tiny_config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_tiny_model_config().to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)


@pytest.fixture(scope="session")
def tiny_pretrain_data(tmp_workspace):
    """生成一份小语料用于 pretrain 测试。"""
    p = tmp_workspace / "pretrain.jsonl"
    rows = [{"text": f"this is sample text number {i}, hello world {i}"} for i in range(64)]
    write_jsonl(p, rows)
    return str(p)


@pytest.fixture(scope="session")
def tiny_sft_data(tmp_workspace):
    p = tmp_workspace / "sft.jsonl"
    rows = []
    for i in range(32):
        rows.append({
            "messages": [
                {"role": "user", "content": f"compute {i}+{i}"},
                {"role": "assistant", "content": f"the answer is {2 * i}"},
            ]
        })
    write_jsonl(p, rows)
    return str(p)


@pytest.fixture(scope="session")
def tiny_dpo_data(tmp_workspace):
    p = tmp_workspace / "dpo.jsonl"
    rows = []
    for i in range(16):
        rows.append({
            "prompt": f"What is {i}+{i}?",
            "chosen": f"The answer is {2 * i}",
            "rejected": "I don't know",
        })
    write_jsonl(p, rows)
    return str(p)
