"""Tokenizer 单测：BPE / Special Tokens / Encoding。"""
import json
import tempfile
from pathlib import Path

import pytest

from deepseek_v4.tokenizer.bpe import (
    BPETokenizer, BPETrainer, BPETrainerConfig, decode_bytes, encode_bytes,
    pretokenize,
)
from deepseek_v4.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, SpecialTokens,
    UNK_TOKEN, USER_TOKEN, ASSISTANT_TOKEN,
)
from deepseek_v4.tokenizer.encoding import (
    encode_messages, parse_message_from_completion_text, render_tools,
    merge_tool_messages,
)


# ============================================================
# BPE byte encoding
# ============================================================

def test_encode_decode_bytes_roundtrip():
    for s in ["hello", "你好世界", "混合123English"]:
        assert decode_bytes(encode_bytes(s)) == s


def test_pretokenize_english():
    tokens = pretokenize("Hello world, how are you?")
    assert len(tokens) > 1
    # 验证基础切分
    assert any("Hello" in t for t in tokens)


def test_pretokenize_cjk():
    tokens = pretokenize("你好世界")
    assert len(tokens) >= 1


def test_pretokenize_numbers():
    tokens = pretokenize("abc123def")
    assert len(tokens) >= 1


# ============================================================
# BPE Trainer
# ============================================================

def test_bpe_trainer_small():
    config = BPETrainerConfig(vocab_size=300, min_frequency=1, show_progress=False)
    trainer = BPETrainer(config)
    texts = [
        "hello world " * 5,
        "hello there " * 5,
        "world peace " * 5,
    ]
    vocab, merges = trainer.train(texts=texts)
    assert len(vocab) == 300
    assert len(merges) > 0
    # 初始字符都在 vocab 里
    for ch in encode_bytes("h"):
        assert ch in vocab


def test_bpe_trainer_save_load(tmp_path):
    config = BPETrainerConfig(vocab_size=300, min_frequency=1, show_progress=False)
    trainer = BPETrainer(config)
    trainer.train(texts=["hello world " * 10])
    trainer.save(str(tmp_path))

    tok = BPETokenizer.from_directory(str(tmp_path))
    assert tok.vocab_size == 300
    ids = tok.encode("hello world")
    assert len(ids) > 0
    assert tok.decode(ids) == "hello world"


# ============================================================
# BPE Tokenizer
# ============================================================

@pytest.fixture
def tiny_bpe_tokenizer():
    """快速构造一个微型 BPE tokenizer。"""
    config = BPETrainerConfig(vocab_size=500, min_frequency=1, show_progress=False)
    trainer = BPETrainer(config)
    trainer.train(texts=["hello world " * 10, "good morning " * 10, "test " * 10])
    return BPETokenizer(vocab=trainer.vocab, merges=trainer.merges)


def test_bpe_encode_decode_roundtrip(tiny_bpe_tokenizer):
    text = "hello world test"
    ids = tiny_bpe_tokenizer.encode(text)
    decoded = tiny_bpe_tokenizer.decode(ids)
    assert decoded == text


def test_bpe_encode_empty(tiny_bpe_tokenizer):
    assert tiny_bpe_tokenizer.encode("") == []


def test_bpe_token_to_id(tiny_bpe_tokenizer):
    # 空格令牌应该存在
    space_bytes = encode_bytes(" ")
    assert tiny_bpe_tokenizer.token_to_id(space_bytes) is not None


# ============================================================
# Special Tokens
# ============================================================

def test_special_tokens_constants():
    st = SpecialTokens.default()
    assert st.bos == BOS_TOKEN
    assert st.eos == EOS_TOKEN
    assert st.pad == PAD_TOKEN
    assert st.unk == UNK_TOKEN
    assert st.user == USER_TOKEN
    assert st.assistant == ASSISTANT_TOKEN
    assert len(st.all_tokens) == len(ALL_SPECIAL_TOKENS)


def test_special_tokens_id_map():
    st = SpecialTokens.default()
    id_map = st.all_ids_map
    assert id_map[st.bos] == 0
    assert id_map[st.eos] == 1
    assert id_map[st.pad] == 2
    assert st.is_special(st.bos) is True
    assert st.is_special("random") is False


def test_special_tokens_order():
    """ALL_SPECIAL_TOKENS 顺序必须与 ID 一一对应。"""
    for i, tok in enumerate(ALL_SPECIAL_TOKENS):
        assert SpecialTokens.default().all_ids_map[tok] == i


# ============================================================
# Encode Messages
# ============================================================

def test_encode_messages_basic_chat():
    messages = [
        {"role": "user", "content": "Hi"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    assert BOS_TOKEN in prompt
    assert USER_TOKEN in prompt
    assert ASSISTANT_TOKEN in prompt
    assert "Hi" in prompt


def test_encode_messages_multi_turn():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    assert "Q1" in prompt
    assert "A1" in prompt
    assert "Q2" in prompt


def test_encode_messages_with_context():
    context = [
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "prev response"},
    ]
    messages = [
        {"role": "user", "content": "hi"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat", context=context)
    assert "previous" in prompt
    assert "hi" in prompt


def test_encode_messages_thinking_mode():
    messages = [
        {"role": "user", "content": "Solve this."},
        {"role": "assistant", "content": "42", "reasoning_content": "thinking..."},
    ]
    prompt = encode_messages(messages, thinking_mode="thinking", drop_thinking=False)
    assert "thinking..." in prompt


def test_encode_messages_with_tools():
    messages = [
        {
            "role": "system",
            "content": "You have tools.",
            "tools": [{"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {}}}],
        },
        {"role": "user", "content": "calc 1+1"},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    assert "calc" in prompt


# ============================================================
# Merge Tool Messages
# ============================================================

def test_merge_tool_messages():
    messages = [
        {"role": "user", "content": "call calc"},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "result=42"},
    ]
    merged = merge_tool_messages(messages)
    # tool 消息应合并到 user
    roles = [m["role"] for m in merged]
    assert "tool" not in roles


# ============================================================
# Parse Message From Completion Text
# ============================================================

def test_parse_simple_response():
    text = "The answer is 42." + EOS_TOKEN
    result = parse_message_from_completion_text(text, thinking_mode="chat")
    assert result["role"] == "assistant"
    assert result["content"] == "The answer is 42."
    assert result["reasoning_content"] == ""


def test_parse_with_reasoning():
    from deepseek_v4.tokenizer.special_tokens import THINK_START, THINK_END
    text = THINK_START + "hmm..." + THINK_END + "final answer" + EOS_TOKEN
    result = parse_message_from_completion_text(text, thinking_mode="thinking")
    assert result["role"] == "assistant"
    assert "final answer" in result["content"]
    assert "hmm..." in result["reasoning_content"]


# ============================================================
# Render Tools
# ============================================================

def test_render_tools():
    tools = [{"name": "calculator", "description": "Perform calculations", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}]
    rendered = render_tools(tools)
    assert "calculator" in rendered
    assert "tool_calls" in rendered
