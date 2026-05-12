"""服务端协议 / 解析 / 路由单测。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.inference.server.parsing import (
    IncrementalParser, parse_full_completion,
)
from deepseek_v4.inference.server.protocol import (
    ChatCompletionRequest, ChatMessage, CompletionRequest, ToolDefinition,
)
from deepseek_v4.tokenizer.special_tokens import (
    DSML_TOKEN, EOS_TOKEN, THINK_END, THINK_START,
)


# ============================================================
# Protocol 解析
# ============================================================

def test_chat_request_basic():
    req = ChatCompletionRequest(
        model="deepseek-v4-mini",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.messages[0].role == "user"
    assert req.get_stop_list() == []


def test_resolve_thinking_mode_explicit():
    req = ChatCompletionRequest(model="x", messages=[], thinking_mode="thinking")
    assert req.resolve_thinking_mode() == "thinking"


def test_resolve_thinking_mode_via_open_thinking():
    req = ChatCompletionRequest(model="x", messages=[], open_thinking=True)
    assert req.resolve_thinking_mode() == "thinking"


def test_resolve_thinking_mode_via_enable_thinking():
    req = ChatCompletionRequest(model="x", messages=[], enable_thinking=False)
    assert req.resolve_thinking_mode(default="auto") == "chat"


def test_resolve_thinking_mode_default():
    req = ChatCompletionRequest(model="x", messages=[])
    assert req.resolve_thinking_mode(default="auto") == "chat"
    assert req.resolve_thinking_mode(default="thinking") == "thinking"


def test_get_max_tokens_priority():
    """max_completion_tokens 优先于 max_tokens。"""
    req = ChatCompletionRequest(
        model="x", messages=[], max_tokens=100, max_completion_tokens=200,
    )
    assert req.get_max_tokens(default=10) == 200
    req2 = ChatCompletionRequest(model="x", messages=[], max_tokens=100)
    assert req2.get_max_tokens(default=10) == 100
    req3 = ChatCompletionRequest(model="x", messages=[])
    assert req3.get_max_tokens(default=42) == 42


def test_stop_list_normalization():
    req1 = ChatCompletionRequest(model="x", messages=[], stop="\n\n")
    assert req1.get_stop_list() == ["\n\n"]
    req2 = ChatCompletionRequest(model="x", messages=[], stop=["a", "b"])
    assert req2.get_stop_list() == ["a", "b"]
    req3 = ChatCompletionRequest(model="x", messages=[])
    assert req3.get_stop_list() == []


def test_tool_definition():
    t = ToolDefinition(function={"name": "x", "description": "y", "parameters": {}})
    assert t.type == "function"


# ============================================================
# Streaming parser
# ============================================================

def test_parser_pure_content():
    p = IncrementalParser(thinking_mode="chat")
    out = p.feed("Hello, world!")
    assert len(out) >= 1
    assert "content" in out[0]
    assert out[0]["content"] == "Hello, world!"


def test_parser_thinking_block():
    text = THINK_START + "Let me think" + THINK_END + "The answer is 42."
    p = IncrementalParser(thinking_mode="chat")
    out = p.feed(text)
    # 应有 reasoning_content + content
    rc = "".join(d.get("reasoning_content", "") for d in out)
    cc = "".join(d.get("content", "") for d in out)
    assert rc == "Let me think"
    assert cc == "The answer is 42."


def test_parser_thinking_mode_default_in_think():
    """thinking_mode='thinking' 时，第一段就是 reasoning。"""
    text = "first reasoning..." + THINK_END + "final"
    p = IncrementalParser(thinking_mode="thinking")
    out = p.feed(text)
    rc = "".join(d.get("reasoning_content", "") for d in out)
    cc = "".join(d.get("content", "") for d in out)
    assert rc == "first reasoning..."
    assert cc == "final"


def test_parser_streaming_chunks():
    """分多段 feed 应得到与一次性 feed 等价的结果。"""
    text = THINK_START + "reasoning" + THINK_END + "answer"
    p1 = IncrementalParser(thinking_mode="chat")
    full_deltas = p1.feed(text)

    p2 = IncrementalParser(thinking_mode="chat")
    streamed: List[Dict] = []
    # 按字符喂
    for ch in text:
        streamed.extend(p2.feed(ch))

    def _aggregate(deltas):
        rc, cc = "", ""
        for d in deltas:
            rc += d.get("reasoning_content", "")
            cc += d.get("content", "")
        return rc, cc

    rc1, cc1 = _aggregate(full_deltas)
    rc2, cc2 = _aggregate(streamed)
    assert rc1 == rc2
    assert cc1 == cc2


def test_parser_tool_call_block():
    tool_text = (
        "I'll call the tool.\n\n"
        f"<{DSML_TOKEN}tool_calls>\n"
        f"<{DSML_TOKEN}invoke name=\"calc\">\n"
        f"<{DSML_TOKEN}parameter name=\"expr\" string=\"true\">1+1</{DSML_TOKEN}parameter>\n"
        f"</{DSML_TOKEN}invoke>\n"
        f"</{DSML_TOKEN}tool_calls>"
    )
    p = IncrementalParser(thinking_mode="chat")
    out = p.feed(tool_text)
    out.extend(p.finalize())
    # 应有 1 个 tool_calls delta
    tool_deltas = [d for d in out if "tool_calls" in d]
    assert len(tool_deltas) == 1
    tc = tool_deltas[0]["tool_calls"][0]
    assert tc["function"]["name"] == "calc"


def test_parser_eos_terminates():
    text = "Hello" + EOS_TOKEN + "ignored"
    p = IncrementalParser(thinking_mode="chat")
    out = p.feed(text)
    cc = "".join(d.get("content", "") for d in out)
    assert cc == "Hello"
    assert p.finished
    assert p.finish_reason == "stop"


# ============================================================
# parse_full_completion
# ============================================================

def test_parse_full_completion_simple():
    content, reasoning, tools, fr = parse_full_completion("hello world", thinking_mode="chat")
    assert content == "hello world"
    assert reasoning == ""
    assert tools == []


def test_parse_full_completion_thinking():
    text = THINK_START + "step1: think" + THINK_END + "final"
    content, reasoning, tools, fr = parse_full_completion(text, thinking_mode="chat")
    assert reasoning == "step1: think"
    assert content == "final"


def test_parse_full_completion_with_tool_calls():
    text = (
        "Sure.\n\n"
        f"<{DSML_TOKEN}tool_calls>\n"
        f"<{DSML_TOKEN}invoke name=\"f\">\n"
        f"<{DSML_TOKEN}parameter name=\"x\" string=\"false\">42</{DSML_TOKEN}parameter>\n"
        f"</{DSML_TOKEN}invoke>\n"
        f"</{DSML_TOKEN}tool_calls>"
    )
    content, reasoning, tools, fr = parse_full_completion(text, thinking_mode="chat")
    assert "Sure." in content
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "f"
    assert fr == "tool_calls"


# ============================================================
# CompletionRequest
# ============================================================

def test_completion_request():
    req = CompletionRequest(model="x", prompt="hi")
    assert req.prompt == "hi"
    assert req.get_stop_list() == []
