"""
端到端服务端集成测试。

使用 mock engine 避免加载真实模型。
"""
from __future__ import annotations

import asyncio
import json
from typing import List

import pytest
from fastapi.testclient import TestClient

from deepseek_v4.inference.server.app import build_app
from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.inference.server.engine import GenerationRequest, ServerEngine


# ============================================================
# Mock Engine
# ============================================================

class MockEngine(ServerEngine):
    """不真正加载模型的 mock，按字符 yield 输出。"""
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.loop = None
        self._stop = False

    def load(self): pass
    def start(self):
        import asyncio
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
    def stop(self, timeout=None): pass

    async def submit(self, req: GenerationRequest):
        if req.output_queue is None:
            req.output_queue = asyncio.Queue()
        # 立即写出 mock 输出
        for ch in "Hello world.":
            await req.output_queue.put({"type": "delta", "text": ch})
        await req.output_queue.put({"type": "finish", "reason": "stop"})


class MockTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

    def encode(self, text):
        return [ord(c) % 1000 for c in text]


# ============================================================
# Test fixtures
# ============================================================

@pytest.fixture
def client():
    cfg = ServerConfig(
        model_path="dummy", tokenizer_path="dummy",
        model_name="mock-model",
        max_model_len=1024,
        default_thinking_mode="chat",
    )
    engine = MockEngine()
    app = build_app(cfg, engine=engine)
    return TestClient(app)


# ============================================================
# 测试
# ============================================================

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["id"] == "mock-model" for m in data["data"])


def test_chat_nonstream(client):
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    # mock 输出包含 "Hello world."
    assert "Hello" in data["choices"][0]["message"]["content"]


def test_chat_stream(client):
    """流式：应至少包含 role 首帧 + 一个 [DONE]。"""
    with client.stream(
        "POST", "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        lines: List[str] = []
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                lines.append(payload)
        assert "[DONE]" in lines
        # 解析首帧
        first = json.loads(lines[0])
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"


def test_legacy_completion(client):
    r = client.post(
        "/v1/completions",
        json={"model": "mock-model", "prompt": "Hello"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "text_completion"


def test_auth_required(client):
    """启用 api_key 后无 token 应 401。"""
    cfg = ServerConfig(
        model_path="dummy", tokenizer_path="dummy",
        api_key="secret",
    )
    engine = MockEngine()
    app = build_app(cfg, engine=engine)
    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model": "x", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 401

    # 正确 key 通过
    r2 = c.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r2.status_code == 200


def test_invalid_model_ignored(client):
    """model 名不匹配也能跑（OpenAI 标准并不强校验）。"""
    r = client.post(
        "/v1/chat/completions",
        json={"model": "random-name", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
