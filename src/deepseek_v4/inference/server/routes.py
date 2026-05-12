"""
API 路由：
- POST /v1/chat/completions
- POST /v1/completions
- GET  /v1/models
- GET  /health
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from deepseek_v4.inference.server.config import ServerConfig
from deepseek_v4.inference.server.engine import GenerationRequest, ServerEngine
from deepseek_v4.inference.server.parsing import (
    IncrementalParser, parse_full_completion,
)
from deepseek_v4.inference.server.protocol import (
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamChoice, ChatCompletionStreamResponse, ChatMessage,
    CompletionChoice, CompletionRequest, CompletionResponse, ErrorObject,
    ErrorResponse, FunctionCall, ModelInfo, ModelList, ToolCall, UsageInfo,
)
from deepseek_v4.tokenizer.encoding import encode_messages
from deepseek_v4.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 鉴权
# ============================================================

def _verify_api_key(
    request: Request,
    config: ServerConfig,
) -> None:
    if config.api_key is None and not config.api_keys_file:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing api key")
    key = auth[7:].strip()

    valid_keys = set()
    if config.api_key:
        valid_keys.add(config.api_key)
    if config.api_keys_file:
        try:
            with open(config.api_keys_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        valid_keys.add(line)
        except FileNotFoundError:
            pass
    if key not in valid_keys:
        raise HTTPException(status_code=401, detail="invalid api key")


# ============================================================
# Router 工厂
# ============================================================

def build_router(engine: ServerEngine, config: ServerConfig) -> APIRouter:
    router = APIRouter()

    # ---- Auth dependency ----
    async def require_auth(request: Request):
        _verify_api_key(request, config)

    # ============================================================
    # GET /v1/models
    # ============================================================
    if config.enable_models:
        @router.get("/v1/models", dependencies=[Depends(require_auth)])
        async def list_models() -> ModelList:
            items = [ModelInfo(id=config.model_name)]
            for alias in config.model_aliases:
                items.append(ModelInfo(id=alias))
            return ModelList(data=items)

    # ============================================================
    # GET /health
    # ============================================================
    if config.enable_health:
        @router.get("/health")
        async def health():
            return {"status": "ok", "model": config.model_name}

    # ============================================================
    # POST /v1/chat/completions
    # ============================================================

    if config.enable_chat_completions:
        @router.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
        async def chat_completions(req: ChatCompletionRequest, request: Request):
            if config.log_request_payload:
                logger.info(f"[chat] payload: {req.model_dump_json()[:500] if hasattr(req, 'model_dump_json') else req.json()[:500]}")
            try:
                return await _handle_chat(req, engine, config, request)
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"[chat] failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(error=ErrorObject(message=str(e), type="server_error")).model_dump()
                    if hasattr(ErrorResponse(error=ErrorObject(message="x")), "model_dump")
                    else {"error": {"message": str(e), "type": "server_error"}},
                )

    # ============================================================
    # POST /v1/completions
    # ============================================================

    if config.enable_completions:
        @router.post("/v1/completions", dependencies=[Depends(require_auth)])
        async def completions(req: CompletionRequest, request: Request):
            try:
                return await _handle_completion(req, engine, config, request)
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"[cmpl] failed: {e}")
                return JSONResponse(status_code=500, content={"error": {"message": str(e)}})

    return router


# ============================================================
# Chat handler
# ============================================================

async def _handle_chat(
    req: ChatCompletionRequest,
    engine: ServerEngine,
    config: ServerConfig,
    request: Request,
):
    tokenizer = engine.tokenizer
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id

    # ---- thinking mode 解析 ----
    thinking_mode = req.resolve_thinking_mode(default=config.default_thinking_mode)
    if thinking_mode not in ("chat", "thinking"):
        thinking_mode = "chat"

    # ---- 把 messages + tools 渲染为 chat template ----
    messages = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in req.messages]
    # tools 挂到 system
    if req.tools:
        tools_list = [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in req.tools]
        injected = False
        for m in messages:
            if m.get("role") == "system":
                m["tools"] = tools_list
                injected = True
                break
        if not injected:
            messages.insert(0, {"role": "system", "content": "", "tools": tools_list})

    try:
        prompt_text = encode_messages(
            messages=messages,
            thinking_mode=thinking_mode,
            drop_thinking=config.drop_thinking_in_context,
            add_default_bos_token=True,
            reasoning_effort=req.reasoning_effort,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"chat template failed: {e}")

    prompt_ids = tokenizer.encode(prompt_text)
    # 截断保留尾部
    max_in = config.max_model_len - 16
    if len(prompt_ids) > max_in:
        prompt_ids = prompt_ids[-max_in:]

    max_tokens = req.get_max_tokens(config.default_max_tokens)
    max_tokens = min(max_tokens, config.max_model_len - len(prompt_ids) - 1)
    if max_tokens <= 0:
        raise HTTPException(status_code=400, detail="prompt too long")

    gen_req = GenerationRequest(
        request_id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
        prompt_ids=prompt_ids,
        max_new_tokens=max_tokens,
        temperature=(req.temperature if req.temperature is not None else config.default_temperature),
        top_p=(req.top_p if req.top_p is not None else config.default_top_p),
        top_k=(req.top_k if req.top_k is not None else config.default_top_k),
        repetition_penalty=(req.repetition_penalty if req.repetition_penalty is not None else config.default_repetition_penalty),
        stop_strings=req.get_stop_list(),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        stream=bool(req.stream),
    )
    gen_req.output_queue = asyncio.Queue()
    await engine.submit(gen_req)

    if req.stream:
        return EventSourceResponse(
            _chat_stream(gen_req, req, thinking_mode, config),
            ping=20,
        )
    else:
        return await _chat_nonstream(gen_req, req, thinking_mode, config)


async def _chat_nonstream(
    gen_req: GenerationRequest,
    req: ChatCompletionRequest,
    thinking_mode: str,
    config: ServerConfig,
) -> ChatCompletionResponse:
    full_text = ""
    finish_reason = "stop"
    while True:
        msg = await gen_req.output_queue.get()
        if msg["type"] == "delta":
            full_text += msg["text"]
        elif msg["type"] == "finish":
            finish_reason = msg["reason"]
            break
        elif msg["type"] == "error":
            raise HTTPException(status_code=500, detail=msg["message"])

    if config.enable_tool_calls:
        content, reasoning, tool_calls, fr = parse_full_completion(full_text, thinking_mode=thinking_mode)
        if tool_calls:
            finish_reason = "tool_calls"
        elif fr == "stop":
            finish_reason = "stop"
    else:
        content = full_text
        reasoning = ""
        tool_calls = []

    msg = ChatMessage(
        role="assistant",
        content=content if content else None,
        reasoning_content=reasoning if reasoning else None,
        tool_calls=[
            ToolCall(
                id=tc["id"],
                function=FunctionCall(name=tc["function"]["name"], arguments=tc["function"]["arguments"]),
            )
            for tc in tool_calls
        ] or None,
    )

    return ChatCompletionResponse(
        model=req.model,
        choices=[ChatCompletionChoice(index=0, message=msg, finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=len(gen_req.prompt_ids),
            completion_tokens=len(full_text),     # 近似值（按字符）
            total_tokens=len(gen_req.prompt_ids) + len(full_text),
        ),
    )


async def _chat_stream(
    gen_req: GenerationRequest,
    req: ChatCompletionRequest,
    thinking_mode: str,
    config: ServerConfig,
) -> AsyncIterator[Dict[str, str]]:
    """SSE 流。"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    parser = IncrementalParser(thinking_mode=thinking_mode)

    # 首帧：role
    head = ChatCompletionStreamResponse(
        id=chunk_id, model=req.model,
        choices=[ChatCompletionStreamChoice(
            index=0, delta={"role": "assistant"}, finish_reason=None,
        )],
    )
    yield {"event": "message", "data": _to_json(head)}

    finish_reason = None
    completion_chars = 0
    try:
        while True:
            msg = await gen_req.output_queue.get()
            if msg["type"] == "delta":
                text = msg["text"]
                completion_chars += len(text)
                if config.enable_tool_calls:
                    deltas = parser.feed(text)
                else:
                    deltas = [{"content": text}]
                for d in deltas:
                    chunk = ChatCompletionStreamResponse(
                        id=chunk_id, model=req.model,
                        choices=[ChatCompletionStreamChoice(index=0, delta=d, finish_reason=None)],
                    )
                    yield {"event": "message", "data": _to_json(chunk)}
            elif msg["type"] == "finish":
                finish_reason = msg["reason"]
                # flush
                if config.enable_tool_calls:
                    for d in parser.finalize():
                        chunk = ChatCompletionStreamResponse(
                            id=chunk_id, model=req.model,
                            choices=[ChatCompletionStreamChoice(index=0, delta=d, finish_reason=None)],
                        )
                        yield {"event": "message", "data": _to_json(chunk)}
                if parser.emitted_tool_calls > 0:
                    finish_reason = "tool_calls"
                break
            elif msg["type"] == "error":
                err = {"error": {"message": msg["message"], "type": "server_error"}}
                yield {"event": "message", "data": json.dumps(err)}
                return
    except asyncio.CancelledError:
        gen_req.cancelled = True
        raise

    # 终止帧
    end_chunk = ChatCompletionStreamResponse(
        id=chunk_id, model=req.model,
        choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason=finish_reason or "stop")],
    )
    if req.stream_options and req.stream_options.include_usage:
        end_chunk.usage = UsageInfo(
            prompt_tokens=len(gen_req.prompt_ids),
            completion_tokens=completion_chars,
            total_tokens=len(gen_req.prompt_ids) + completion_chars,
        )
    yield {"event": "message", "data": _to_json(end_chunk)}

    # OpenAI 标准 [DONE]
    yield {"event": "message", "data": "[DONE]"}


# ============================================================
# Completion handler
# ============================================================

async def _handle_completion(
    req: CompletionRequest,
    engine: ServerEngine,
    config: ServerConfig,
    request: Request,
):
    tokenizer = engine.tokenizer
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    if len(prompts) > 1:
        raise HTTPException(status_code=400, detail="batch prompt not supported in legacy completions")

    prompt_ids = tokenizer.encode(prompts[0])
    max_in = config.max_model_len - 16
    if len(prompt_ids) > max_in:
        prompt_ids = prompt_ids[-max_in:]

    max_tokens = req.max_tokens or 16
    gen_req = GenerationRequest(
        request_id=f"cmpl-{uuid.uuid4().hex[:16]}",
        prompt_ids=prompt_ids,
        max_new_tokens=max_tokens,
        temperature=(req.temperature if req.temperature is not None else config.default_temperature),
        top_p=(req.top_p if req.top_p is not None else config.default_top_p),
        top_k=(req.top_k if req.top_k is not None else config.default_top_k),
        repetition_penalty=(req.repetition_penalty if req.repetition_penalty is not None else config.default_repetition_penalty),
        stop_strings=req.get_stop_list(),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        stream=bool(req.stream),
    )
    gen_req.output_queue = asyncio.Queue()
    await engine.submit(gen_req)

    if req.stream:
        return EventSourceResponse(
            _completion_stream(gen_req, req),
            ping=20,
        )
    full_text = ""
    finish_reason = "stop"
    while True:
        msg = await gen_req.output_queue.get()
        if msg["type"] == "delta":
            full_text += msg["text"]
        elif msg["type"] == "finish":
            finish_reason = msg["reason"]
            break
        elif msg["type"] == "error":
            raise HTTPException(status_code=500, detail=msg["message"])

    if req.echo:
        full_text = prompts[0] + full_text

    return CompletionResponse(
        model=req.model,
        choices=[CompletionChoice(index=0, text=full_text, finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(full_text),
            total_tokens=len(prompt_ids) + len(full_text),
        ),
    )


async def _completion_stream(gen_req: GenerationRequest, req: CompletionRequest):
    chunk_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    while True:
        msg = await gen_req.output_queue.get()
        if msg["type"] == "delta":
            payload = {
                "id": chunk_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": msg["text"], "finish_reason": None}],
            }
            yield {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}
        elif msg["type"] == "finish":
            payload = {
                "id": chunk_id, "object": "text_completion", "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "text": "", "finish_reason": msg["reason"]}],
            }
            yield {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}
            break
        elif msg["type"] == "error":
            yield {"event": "message", "data": json.dumps({"error": {"message": msg["message"]}})}
            return
    yield {"event": "message", "data": "[DONE]"}


# ============================================================
# 工具
# ============================================================

def _to_json(obj) -> str:
    """Pydantic v1/v2 兼容序列化。"""
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(exclude_none=True)
    return obj.json(exclude_none=True)
