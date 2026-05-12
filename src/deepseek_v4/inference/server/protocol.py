"""
OpenAI Chat Completions / Completions 兼容协议（Pydantic）。

参考：
- https://platform.openai.com/docs/api-reference/chat
- https://platform.openai.com/docs/api-reference/completions

扩展字段（DeepSeek-V4 特有，向后兼容）：
- ChatCompletionRequest:
    open_thinking / enable_thinking / thinking_mode
    reasoning_effort: "max" | "high" | None
    tools / tool_choice
- Message:
    reasoning_content        # 思考链
    tool_calls               # OpenAI 兼容
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, Field, ConfigDict
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseModel, Field
    PYDANTIC_V2 = False
    ConfigDict = None


# ============================================================
# 基础
# ============================================================

def _gen_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepseek"
    permission: List[Dict[str, Any]] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo] = Field(default_factory=list)


# ============================================================
# Messages
# ============================================================

class FunctionCall(BaseModel):
    """OpenAI 函数调用对象。"""
    name: str
    arguments: str          # JSON string


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """单条消息。"""
    role: Literal["system", "user", "assistant", "tool", "developer", "function"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None     # V4 扩展

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"


# ============================================================
# Chat Completion Request
# ============================================================

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[Dict[str, Any]] = None


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    # 标准 OpenAI 字段
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # 新版字段
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    response_format: Optional[ResponseFormat] = None

    # Function/Tool calling
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # === DeepSeek-V4 扩展 ===
    open_thinking: Optional[bool] = None         # 兼容名
    enable_thinking: Optional[bool] = None       # 兼容名
    thinking_mode: Optional[Literal["chat", "thinking", "auto"]] = None
    reasoning_effort: Optional[Literal["max", "high"]] = None

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"

    def resolve_thinking_mode(self, default: str = "auto") -> str:
        """统一三个等价字段为单一 thinking_mode ∈ {chat, thinking}。"""
        if self.thinking_mode and self.thinking_mode != "auto":
            return self.thinking_mode
        if self.open_thinking is True or self.enable_thinking is True:
            return "thinking"
        if self.open_thinking is False or self.enable_thinking is False:
            return "chat"
        # auto / 未指定 → 由 server 决定
        if default == "auto":
            return "chat"
        return default

    def get_max_tokens(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default

    def get_stop_list(self) -> List[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)


# ============================================================
# Chat Completion Response
# ============================================================

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = "stop"
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None


# ============================================================
# Chat Completion Streaming
# ============================================================

class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("chatcmpl"))
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[UsageInfo] = None


# ============================================================
# Legacy completion API
# ============================================================

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None
    echo: Optional[bool] = False

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"

    def get_stop_list(self) -> List[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"
    logprobs: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _gen_id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ============================================================
# 错误响应
# ============================================================

class ErrorObject(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorObject
