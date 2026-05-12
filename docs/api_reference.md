# API 参考

## POST /v1/chat/completions

### 标准 OpenAI 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 模型 ID（默认 `deepseek-v4-mini`） |
| `messages` | array | OpenAI messages 格式 |
| `max_tokens` | int | 最大生成 token |
| `max_completion_tokens` | int | 等价（优先级更高） |
| `temperature` | float | 默认 0.7 |
| `top_p` / `top_k` | float / int | nucleus / top-k 截断 |
| `stop` | string / array | 自定义停止字符串 |
| `stream` | bool | SSE 流式 |
| `stream_options.include_usage` | bool | 末帧返回 usage |
| `presence_penalty` / `frequency_penalty` | float | OpenAI 兼容（暂未实现，忽略） |
| `repetition_penalty` | float | 1.0 = 禁用 |
| `seed` | int | 复现性 |
| `tools` | array | OpenAI tools 列表 |
| `tool_choice` | string / object | (V4 暂不强制约束) |
| `response_format` | object | `{type: "json_object"}` 等 |

### V4 扩展字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `thinking_mode` | `"chat"` / `"thinking"` / `"auto"` | 思考模式 |
| `open_thinking` | bool | 等价 `thinking_mode` |
| `enable_thinking` | bool | 等价 `thinking_mode` |
| `reasoning_effort` | `"max"` / `"high"` / null | 思考强度 |

### 响应字段（assistant message）

| 字段 | 类型 | 说明 |
|---|---|---|
| `role` | `"assistant"` |  |
| `content` | string / null | 最终回答 |
| `reasoning_content` | string / null | **V4 扩展**：思考链 |
| `tool_calls` | array / null | OpenAI 兼容 tool_calls |
| `finish_reason` | `"stop"` / `"length"` / `"tool_calls"` |  |

### 流式 delta

```
data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}
data: {"choices":[{"delta":{"reasoning_content":"Let me think..."}}]}
data: {"choices":[{"delta":{"content":"The answer is 42."}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","type":"function","function":{"name":"calc","arguments":"{}"}}]}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

## POST /v1/completions

Legacy 格式，仅支持单 prompt。

## GET /v1/models

返回 `model_name` + `model_aliases` 列表。

## GET /health

```json
{"status": "ok", "model": "deepseek-v4-mini"}
```

## 鉴权

设置 `api_key` 后，所有请求需带 `Authorization: Bearer <key>`。
也可用 `api_keys_file` 指向一个一行一 key 的文件。
