"""
通过 openai-python 调用本服务。

启动服务后：
    python scripts/serve.py --config configs/inference/server.yaml

运行：
    OPENAI_BASE_URL=http://localhost:8000/v1 \
    OPENAI_API_KEY=EMPTY \
    python examples/client/openai_chat.py
"""
import os
import openai

client = openai.OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
)

# 1. 非流式
resp = client.chat.completions.create(
    model="deepseek-v4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "用一句话介绍 DeepSeek-V4."},
    ],
    temperature=0.7,
    max_tokens=128,
)
print("=== Non-stream ===")
print(resp.choices[0].message.content)
if getattr(resp.choices[0].message, "reasoning_content", None):
    print(f"\n[reasoning] {resp.choices[0].message.reasoning_content}")

# 2. 流式 + 思考模式
print("\n=== Streaming (thinking mode) ===")
stream = client.chat.completions.create(
    model="deepseek-v4-mini",
    messages=[{"role": "user", "content": "13 * 27 = ?"}],
    stream=True,
    extra_body={"thinking_mode": "thinking"},
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(f"[think] {delta.reasoning_content}", end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
print()

# 3. 工具调用
print("\n=== Tool calling ===")
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]
resp = client.chat.completions.create(
    model="deepseek-v4-mini",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    temperature=0.1,
)
msg = resp.choices[0].message
print("content:", msg.content)
print("tool_calls:", msg.tool_calls)
