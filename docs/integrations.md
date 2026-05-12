# 第三方 Chat UI 接入指南

DeepSeek-V4-Mini 服务端兼容 OpenAI Chat API，可直接接入主流前端。

## 通用配置

启动服务：

```bash
python scripts/serve.py --config configs/inference/server.yaml \
    api_key=mysecret
```

服务端点：

- Base URL: `http://<host>:<port>/v1`
- API Key: `mysecret`（或 `EMPTY` 禁用鉴权）
- Model: `deepseek-v4-mini`

## 1. Open-WebUI

1. 启动 Open-WebUI（docker）：

   ```
   docker run -d -p 3000:8080 \
       -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
       -e OPENAI_API_KEY=mysecret \
       --name open-webui ghcr.io/open-webui/open-webui:main
   ```

2. 浏览器打开 `http://localhost:3000`，进入设置 → 模型，应自动看到 `deepseek-v4-mini`。

启用「思考模式」（reasoning_content）：在 Open-WebUI 中编辑 model preset，加入：

```
{"thinking_mode": "thinking"}
```

## 2. FastGPT

1. 在 FastGPT 后台 → 渠道（OneAPI / 直连）：
   - Base URL: `http://<server>:8000/v1`
   - API Key: `mysecret`
2. 模型表加入：`deepseek-v4-mini` → 上下文 4096
3. 测试对话即可

## 3. LobeChat / ChatGPT-Next-Web 等

任一兼容 OpenAI 的前端，填入相同的 Base URL + API Key + Model 即可。

## 高级功能

- **思考模式**：请求 body 加 `"thinking_mode": "thinking"`
- **reasoning_effort**：加 `"reasoning_effort": "max"`
- **工具调用**：与 OpenAI tools/tool_calls 完全一致
- **流式 + usage**：`"stream_options": {"include_usage": true}`
- **YaRN 长上下文**：服务端启动时加 `yarn_factor=16`

## 排错

| 现象                   | 原因                                                  |
| ---------------------- | ----------------------------------------------------- |
| 503 / 401              | 检查 api_key 与 Authorization header                  |
| 流式断流               | 客户端 timeout 太短，建议 ≥ 300s                      |
| reasoning_content 缺失 | `thinking_mode != thinking` 或客户端没读 SSE 扩展字段 |
| tool_calls 解析失败    | 服务端 `enable_tool_calls=false`                      |
