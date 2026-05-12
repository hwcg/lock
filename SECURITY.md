# Security Policy

## 报告漏洞

如发现安全漏洞，请通过邮件私下联系 `security@deepseek-v4-mini.example`，**不要**直接在 GitHub Issues 里公开。

## 已知风险

1. **Tool Use 沙盒**：`code_python` reward 与 `python` 工具均通过 `subprocess + RLIMIT` 限制资源，但**不是完整 sandbox**。生产环境强烈建议替换为 firejail / docker / gvisor。
2. **`calculator` 工具**：使用 AST 白名单，但仍建议二次审计。
3. **OpenAI 服务**：`api_key=null` 时无鉴权，**勿暴露公网**。
4. **加载第三方 checkpoint**：`torch.load(..., weights_only=False)` 存在 RCE 风险，请用 `safetensors`。

## 支持的版本

| 版本 | 状态 |
|---|---|
| 0.1.x | ✅ 安全更新 |
