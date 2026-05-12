# DeepSeek-V4-Mini 文档

> 工业级 DeepSeek-V4 训练 / 推理 / 部署完整框架

## 目录

- [架构详解](architecture.md) — 模型结构、MLA / mHC / CSA / HCA / MoE 一文讲清
- [训练指南](training_guide.md) — 从 Tokenizer 到 RL 的完整 pipeline
- [推理与部署](inference_guide.md) — 服务端、长上下文、格式转换
- [API 参考](api_reference.md) — OpenAI 兼容 API 完整字段
- [集成第三方 UI](integrations.md) — Open-WebUI / FastGPT / LobeChat
- [故障排查](troubleshooting.md) — 常见问题与解决方案
- [贡献指南](../CONTRIBUTING.md)

## 快速开始（30 秒）

```bash
# 1. 安装
pip install -e ".[train,serve,eval]"

# 2. 训练 tokenizer（5 分钟）
python scripts/train_tokenizer.py --config configs/tokenizer/train_config.yaml

# 3. 跑一个最小 pretrain 验证（10 步）
python scripts/pretrain.py --config configs/training/pretrain.yaml \
    max_steps=10 micro_batch_size=1

# 4. 启动服务
python scripts/serve.py --config configs/inference/server.yaml
```
