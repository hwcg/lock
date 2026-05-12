# 推理与部署

## 三种推理引擎

### 1. LocalEngine（最简单）
```python
from deepseek_v4.evaluation.engine import LocalEngine
engine = LocalEngine(
    model_path="checkpoints/sft/checkpoint-final",
    tokenizer_path="checkpoints/tokenizer",
    device="cuda", dtype=torch.bfloat16,
)
print(engine.generate(["Hello!"], max_new_tokens=64)[0])
```

### 2. vLLM（最高吞吐）
```bash
make export-vllm
python -m vllm.entrypoints.openai.api_server \
    --model exports/vllm --trust-remote-code --dtype bfloat16
```

### 3. llama.cpp / Ollama（端侧）
```bash
make export-gguf
make export-ollama
ollama run deepseek-v4-mini "Hello"
```

## OpenAI 兼容服务

```bash
make serve
# 默认 http://localhost:8000/v1
```

`curl /v1/chat/completions`：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-mini",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    "thinking_mode": "thinking"
  }'
```

## YaRN 长上下文

启动时一行配置切到 128K：

```bash
python scripts/serve.py --config configs/inference/server.yaml \
    yarn_factor=16 max_model_len=131072
```

或在 LocalEngine 中：

```python
engine = LocalEngine(..., yarn_factor=16, max_seq_len=131072)
```

完成 Needle 测试：

```bash
make needle-long
```

## 性能调优

| 选项 | 影响 | 推荐 |
|---|---|---|
| `dtype` | 速度 / 精度 | A100 / H100 用 `bfloat16`，老 GPU 用 `float16` |
| `max_batch_size` | 吞吐 | 8 ~ 16 |
| `batch_wait_ms` | 延迟 vs 吞吐 | 8 ms（默认）|
| `gradient_checkpointing` | 训练显存 | 训练 ON / 推理 OFF |

## 故障排查

| 问题 | 解决 |
|---|---|
| OOM | 减 `max_model_len` / `max_batch_size`；上 ZeRO-3 |
| Throughput 低 | 检查 dtype / batching / 是否被 IO 阻塞 |
| 生成乱码 | 检查 tokenizer special token 是否对齐 |
| 流式中断 | 客户端 timeout 设 ≥ 300s |
