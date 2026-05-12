# 故障排查

## 训练

| 现象 | 可能原因 | 解决 |
|---|---|---|
| `CUDA out of memory` | 模型 / batch 太大 | 减 `micro_batch_size`、加 `gradient_accumulation_steps`、上 ZeRO-2/3、开 `gradient_checkpointing` |
| Loss 震荡 / NaN | bf16 数值不稳 / lr 过大 | 改 `precision: fp32`，减小 `learning_rate`，提高 `max_grad_norm`，关 `aux_loss` |
| 数据加载慢 | num_workers 太少 / 磁盘 IO | `num_workers=8 ~ 16`，确保数据在 SSD 上 |
| DDP hang | NCCL 超时 / unused params | `find_unused_parameters=true` 或检查模型是否所有参数都参与 forward |

## 推理

| 现象 | 解决 |
|---|---|
| 生成无尽循环 | `repetition_penalty=1.05`，加合适 `stop` |
| 流式中断 | 客户端 timeout ≥ 300s；nginx `proxy_read_timeout 600s` |
| reasoning_content 缺失 | 客户端没解析 SSE 扩展字段，或没传 `thinking_mode=thinking` |
| tool_calls 解析失败 | DSML 格式必须严格闭合 |

## 部署

| 现象 | 解决 |
|---|---|
| Open-WebUI 不显示模型 | 检查 `/v1/models` 是否能访问；后端 URL 不要有 `/v1` 重复 |
| FastGPT 401 | API Key 不匹配 |
| 容器 OOM | docker `--shm-size=8g` |

## 评测

| 现象 | 解决 |
|---|---|
| `datasets` 下载失败 | `HF_ENDPOINT=https://hf-mirror.com` |
| HumanEval 都 fail | 检查 `extract_function_body` 是否正常；沙盒 timeout 设大 |
| Needle 全 0 | 没启用 YaRN，或 `max_seq_len` 比 context 小 |
