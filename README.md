# DeepSeek-V4-Mini

> 工业级 DeepSeek-V4 训练框架（含 2B Mini / 671B Full）。

## 核心特性

- **完整模型实现**：纯 PyTorch 实现 V4 所有核心结构（MLA / mHC / CSA / HCA / Lightning Indexer / MoE）。
- **完整训练流程**：Pretrain → SFT → LoRA → RLHF-DPO → RLAIF (PPO/GRPO/CISPO) → Tool Use → Agentic RL → 蒸馏 → 自适应思考。
- **从 0 实现关键算法**：BPE、DPO、PPO、GRPO、CISPO 全部从 0 实现，不依赖第三方算法库。
- **生态兼容**：兼容 transformers / trl / peft / Llama-Factory，可导出 llama.cpp / vLLM / Ollama 格式。
- **分布式训练**：原生支持 DDP、DeepSpeed (ZeRO 1/2/3)，可单机单卡到单机多卡无缝切换。
- **可视化与调度**：wandb / swanlab 双后端，支持动态启停（信号驱动）。
- **完整评测**：C-Eval / C-MMLU / OpenBookQA / HumanEval / GSM8K。
- **长文本支持**：原生 YaRN 实现，训练时即支持 1M 上下文（`max_position_embeddings=1048576`）。
- **OpenAI API 兼容**：极简服务端，支持 `reasoning_content` / `tool_calls` / `open_thinking`，可接入 FastGPT / Open-WebUI。

## 快速开始

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 训练 Tokenizer
python scripts/train_tokenizer.py \
    --config configs/tokenizer/train_config.yaml \
    --output_dir checkpoints/tokenizer

# 3. 预训练 2B Mini
torchrun --nproc_per_node 8 scripts/pretrain.py \
    --config configs/training/pretrain.yaml

# 4. SFT
torchrun --nproc_per_node 8 scripts/sft.py \
    --config configs/training/sft.yaml

# 5. DPO
torchrun --nproc_per_node 8 scripts/dpo.py \
    --config configs/training/dpo.yaml

# 6. 启动 OpenAI API 服务
python scripts/serve.py --config configs/inference/server.yaml
```

## 文档

- [架构详解](https://www.chatopens.ai/c/docs/architecture.md)
- [训练指南](https://www.chatopens.ai/c/docs/training_guide.md)
- [推理与部署](https://www.chatopens.ai/c/docs/inference_guide.md)
- [API 参考](https://www.chatopens.ai/c/docs/api_reference.md)
