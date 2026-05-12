# 训练指南

## 完整 Pipeline

```
0. 数据准备
   make data-pretrain         # 下载 + 清洗 + 去重
   make data-sft
   make data-dpo

1. Tokenizer
   make tokenizer

2. Pretrain
   make pretrain              # 单卡
   N_GPUS=8 make pretrain-multi
   make pretrain-deepspeed    # ZeRO-2

3. SFT
   make sft

4. (可选) LoRA
   make lora
   make lora-merge

5. Reward Model
   make rm

6. RLHF
   make dpo                   # 或 ppo / grpo / cispo

7. Tool Use SFT
   make tool-use

8. (可选) Agentic RL
   make agentic-rl

9. (可选) 蒸馏
   make distill

10. 评测
    make eval-all

11. 部署
    make serve
```

## 数据准备

每个阶段对应 `data/processed/<stage>.jsonl`，统一字段：

| Stage | 字段 |
|---|---|
| pretrain | `text` |
| sft | `messages` (OpenAI 格式) + 可选 `tools` |
| dpo  | `prompt` / `chosen` / `rejected`，或 `chosen`/`rejected` 两份完整 messages |
| tool | `messages` + `tools` |
| thinking | `messages_user` + `assistant_chat` + `assistant_thinking` |

## 配置覆盖

所有 YAML 字段都可命令行覆盖：

```bash
python scripts/sft.py --config configs/training/sft.yaml \
    learning_rate=1e-5 \
    micro_batch_size=4 \
    "logger_backends=[jsonl,wandb]"
```

## 分布式

```bash
# 单机 8 卡 DDP
torchrun --standalone --nproc_per_node 8 scripts/sft.py --config configs/training/sft.yaml

# DeepSpeed ZeRO-3 + offload
torchrun --standalone --nproc_per_node 8 scripts/pretrain.py \
    --config configs/training/pretrain.yaml \
    use_deepspeed=true \
    deepspeed_config=configs/deepspeed/zero3_offload.json
```

## 动态启停

```bash
# 暂停（worker 完成当前 step 后阻塞）
kill -USR1 <pid>

# 触发"保存并退出"
kill -USR2 <pid>
```

## 监控

```bash
# wandb
WANDB_API_KEY=xxx make pretrain

# swanlab
SWANLAB_MODE=cloud make pretrain
```

JSONL 后端总是开启，写到 `<output_dir>/tracker/<project>/<run>.jsonl`。

## 选 Optimizer

| 场景 | 推荐 |
|---|---|
| Pretrain | adamw (lr 3e-4, β=(0.9, 0.95)) |
| Pretrain (实验) | muon (lr 0.4 × adamw_lr，仅 2D 矩阵) |
| SFT  | adamw (lr 2e-5) |
| LoRA | adamw (lr 1e-4) |
| DPO  | adamw (lr 5e-7) |
| PPO/GRPO | adamw (lr 1e-6) |

## 调试技巧

**梯度爆炸 / NaN**
- 把 `precision: bf16` 改 `fp32`
- 减小 `learning_rate`
- 增大 `max_grad_norm`
- 关闭 `gradient_checkpointing` 看是否复现

**收敛慢**
- `aux_loss_weight` 调低（0.001 ~ 0.01）
- 检查 token throughput / MFU 是否合理
- 看是不是 IO 瓶颈：`time torch.utils.data.DataLoader`

**Loss 居高不下**
- 检查 SFT 数据 loss mask：`labels == -100` 比例应 < 95%
- 检查 chat template 渲染是否正确
