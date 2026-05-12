#!/usr/bin/env bash
# DeepSeek-V4 统一启动脚本
# 用法：bash scripts/launch.sh <mode> <config_path> [overrides...]
# 例：  bash scripts/launch.sh pretrain configs/training/pretrain.yaml
#       bash scripts/launch.sh sft configs/training/sft.yaml learning_rate=1e-5

set -euo pipefail

MODE="${1:?Usage: launch.sh <mode> <config_path> [overrides...]}"
CONFIG="${2:?Missing config path}"
shift 2
OVERRIDES=("$@")

SCRIPT_MAP=(
    "pretrain:scripts/pretrain.py"
    "sft:scripts/sft.py"
    "lora:scripts/lora.py"
    "train_reward_model:scripts/train_reward_model.py"
    "dpo:scripts/dpo.py"
    "ppo:scripts/ppo.py"
    "grpo:scripts/grpo.py"
    "cispo:scripts/cispo.py"
    "tool_use:scripts/tool_use.py"
    "agentic_rl:scripts/agentic_rl.py"
    "distill:scripts/distill.py"
    "adaptive_thinking:scripts/adaptive_thinking.py"
)

SCRIPT=""
for entry in "${SCRIPT_MAP[@]}"; do
    key="${entry%%:*}"
    val="${entry#*:}"
    if [ "$key" = "$MODE" ]; then
        SCRIPT="$val"
        break
    fi
done

if [ -z "$SCRIPT" ]; then
    echo "Unknown mode: $MODE"
    echo "Available: ${SCRIPT_MAP[*]%%:*}"
    exit 1
fi

N_GPUS="${N_GPUS:-1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_PATH="$ROOT/$SCRIPT"

if [ "$N_GPUS" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$N_GPUS" \
        "$SCRIPT_PATH" --config "$ROOT/$CONFIG" "${OVERRIDES[@]}"
else
    python "$SCRIPT_PATH" --config "$ROOT/$CONFIG" "${OVERRIDES[@]}"
fi
