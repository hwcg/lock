#!/usr/bin/env bash
# 完整流水线演示（在 tiny 模型上跑通，~5 分钟）

set -euo pipefail

WORKSPACE="./demo_workspace"
mkdir -p "$WORKSPACE"

echo "=== [1/7] Train tokenizer ==="
python scripts/train_tokenizer.py \
    --config configs/tokenizer/train_config.yaml \
    --output_dir "$WORKSPACE/tokenizer" \
    --vocab_size 8000

echo "=== [2/7] Pretrain (10 steps) ==="
python scripts/pretrain.py --config configs/training/pretrain.yaml \
    output_dir="$WORKSPACE/pretrain" \
    tokenizer_path="$WORKSPACE/tokenizer" \
    train_data_paths='["data/processed/pretrain.jsonl"]' \
    max_steps=10 \
    micro_batch_size=1 \
    gradient_accumulation_steps=1

echo "=== [3/7] SFT (10 steps) ==="
python scripts/sft.py --config configs/training/sft.yaml \
    output_dir="$WORKSPACE/sft" \
    tokenizer_path="$WORKSPACE/tokenizer" \
    init_from_checkpoint="$WORKSPACE/pretrain/checkpoint-10" \
    train_data_paths='["data/processed/sft.jsonl"]' \
    max_steps=10

echo "=== [4/7] DPO (5 steps) ==="
python scripts/dpo.py --config configs/training/dpo.yaml \
    output_dir="$WORKSPACE/dpo" \
    tokenizer_path="$WORKSPACE/tokenizer" \
    init_from_checkpoint="$WORKSPACE/sft/checkpoint-10" \
    train_data_paths='["data/processed/dpo.jsonl"]' \
    max_steps=5

echo "=== [5/7] Export to HF ==="
python scripts/convert_model.py to-hf \
    --state_dict_dir "$WORKSPACE/dpo/checkpoint-5" \
    --model_config configs/model/mini_2b.json \
    --tokenizer_dir "$WORKSPACE/tokenizer" \
    --output_dir "$WORKSPACE/exports/hf"

echo "=== [6/7] Quick eval ==="
python scripts/evaluate.py \
    --task gsm8k --shots 0 --max_samples 10 \
    --backend local \
    --model_path "$WORKSPACE/exports/hf" \
    --tokenizer_path "$WORKSPACE/tokenizer" \
    --output_dir "$WORKSPACE/eval"

echo "=== [7/7] Start server (background, 60s) ==="
python scripts/serve.py --config configs/inference/server.yaml \
    model_path="$WORKSPACE/exports/hf" \
    tokenizer_path="$WORKSPACE/tokenizer" &
SERVER_PID=$!
sleep 10

echo "Testing /health ..."
curl -s http://localhost:8000/health
echo
echo "Testing /v1/chat/completions ..."
curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-mini","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}' \
    | python -m json.tool

kill $SERVER_PID
echo "✅ Full pipeline demo complete! Artifacts in $WORKSPACE"
