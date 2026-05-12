#!/usr/bin/env bash
# 简单 curl 验证

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

echo "==== /health ===="
curl -s "${BASE_URL%/v1}/health" && echo

echo -e "\n==== /v1/models ===="
curl -s "${BASE_URL}/models" && echo

echo -e "\n==== /v1/chat/completions (non-stream) ===="
curl -s -X POST "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4-mini",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 32
    }' | python -m json.tool

echo -e "\n==== /v1/chat/completions (stream) ===="
curl -N -s -X POST "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4-mini",
        "messages": [{"role": "user", "content": "Count to 5."}],
        "max_tokens": 32,
        "stream": true
    }'
echo
