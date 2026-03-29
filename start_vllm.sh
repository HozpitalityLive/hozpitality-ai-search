#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

pkill -f vllm || true
sleep 2

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b-it \
  --dtype=half \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

lsof -i :$PORT

echo "================================="
echo "vLLM running on port $PORT"
echo "================================="