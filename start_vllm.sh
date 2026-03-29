#!/bin/bash

echo "================================="
echo "Starting vLLM (Phi-3-mini)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

pkill -f vllm || true
sleep 2

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model microsoft/Phi-3-mini-4k-instruct \
  --dtype=half \
  --gpu-memory-utilization 0.6 \
  --max-model-len 2048 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

lsof -i :$PORT

echo "================================="
echo "vLLM running on port $PORT"
echo "================================="