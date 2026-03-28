#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B)"
echo "================================="

PORT=8000

echo "Stopping old vLLM..."

# safer kill (only exact process)
pkill -f "vllm.entrypoints.openai.api_server" || true

sleep 2

echo "Starting vLLM..."

nohup python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 1024 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

echo "Checking server..."
lsof -i :$PORT

echo ""
echo "vLLM started on port $PORT"
echo "Logs: tail -f vllm.log"
echo "================================="