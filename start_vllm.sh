#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

# Kill old processes
pkill -f vllm || true
sleep 2

# Prevent CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting server..."

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b-it \
  --dtype=half \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

echo "Checking server..."
lsof -i :$PORT || echo "Port not open yet"

echo ""
echo "Process check:"
ps aux | grep vllm | grep -v grep

echo ""
echo "================================="
echo "vLLM running on port $PORT"
echo "Logs: tail -f vllm.log"
echo "================================="