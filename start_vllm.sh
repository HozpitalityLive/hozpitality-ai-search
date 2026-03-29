#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B - SAFE)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

pkill -f vllm || true
sleep 2

# Kill any zombie GPU processes
fuser -k 8000/tcp || true

# Prevent fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting server..."

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b-it \
  --dtype=half \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --max-num-seqs 4 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

lsof -i :$PORT || echo "Port not open yet"

echo "================================="
echo "vLLM running on port $PORT"
echo "================================="