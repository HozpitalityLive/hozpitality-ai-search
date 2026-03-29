#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B - Stable)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

pkill -f vllm || true
sleep 2

# Prevent fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting server..."

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b-it \
  --dtype=half \
  --gpu-memory-utilization 0.6 \
  --max-model-len 1024 \
  --max-num-seqs 8 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

lsof -i :$PORT || echo "Port not open yet"

echo "================================="
echo "vLLM running on port $PORT"
echo "================================="