#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B FINAL)"
echo "================================="

PORT=8000
PYTHON="/home/dev/hozpitality-ai-search/env/bin/python"

pkill -f vllm || true
sleep 2

# Critical fix
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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