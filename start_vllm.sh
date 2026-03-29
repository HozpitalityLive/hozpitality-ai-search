#!/bin/bash

echo "================================="
echo "Starting vLLM (Mistral 7B)"
echo "================================="

PORT=8000
VENV_PATH="/home/dev/hozpitality-ai-search/env"
PYTHON="$VENV_PATH/bin/python"

echo "Stopping old vLLM..."
pkill -f "vllm.entrypoints.openai.api_server" || true

sleep 2

echo "Verifying environment..."

# Ensure correct Python is used
echo "Python path: $PYTHON"
$PYTHON -c "import sys; print('Executable:', sys.executable)"

# Verify pyairports is actually importable
$PYTHON -c "import pyairports; print('pyairports OK')" || {
  echo "ERROR: pyairports not found in this environment"
  exit 1
}

echo "Starting vLLM..."

nohup $PYTHON -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --gpu-memory-utilization 0.7 \
  --port $PORT \
  > vllm.log 2>&1 &

sleep 5

echo "Checking server..."
lsof -i :$PORT || echo "Port not open yet"

echo ""
echo "Process check:"
ps aux | grep vllm | grep -v grep

echo ""
echo "vLLM started on port $PORT"
echo "Logs: tail -f vllm.log"
echo "================================="