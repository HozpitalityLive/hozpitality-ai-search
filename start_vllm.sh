#!/bin/bash

echo "================================="
echo "Starting vLLM (Gemma 2B - Docker)"
echo "================================="

PORT=8000
CONTAINER_NAME="vllm-gemma"
HF_TOKEN="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"

echo "Stopping old container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

sleep 2

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -p $PORT:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  --ipc=host \
  vllm/vllm-openai:latest \
  google/gemma-2b-it \
  --dtype half \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 3

sleep 5

echo "Checking server..."
curl -s http://localhost:$PORT/v1/models || echo "Server not ready yet"

echo ""
echo "Container status:"
docker ps | grep $CONTAINER_NAME

echo ""
echo "Logs:"
echo "docker logs -f $CONTAINER_NAME"

echo "================================="
echo "vLLM running on port $PORT"
echo "================================="