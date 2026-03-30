#!/bin/bash

echo "Loading env..."
source .env

PORT=8000
CONTAINER_NAME="vllm-mistral"

echo "Stopping old container..."

docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -p $PORT:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  --ipc=host \
  vllm/vllm-openai:latest \
  mistralai/Mistral-7B-Instruct-v0.2 \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 10