#!/bin/bash

echo "Loading env..."
source .env

PORT=8000
CONTAINER_NAME="vllm-phi"

echo "Cleaning port..."
sudo fuser -k $PORT/tcp 2>/dev/null || true

echo "Stopping old container..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "Starting new container..."

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  -p $PORT:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  --ipc=host \
  vllm/vllm-openai:latest \
  microsoft/Phi-3-mini-4k-instruct \
  --dtype half \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 5