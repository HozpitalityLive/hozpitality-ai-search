#!/bin/bash

echo "Loading env..."
source .env

PORT=8000
CONTAINER_NAME="vllm-gemma"

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
  google/gemma-2b-it \
  --dtype half \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 5

sleep 5

curl http://localhost:$PORT/v1/models