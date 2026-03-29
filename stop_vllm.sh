#!/bin/bash
docker stop vllm-gemma || true
docker rm vllm-gemma || true
echo "vLLM stopped"