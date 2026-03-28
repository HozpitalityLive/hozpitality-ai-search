#!/bin/bash

echo "Stopping llama.cpp server..."

pkill -f llama-server || true
pkill -f mistral || true
pkill -f gguf || true

sleep 2

echo "Checking ports..."

lsof -i :8080 || true
lsof -i :8000 || true

echo "Done."