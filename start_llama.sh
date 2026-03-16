#!/bin/bash

echo "================================="
echo "Starting LLaMA Server"
echo "================================="

LLAMA_DIR="$HOME/llama.cpp"
MODEL_PATH="$HOME/models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
PORT=8080

echo "LLAMA DIR: $LLAMA_DIR"
echo "MODEL: $MODEL_PATH"
echo "PORT: $PORT"

echo "---------------------------------"
echo "Checking model..."
echo "---------------------------------"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

echo "Model found."

echo "---------------------------------"
echo "Building llama.cpp (if needed)..."
echo "---------------------------------"

cd $LLAMA_DIR

if [ ! -f "$LLAMA_DIR/build/bin/llama-server" ]; then
  echo "Compiling llama.cpp..."

  cmake -B build
  cmake --build build --config Release

else
  echo "Binary already exists. Skipping build."
fi

echo "---------------------------------"
echo "Stopping old server (if running)..."
echo "---------------------------------"

pkill -f llama-server || true

echo "---------------------------------"
echo "Starting LLM server..."
echo "---------------------------------"

nohup $LLAMA_DIR/build/bin/llama-server \
-m $MODEL_PATH \
--host 0.0.0.0 \
--port $PORT \
--ctx-size 4096 \
--batch-size 512 \
-ngl 100 \
> $LLAMA_DIR/llm.log 2>&1 &

sleep 2

echo "---------------------------------"
echo "Checking server status..."
echo "---------------------------------"

lsof -i :$PORT

echo ""
echo "LLaMA server started."
echo "Logs: $LLAMA_DIR/llm.log"
echo "API: http://127.0.0.1:$PORT"
echo "================================="