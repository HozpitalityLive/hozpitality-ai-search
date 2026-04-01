#!/bin/bash

echo "================================="
echo "Restarting AI server..."
echo "================================="

# Stop existing uvicorn
echo "Stopping existing server..."
pkill -f "uvicorn main:main_app" || true

sleep 2

# Activate virtual environment
echo "Activating virtual environment..."
source /home/dev/hozpitality-ai-search/env/bin/activate

# Verify uvicorn exists
which uvicorn

echo "Starting AI server on port 9000..."

nohup uvicorn main:main_app \
  --host 0.0.0.0 \
  --port 9000 \
  --workers 2 \
  --access-log \
  --log-level info \
  > ai_server.log 2>&1 &

sleep 3

echo ""
echo "Process check:"
ps aux | grep uvicorn | grep -v grep

echo ""
echo "================================="
echo "AI server started!"
echo "Logs: tail -f ai_server.log"
echo "================================="