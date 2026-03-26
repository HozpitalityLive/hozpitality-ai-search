#!/bin/bash

echo "Restarting AI server..."

# Stop existing server
echo "Stopping existing server..."
pkill -f "uvicorn ai_server:app"

sleep 2

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Start server
echo "Starting AI server..."

nohup env/bin/uvicorn main:main_app \
--host 0.0.0.0 \
--port 80 \
--access-log \
--log-level info \
> ai_server.log 2>&1 &

sleep 2

echo "AI server restarted successfully!"
echo "Logs: tail -f ai_server.log"