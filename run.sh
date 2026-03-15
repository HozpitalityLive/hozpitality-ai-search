#!/bin/bash

echo "Starting AI server..."

source env/bin/activate

nohup env/bin/uvicorn ai_server:app \
--host 0.0.0.0 \
--port 8000 \
--access-log \
--log-level info \
> ai_server.log 2>&1 &