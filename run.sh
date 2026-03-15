

source env/bin/activate

echo "Starting AI server..."

nohup uvicorn ai_server:app --host 0.0.0.0 --port 80 > ai_server.log 2>&1 &