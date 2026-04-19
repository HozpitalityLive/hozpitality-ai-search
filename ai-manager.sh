#!/bin/bash

# ===============================
# AI SEARCH SYSTEM MANAGER (MENU)
# ===============================

while true; do

echo ""
echo "========================================"
echo "🚀 AI SEARCH SYSTEM MANAGER"
echo "========================================"
echo "1) 🔨 Start (Build + Run)"
echo "2) 🔄 Restart"
echo "3) 🔥 Rebuild (No Cache)"
echo "4) 🛑 Stop"
echo "5) 📜 Logs"
echo "6) 🧪 Check GPU"
echo "7) 📊 Status"
echo "8) 🧠 Ollama Models"
echo "9) 🧹 Clean Reset ⚠️"
echo "0) ❌ Exit"
echo "========================================"

read -p "👉 Enter your choice: " choice

case $choice in

  1)
    echo "🔨 Starting system..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
    echo "⏳ Waiting..."
    sleep 5
    docker exec -it ai-ollama ollama create llama3-hoz -f /app/Modelfile.llama || true
    docker exec -it ai-ollama ollama create phi3-hoz -f /app/Modelfile.phi3 || true
    echo "✅ Started!"
    ;;

  2)
    echo "🔄 Restarting..."
    docker-compose down
    docker-compose up -d
    echo "✅ Restarted!"
    ;;

  3)
    echo "🔥 Full rebuild..."
    docker-compose down
    docker system prune -f
    docker-compose build --no-cache
    docker-compose up -d
    echo "✅ Rebuilt!"
    ;;

  4)
    echo "🛑 Stopping..."
    docker-compose down
    echo "✅ Stopped!"
    ;;

  5)
    echo "📜 Logs (Ctrl+C to exit)..."
    docker logs -f ai-search-api
    ;;

  6)
    echo "🧪 Checking GPU..."
    docker exec -it ai-search-api nvidia-smi
    ;;

  7)
    echo "📊 Status:"
    docker ps
    ;;

  8)
    echo "🧠 Ollama Models:"
    docker exec -it ai-ollama ollama list
    ;;

  9)
    echo "⚠️ WARNING: This will delete EVERYTHING!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
      docker-compose down -v
      docker system prune -af
      docker volume prune -f
      echo "✅ Clean reset done!"
    else
      echo "❌ Cancelled"
    fi
    ;;

  0)
    echo "👋 Exiting..."
    exit 0
    ;;

  *)
    echo "❌ Invalid option"
    ;;

esac

done