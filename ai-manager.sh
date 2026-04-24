#!/bin/bash

# ===============================
# AI SEARCH SYSTEM MANAGER (SAFE)
# ===============================

PROJECT_NAME="ai-search"

while true; do

echo ""
echo "========================================"
echo "🚀 AI SEARCH SYSTEM MANAGER (SAFE MODE)"
echo "========================================"
echo "1) 🔨 Start (Build + Run)"
echo "2) 🔄 Restart"
echo "3) 🔥 Rebuild (No Cache)"
echo "4) 🛑 Stop"
echo "5) 📜 Logs"
echo "6) 🧪 Check GPU"
echo "7) 📊 Status"
echo "8) 🧠 Ollama Models"
echo "9) 🧹 SAFE RESET (ONLY THIS PROJECT)"
echo "10) 🔄 Reindex ES (manual)"
echo "11) 💀 Full Reset (DELETE ES DATA)"
echo "0) ❌ Exit"
echo "========================================"

read -p "👉 Enter your choice: " choice

case $choice in

  1)
    echo "🔨 Starting system..."
    docker-compose up -d --build
    echo "⏳ Waiting..."
    sleep 5
    echo "✅ Started!"
    ;;

  2)
    echo "⚡ Fast Restart..."
    docker-compose restart
    echo "✅ Restarted instantly"
    ;;

  3)
    echo "🔥 Rebuilding ONLY this project..."
    docker-compose build --no-cache
    docker-compose up -d
    echo "✅ Rebuilt!"
    ;;

  4)
    echo "🛑 Stopping ONLY this project..."
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
    echo "📊 Status (only project containers):"
    docker-compose ps
    ;;

  8)
    echo "🧠 Ollama Models:"
    docker exec -it ai-ollama ollama list
    ;;

  9)
    echo "⚠️ SAFE RESET (keeps Elasticsearch data)"
    read -p "Are you sure? (yes/no): " confirm

    if [ "$confirm" = "yes" ]; then
      echo "🧹 Removing containers ONLY (keeping volumes)..."
      docker-compose down   # ❗ NO -v
      echo "✅ Safe reset done (ES preserved)"
    else
      echo "❌ Cancelled"
    fi
    ;;
  
  10)
    echo "🔥 Running ES reindex..."
    docker exec -it ai-search-api python3 es_indexer.py
    ;;
  
  11)
    echo "🚨 FULL RESET (WILL DELETE ES DATA)"
    read -p "Type DELETE to confirm: " confirm

    if [ "$confirm" = "DELETE" ]; then
      echo "🔥 Removing ALL containers + volumes..."
      docker-compose down -v
      echo "💀 Everything wiped (including ES index)"
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