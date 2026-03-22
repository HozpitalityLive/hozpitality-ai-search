
#!/bin/bash

echo "🧹 Cleaning FAISS files..."

rm -f faiss.index vector_data.json id_map.json

echo "♻️ FAISS cleaned!"

./restart.sh

echo "✅ Clean restart completed!"