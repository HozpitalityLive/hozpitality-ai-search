# 🚀 Hozpitality AI Search

A GPU-powered AI search system using **Docker, Ollama, Redis, and Elasticsearch**.

---

## 📦 Tech Stack

* 🐳 Docker & Docker Compose
* 🤖 Ollama (LLM models)
* 🔍 Elasticsearch
* ⚡ Redis
* 🧠 PyTorch (CUDA enabled)
* 🚀 FastAPI (Uvicorn)

---

## 📁 Project Structure

```
.
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── main.py
├── Modelfile.llama
├── Modelfile.phi3
└── manage.sh
```

---

## ⚙️ Setup & Run

### 🔨 Start (Build + Run)

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

### 🔄 Restart

```bash
docker-compose down
docker-compose up -d
```

---

### 🔥 Rebuild (Clean)

```bash
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

---

### 🛑 Stop

```bash
docker-compose down
```

---

## 📜 Logs

```bash
docker logs -f ai-search-api
```

---

## 📊 Status

```bash
docker ps
```

---

## 🧪 GPU Check

```bash
docker exec -it ai-search-api nvidia-smi
```

---

## 🧠 Ollama Model Management

### 📥 Pull Base Models

```bash
docker exec -it ai-ollama ollama pull llama3
docker exec -it ai-ollama ollama pull phi3
```

---

### 🏗️ Create Custom Models

```bash
docker exec -it ai-ollama ollama create llama3-hoz -f /app/Modelfile.llama
docker exec -it ai-ollama ollama create phi3-hoz -f /app/Modelfile.phi3
```

---

### ❌ Remove Models

```bash
docker exec -it ai-ollama ollama rm llama3
docker exec -it ai-ollama ollama rm phi3
```

---

### 📋 List Models

```bash
docker exec -it ai-ollama ollama list
```

---

## 🧹 Clean Reset ⚠️

⚠️ This deletes **all containers, volumes, and cache**

```bash
docker-compose down -v
docker system prune -af
docker volume prune -f
```

---

## 🧩 Services

### 🧠 AI Search API

* Port: `8001`
* Container: `ai-search-api`

---

### ⚡ Redis

* Port: `6377`

---

### 🔍 Elasticsearch

* Port: `9200`
* Security disabled
* Single node

---

### 🤖 Ollama

* Port: `11434`
* GPU enabled

---

## 🐳 Docker Compose

```yaml
version: "3.9"

services:

  ai-search:
    build: .
    container_name: ai-search-api
    ports:
      - "8001:8000"
    env_file:
      - .env
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
    depends_on:
      - redis
      - elasticsearch
      - ollama
    restart: always

  redis:
    image: redis:7
    container_name: ai-redis
    ports:
      - "6377:6379"
    restart: always

  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: ai-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    ports:
      - "9200:9200"
    restart: always

  ollama:
    image: ollama/ollama
    container_name: ai-ollama
    ports:
      - "11434:11434"
    volumes:
      - .:/app
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
    restart: always
```

---

## 🐋 Dockerfile

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:main_app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🧰 Optional: Interactive Manager Script

Run:

```bash
chmod +x manage.sh
./manage.sh
```

Features:

* Start / Restart / Stop
* Rebuild
* Logs
* GPU check
* Model management
* Full reset

---

## ⚠️ Notes

* Ensure **NVIDIA Docker runtime** is installed
* GPU required for best performance
* Avoid committing `.env` with secrets

---

## ✅ Quick Start (TL;DR)

```bash
docker-compose up -d --build
```

Then:

```bash
docker exec -it ai-ollama ollama pull llama3
docker exec -it ai-ollama ollama create llama3-hoz -f /app/Modelfile.llama
```

---

## 👨‍💻 Author

Built for scalable AI-powered search systems 🚀
