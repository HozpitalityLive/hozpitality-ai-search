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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]