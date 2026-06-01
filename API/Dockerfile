# ── CPU (padrão) ───────────────────────────────────────────────────────────────
# Imagem leve, sem CUDA. Boa para Azure B-series (uso geral).
FROM python:3.11-slim AS base

# ── GPU (Azure NC-series) ──────────────────────────────────────────────────────
# Troque o FROM acima por este se a VM tiver GPU NVIDIA:
# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

WORKDIR /app

# Instala dependências do sistema necessárias para Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (cache de camada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY model.py inference.py main.py ./

# Os pesos do modelo NÃO são copiados para a imagem —
# são montados como volume em tempo de execução (ver docker-compose.yml)
# Isso evita rebuildar a imagem a cada retreino do modelo.

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
