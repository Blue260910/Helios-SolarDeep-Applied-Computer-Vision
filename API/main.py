"""
HELIOS Flare Detection API

FastAPI app que expõe o modelo SolarDeep como serviço REST.
Rotas:
  GET  /health               — status do serviço e do modelo
  GET  /predict/live         — inferência com imagens NASA SDO em tempo real
  POST /predict/upload       — inferência com imagem enviada pelo cliente
  GET  /meta                 — metadados e métricas do modelo
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference import (
    build_result,
    fetch_sdo_channels,
    preprocess_array,
    preprocess_pil,
    run_inference,
)
from model import SolarDeep, load_model

# ── Configuração ───────────────────────────────────────────────────────────────

WEIGHTS_PATH = os.getenv("HELIOS_WEIGHTS", "solar_deep_best.pth")
META_PATH    = os.getenv("HELIOS_META",    "model_meta.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Origens permitidas para CORS — adicione o domínio do site em produção
ALLOWED_ORIGINS = [
    "http://localhost:3000",           # Next.js dev
    "https://helius-zeta.vercel.app",  # produção
]

# ── Estado global (carregado uma vez no startup) ───────────────────────────────

_model: SolarDeep | None = None
_meta:  dict | None      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _meta
    print(f"[HELIOS] Carregando modelo em {DEVICE}...")
    _model, _meta = load_model(WEIGHTS_PATH, META_PATH, DEVICE)
    print(f"[HELIOS] Modelo pronto — threshold={_meta['threshold']} | AUC={_meta['auc_test']}%")
    yield
    _model = None
    _meta  = None


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HELIOS — Solar Flare Detection API",
    description=(
        "API de inferência do modelo SolarDeep (CNN-B) treinado sobre o SDOBenchmark (NASA/FHNW). "
        "Classifica imagens solares AIA em flare M/X-class ou atividade normal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_model() -> tuple[SolarDeep, dict]:
    if _model is None or _meta is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    return _model, _meta


# ── Rotas ──────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Verifica se o serviço e o modelo estão prontos."""
    return {
        "status":       "ok" if _model is not None else "loading",
        "model":        _meta.get("model") if _meta else None,
        "version":      _meta.get("version") if _meta else None,
        "device":       DEVICE,
        "threshold":    _meta.get("threshold") if _meta else None,
    }


@app.get("/meta", tags=["Modelo"])
def model_meta():
    """Retorna os metadados completos do modelo (métricas, canais, versão)."""
    _, meta = _require_model()
    return meta


@app.get("/predict/live", tags=["Inferência"])
def predict_live():
    """Baixa os 4 canais AIA da NASA SDO e retorna a previsão de flare.

    As imagens da NASA SDO são atualizadas a cada ~15 minutos.
    Ideal para polling periódico no dashboard.
    """
    model, meta = _require_model()
    threshold   = meta["threshold"]

    array_4ch, warnings = fetch_sdo_channels(timeout=12)

    if array_4ch.max() == 0:
        raise HTTPException(
            status_code=502,
            detail="Não foi possível baixar nenhum canal da NASA SDO. Tente novamente.",
        )

    tensor      = preprocess_array(array_4ch)
    probability = run_inference(model, tensor, DEVICE)

    return build_result(probability, threshold, source="live", warnings=warnings)


@app.post("/predict/upload", tags=["Inferência"])
async def predict_upload(file: Annotated[UploadFile, File(description="Imagem solar (JPG/PNG)")]):
    """Classifica uma imagem solar enviada pelo cliente.

    A imagem é convertida para escala de cinza e replicada nos 4 canais AIA.
    Menos preciso que /predict/live — use para testes ou demonstração.
    """
    model, meta = _require_model()
    threshold   = meta["threshold"]

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Formato aceito: JPEG ou PNG.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")

    tensor      = preprocess_pil(image)
    probability = run_inference(model, tensor, DEVICE)

    return build_result(probability, threshold, source="upload")
