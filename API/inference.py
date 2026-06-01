"""
Pipeline de inferência do HELIOS.

Funções independentes de framework: podem ser usadas pelo FastAPI,
por um worker de background, ou em testes unitários.
"""

from __future__ import annotations

import urllib.request
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import torch
from PIL import Image

from model import SolarDeep

# ── Constantes ─────────────────────────────────────────────────────────────────

CHANNELS = ["94", "131", "171", "304"]

NASA_SDO_URLS: dict[str, str] = {
    "94":  "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0094.jpg",
    "131": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0131.jpg",
    "171": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0171.jpg",
    "304": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0304.jpg",
}

AlertLevel = Literal["quiet", "moderate", "elevated", "flare"]

LEVEL_META: dict[AlertLevel, dict] = {
    "quiet":    {"label": "🟢 Sol Quieto",         "color": "#22c55e", "severity": 0},
    "moderate": {"label": "🟠 Atividade Moderada",  "color": "#f97316", "severity": 1},
    "elevated": {"label": "🟡 Atividade Elevada",   "color": "#eab308", "severity": 2},
    "flare":    {"label": "🔴 Flare Detectado",     "color": "#ef4444", "severity": 3},
}


# ── Pré-processamento ──────────────────────────────────────────────────────────

def preprocess_array(array_4ch: np.ndarray) -> torch.Tensor:
    """Converte ndarray (4, 256, 256) float32 [0,1] em tensor pronto para inferência.

    Args:
        array_4ch: array numpy com shape (4, 256, 256), valores em [0, 1]

    Returns:
        Tensor (1, 4, 256, 256) float32 — batch de tamanho 1
    """
    if array_4ch.shape != (4, 256, 256):
        raise ValueError(f"Shape esperado (4, 256, 256), recebido {array_4ch.shape}")
    return torch.tensor(array_4ch, dtype=torch.float32).unsqueeze(0)


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Converte uma imagem PIL (qualquer canal) em tensor (1, 4, 256, 256).

    Replica o canal único nos 4 canais AIA. Use apenas para testes rápidos;
    o modo NASA Live produz resultados mais precisos.
    """
    gray = image.convert("L").resize((256, 256))
    arr  = np.array(gray, dtype=np.float32) / 255.0
    arr4 = np.stack([arr] * 4, axis=0)
    return preprocess_array(arr4)


# ── Fetch NASA SDO ─────────────────────────────────────────────────────────────

def fetch_sdo_channels(timeout: int = 12) -> tuple[np.ndarray, list[str]]:
    """Baixa os 4 canais AIA atuais da NASA SDO.

    Returns:
        array_4ch  — ndarray (4, 256, 256) float32 [0, 1]
        warnings   — lista de canais que falharam (string "94", "131", ...)
    """
    array_4ch = np.zeros((4, 256, 256), dtype=np.float32)
    warnings: list[str] = []

    for i, ch in enumerate(CHANNELS):
        url = NASA_SDO_URLS[ch]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HELIOS-API/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                img = Image.open(resp).convert("L").resize((256, 256))
                array_4ch[i] = np.array(img, dtype=np.float32) / 255.0
        except Exception as exc:
            warnings.append(f"{ch}Å: {exc}")

    return array_4ch, warnings


# ── Inferência ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: SolarDeep, tensor: torch.Tensor, device: str = "cpu") -> float:
    """Executa inferência e retorna probabilidade de flare em [0, 1]."""
    tensor = tensor.to(device)
    logit  = model(tensor)
    return float(torch.sigmoid(logit).item())


def classify(probability: float, threshold: float) -> AlertLevel:
    """Converte probabilidade em nível de alerta."""
    if probability >= threshold:
        return "flare"
    if probability >= 0.60:
        return "elevated"
    if probability >= 0.30:
        return "moderate"
    return "quiet"


# ── Resultado completo ─────────────────────────────────────────────────────────

def build_result(
    probability: float,
    threshold: float,
    source: Literal["live", "upload"] = "live",
    warnings: list[str] | None = None,
) -> dict:
    """Monta o dicionário de resposta padrão da API."""
    level = classify(probability, threshold)
    meta  = LEVEL_META[level]
    now   = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "probability": round(probability, 4),
        "percentage":  round(probability * 100, 1),
        "level":       level,
        "label":       meta["label"],
        "color":       meta["color"],
        "severity":    meta["severity"],
        "threshold":   threshold,
        "is_flare":    level == "flare",
        "source":      source,
        "timestamp":   now,
        "warnings":    warnings or [],
    }
