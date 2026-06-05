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
from PIL import Image, ImageFilter

from model import SolarDeep

# ── Constantes ─────────────────────────────────────────────────────────────────

CHANNELS = ["94", "131", "171", "304"]

# Fração do menor lado da imagem usada como janela de "zoom" na região ativa.
# O SolarDeep foi treinado em patches do SDOBenchmark com zoom numa região ativa
# preenchendo o quadro; as imagens "latest" da NASA são do DISCO INTEIRO. Sem este
# recorte, o Global Average Pooling tira a média de um disco quase vazio e a
# probabilidade fica travada (~0.66). Validado offline: 0.25 dá a melhor separação
# entre dias calmos (~0.12) e flares X (~0.90+).
AR_CROP_FRAC = 0.25

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


# ── Recorte de região ativa (casar o domínio de treino) ─────────────────────────

def _active_region_center(hot: np.ndarray) -> tuple[int, int]:
    """Centro (y, x) da região ativa = pico de brilho suavizado.

    `hot` é um mapa 2D [0,1] (usamos 94+131 Å, onde flares aparecem). O blur
    gaussiano evita travar em pixels quentes isolados (raios cósmicos).
    """
    blurred = np.asarray(
        Image.fromarray((np.clip(hot, 0, 1) * 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=6)),
        dtype=np.float32,
    )
    cy, cx = np.unravel_index(int(blurred.argmax()), blurred.shape)
    return int(cy), int(cx)


def _crop_window(arr: np.ndarray, cy: int, cx: int, side: int) -> np.ndarray:
    """Recorta uma janela quadrada `side`×`side` centrada em (cy, cx), presa às bordas."""
    h, w = arr.shape
    y0 = min(max(0, cy - side // 2), max(0, h - side))
    x0 = min(max(0, cx - side // 2), max(0, w - side))
    return arr[y0:y0 + side, x0:x0 + side]


def active_region_crop(channels: list[np.ndarray], frac: float = AR_CROP_FRAC) -> np.ndarray:
    """Recorta a região ativa nos 4 canais e redimensiona para (4, 256, 256).

    Args:
        channels: lista de 4 mapas 2D [0,1] (94, 131, 171, 304), mesmo tamanho,
                  representando o DISCO INTEIRO.
        frac:     lado da janela como fração do menor lado da imagem.

    Returns:
        ndarray (4, 256, 256) float32 — patch zoomado, no domínio do treino.

    O centro é detectado UMA vez (em 94+131) e aplicado igual aos 4 canais,
    preservando o alinhamento espacial entre comprimentos de onda.
    """
    cy, cx = _active_region_center(channels[0] + channels[1])
    side = max(8, int(min(channels[0].shape) * frac))
    out = np.empty((4, 256, 256), dtype=np.float32)
    for i, c in enumerate(channels):
        win = _crop_window(c, cy, cx, side)
        img = Image.fromarray((np.clip(win, 0, 1) * 255).astype(np.uint8)).resize((256, 256))
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


# ── Fetch NASA SDO ─────────────────────────────────────────────────────────────

def fetch_sdo_channels(timeout: int = 12) -> tuple[np.ndarray, list[str]]:
    """Baixa os 4 canais AIA atuais da NASA SDO.

    Returns:
        array_4ch  — ndarray (4, 256, 256) float32 [0, 1]
        warnings   — lista de canais que falharam (string "94", "131", ...)
    """
    full: list[np.ndarray | None] = [None] * 4
    warnings: list[str] = []

    for i, ch in enumerate(CHANNELS):
        url = NASA_SDO_URLS[ch]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HELIOS-API/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # Mantém o DISCO INTEIRO (sem resize) — o recorte da região ativa vem depois
                img = Image.open(resp).convert("L")
                full[i] = np.asarray(img, dtype=np.float32) / 255.0
        except Exception as exc:
            warnings.append(f"{ch}Å: {exc}")

    ok = [c for c in full if c is not None]
    if not ok:
        # Nenhum canal baixou — devolve array zerado (main.py converte em 502)
        return np.zeros((4, 256, 256), dtype=np.float32), warnings

    # Canais que falharam viram zeros do mesmo tamanho dos que baixaram
    shape = ok[0].shape
    channels = [c if c is not None else np.zeros(shape, dtype=np.float32) for c in full]

    return active_region_crop(channels), warnings


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
