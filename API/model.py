"""
SolarDeep — arquitetura CNN-B do HELIOS.

Este arquivo contém apenas a definição da classe e a função de carga do modelo.
Os pesos são carregados externamente via load_model().
"""

import json
import torch
import torch.nn as nn
from pathlib import Path

# ── Arquitetura ────────────────────────────────────────────────────────────────

class SolarDeep(nn.Module):
    """CNN-B: 5 blocos conv duplos + BatchNorm. Modelo final do HELIOS.

    Input:  tensor (batch, 4, 256, 256) — 4 canais AIA normalizados [0, 1]
    Output: tensor (batch,)             — logit (aplicar sigmoid para probabilidade)
    """

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        def block(in_ch: int, out_ch: int, drop: float = 0.3) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2), nn.Dropout2d(drop),
            )

        self.features = nn.Sequential(
            block(4,   32,  0.2),   # 256 → 128
            block(32,  64,  0.3),   # 128 → 64
            block(64,  128, 0.3),   #  64 → 32
            block(128, 256, 0.4),   #  32 → 16
            block(256, 256, 0.4),   #  16 → 8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x))).squeeze(1)


# ── Carga ──────────────────────────────────────────────────────────────────────

def load_model(
    weights_path: str = "solar_deep_best.pth",
    meta_path: str    = "model_meta.json",
    device: str       = "cpu",
) -> tuple[SolarDeep, dict]:
    """Carrega o modelo e os metadados de calibração.

    Returns:
        model   — SolarDeep em modo eval(), pronto para inferência
        meta    — dict com threshold, channels, métricas etc.
    """
    weights_path = Path(weights_path)
    meta_path    = Path(meta_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Pesos não encontrados em '{weights_path}'. "
            "Baixe 'solar_deep_best.pth' do Colab e coloque na pasta API/."
        )
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadados não encontrados em '{meta_path}'.")

    with open(meta_path) as f:
        meta = json.load(f)

    model = SolarDeep()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, meta
