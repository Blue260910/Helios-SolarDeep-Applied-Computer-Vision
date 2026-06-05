# 🌞 HELIOS — Detecção de Flares Solares via CNN

**Applied Computer Vision · FIAP Global Solution 2026**

Sistema de detecção de flares solares M/X-class usando redes neurais convolucionais treinadas do zero sobre imagens do satélite SDO (NASA). Integra monitoramento em tempo real via API da NASA SDO.

---

## Links da Entrega

- **Repositório GitHub:** _(este repositório público)_
- **Vídeo de demonstração (até 3 min):** _<!-- TODO: inserir link do YouTube -->_

---

## Integrantes

| Nome | RM |
|------|----|
| Felipe Cortez | RM 99750 |
| Julia Lins | RM 98690 |
| Luis Barreto | RM 99210 |
| Victor Aranda | RM 99667 |
| Guilherme Akio | RM 98582 |

---

## Resultados

| Modelo | Parâmetros | AUC | Accuracy | F1 Flare |
|--------|-----------|-----|----------|----------|
| CNN-A Solar-Lite | 101.857 | 85.0% | ~84% | ~40% |
| **CNN-B Solar-Deep** | **2.388.929** | **85.6%** | **83.75%** | **47.8%** |

Dataset: SDOBenchmark (NASA/FHNW) · 8.336 treino + 886 teste · 2 classes (flare / no_flare)

---

## Estrutura do Repositório

```
.
├── helios_acv.ipynb          # Notebook principal — treino, avaliação, Gradio
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
├── helios_export/            # Artefatos gerados pelo notebook
│   ├── model_meta.json       # Metadados do modelo (threshold, métricas)
│   ├── confusion_matrices.png
│   ├── learning_curves.png
│   ├── threshold_sweep.png
│   ├── prediction_examples.png
│   ├── class_distribution.png
│   └── channel_comparison.png
└── API/                      # Serviço de inferência (FastAPI + Gradio, deploy no HF Space)
    ├── main.py · gradio_app.py · inference.py · model.py
    ├── Dockerfile · requirements.txt
    └── models/
        ├── solar_deep_best.pth   # Pesos do melhor modelo (CNN-B Solar-Deep)
        └── model_meta.json       # Metadados do modelo (threshold, métricas)
```

> As imagens de teste (2 flare + 2 no_flare) e os pesos `solar_deep_best.pth`
> são gerados ao rodar o notebook (células de exportação). Os pesos versionados
> ficam em `API/models/` para alimentar o serviço de inferência.

---

## Caso de Uso Real — Integração com o Site HELIOS

O modelo treinado neste projeto é o núcleo de detecção de flares do **site HELIOS** ([Helios](https://helius-zeta.vercel.app/)), plataforma de monitoramento de clima espacial desenvolvida em paralelo pela equipe para a Global Solution 2026.
<img width="2277" height="847" alt="image" src="https://github.com/user-attachments/assets/ac9c7d51-77ed-49d2-8948-08a550edf1c5" />

### Pipeline de produção

```
NASA SDO AIA (público, ~15 min de delay)
    │  94Å · 131Å · 171Å · 304Å
    ▼
Pré-processamento: resize 256×256 · normalização [0,1] · tensor (4, 256, 256)
    ▼
SolarDeep CNN-B — solar_deep_best.pth
    │  threshold: 0.82
    ▼
{ probability, level, timestamp }
    ▼
SolarAlertFeed — dashboard /protection do site
```

### Níveis de alerta gerados

| Probabilidade | Nível | Ação no site |
|---|---|---|
| < 30% | 🟢 Sol Quieto | Sem notificação |
| 30 – 60% | 🟠 Atividade Moderada | Badge no feed |
| 60 – 82% | 🟡 Atividade Elevada | Alerta laranja |
| ≥ 82% | 🔴 Flare Detectado | Alerta vermelho |

O endpoint `/api/flare-detection` do site chama o serviço de inferência (FastAPI + PyTorch) a cada 15 minutos, consumindo os artefatos `helios_export/solar_deep_best.pth` e `helios_export/model_meta.json` gerados por este notebook.

---

## Como Executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Opção 1 — Google Colab (recomendado)

1. Abra `helios_acv.ipynb` no Google Colab
2. Na primeira execução, faça upload do `kaggle.json` quando solicitado
   - Obtenha em: kaggle.com → Account → Create API Token
3. Execute todas as células em sequência (`Runtime > Run all`)
4. O download do dataset (~2GB) ocorre automaticamente
5. A interface Gradio é lançada ao final com link público

### Opção 2 — Apenas a demo (sem retreinar)

Para rodar apenas a interface com os pesos já treinados:

```python
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Carregar modelo
# (copie as classes SolarDeep da célula models001 do notebook)
model = SolarDeep()
model.load_state_dict(torch.load('API/models/solar_deep_best.pth', map_location='cpu'))
model.eval()
```

### Opção 3 — NASA Live (sem dataset)

A célula Gradio do notebook inclui modo **NASA Live** que baixa imagens em tempo real da NASA SDO:
```
https://sdo.gsfc.nasa.gov/assets/img/latest/latest_256_0094.jpg
```
Não requer o dataset SDOBenchmark para demonstração.

---

## Dataset

**SDOBenchmark** — NASA/FHNW  
Disponível em: https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark  
Referência: Galvez et al. (2019), *A Machine-Learning Dataset Prepared from the NASA Solar Dynamics Observatory Mission*

**Classes:**
- `flare`: peak_flux ≥ 1e-5 W/m² (flare M ou X-class)
- `no_flare`: peak_flux < 1e-5 W/m²

**Canais AIA utilizados:** 94Å · 131Å · 171Å · 304Å (último timestep — 10min antes do evento)

---

## Arquitetura

### CNN-A: Solar-Lite (baseline)
- 3 blocos conv · sem BatchNorm · 101k parâmetros
- Input: (4, 256, 256) → Conv→ReLU→MaxPool×3 → GAP → FC

### CNN-B: Solar-Deep (modelo final)
- 5 blocos conv duplos + BatchNorm · 2.4M parâmetros
- Input: (4, 256, 256) → [Conv→BN→ReLU→Conv→BN→ReLU→MaxPool]×5 → GAP → FC

---

## Demonstração Funcional

A interface Gradio no notebook oferece:
- **🛰️ NASA Live**: análise do sol em tempo real usando os 4 canais AIA da NASA SDO
- **📤 Upload Manual**: classificação de imagens solares enviadas pelo usuário
- 4 níveis de alerta: Sol Quieto · Atividade Moderada · Atividade Elevada · Flare Detectado
- Explicação do threshold e barra de probabilidade visual
