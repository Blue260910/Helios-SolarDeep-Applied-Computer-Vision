---
title: HELIOS Solar Flare Detection
emoji: ☀️
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: gradio_app.py
pinned: false
---

# HELIOS — Solar Flare Detection API

Serviço de inferência do modelo **SolarDeep (CNN-B)** treinado sobre o [SDOBenchmark](https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark) (NASA/FHNW).  
Classifica imagens do satélite SDO nos 4 canais AIA em **flare M/X-class** ou **atividade normal**.

---

## Estrutura da pasta

```
API/
├── main.py           # FastAPI app — rotas e middleware
├── model.py          # Classe SolarDeep + load_model()
├── inference.py      # Pipeline: fetch SDO → preprocess → inferência → resposta
├── requirements.txt  # Dependências Python
├── README.md         # Esta documentação
│
│   (arquivos de modelo — baixar do Colab antes de rodar)
├── solar_deep_best.pth   ← pesos do modelo (não incluído no repo — ~9 MB)
└── model_meta.json       ← threshold, canais, métricas
```

---

## Setup

### 1. Pré-requisitos

- Python 3.10+
- Os dois arquivos de modelo na pasta `API/`:
  - `solar_deep_best.pth` — baixar do Colab após treinar (`helios_export/solar_deep_best.pth`)
  - `model_meta.json` — já presente em `helios_export_v3/model_meta.json`

### 2. Instalar dependências

```bash
cd API
pip install -r requirements.txt
```

> Para usar GPU: instale o PyTorch com suporte CUDA em vez do padrão.  
> Guia: [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 3. Rodar o servidor

```bash
uvicorn main:app --reload --port 8000
```

A API ficará disponível em `http://localhost:8000`.  
Documentação interativa (Swagger): `http://localhost:8000/docs`

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `HELIOS_WEIGHTS` | `solar_deep_best.pth` | Caminho para os pesos do modelo |
| `HELIOS_META` | `model_meta.json` | Caminho para o JSON de metadados |

```bash
# Exemplo com caminhos customizados
HELIOS_WEIGHTS=/models/solar_deep_best.pth uvicorn main:app --port 8000
```

---

## Endpoints

### `GET /health`

Verifica se o serviço está pronto.

**Resposta:**
```json
{
  "status": "ok",
  "model": "SolarDeep",
  "version": "5.0.0",
  "device": "cpu",
  "threshold": 0.82
}
```

---

### `GET /meta`

Retorna os metadados completos do modelo: métricas de avaliação, threshold calibrado, canais e versão.

**Resposta:**
```json
{
  "model": "SolarDeep",
  "version": "5.0.0",
  "task": "binary_flare_detection",
  "channels": ["94", "131", "171", "304"],
  "threshold": 0.82,
  "classes": ["no_flare", "flare"],
  "input_shape": [4, 256, 256],
  "f1_test": 47.83,
  "auc_test": 85.59,
  "accuracy_test": 83.75,
  "dataset": "SDOBenchmark (NASA/FHNW)"
}
```

---

### `GET /predict/live`

Baixa os 4 canais AIA diretamente da NASA SDO (imagens públicas, atualizadas a cada ~15 min) e retorna a previsão.

**Resposta:**
```json
{
  "probability": 0.9123,
  "percentage": 91.2,
  "level": "flare",
  "label": "🔴 Flare Detectado",
  "color": "#ef4444",
  "severity": 3,
  "threshold": 0.82,
  "is_flare": true,
  "source": "live",
  "timestamp": "2026-06-01T14:30:00+00:00",
  "warnings": []
}
```

**Níveis de alerta:**

| `level` | `severity` | Faixa de probabilidade |
|---|---|---|
| `quiet` | 0 | < 30% |
| `moderate` | 1 | 30 – 60% |
| `elevated` | 2 | 60 – 82% |
| `flare` | 3 | ≥ 82% |

---

### `POST /predict/upload`

Aceita uma imagem solar (JPG ou PNG) e retorna a previsão.  
A imagem é convertida para escala de cinza e replicada nos 4 canais — menos preciso que `/predict/live`.

**Request:** `multipart/form-data` com campo `file`

```bash
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@minha_imagem_solar.jpg"
```

**Resposta:** mesmo schema de `/predict/live`, com `"source": "upload"`.

---

## Integração com o site HELIOS (Next.js)

### Opção A — Fetch direto do servidor Next.js

Crie a rota `app/api/flare-detection/route.ts`:

```typescript
// app/api/flare-detection/route.ts
import { NextResponse } from 'next/server'

const HELIOS_API = process.env.HELIOS_API_URL ?? 'http://localhost:8000'

export async function GET() {
  try {
    const res = await fetch(`${HELIOS_API}/predict/live`, {
      next: { revalidate: 900 }, // cache de 15 min (intervalo da NASA SDO)
    })

    if (!res.ok) throw new Error(`HELIOS API error: ${res.status}`)

    const data = await res.json()
    return NextResponse.json(data)
  } catch (err) {
    return NextResponse.json(
      { error: 'Serviço de detecção indisponível', details: String(err) },
      { status: 503 }
    )
  }
}
```

Adicione ao `.env.local`:
```env
HELIOS_API_URL=http://localhost:8000   # dev
# HELIOS_API_URL=https://sua-api.railway.app  # produção
```

### Opção B — Hook React para polling automático

```typescript
// hooks/useFlareDetection.ts
import { useEffect, useState } from 'react'

interface FlareResult {
  probability: number
  percentage: number
  level: 'quiet' | 'moderate' | 'elevated' | 'flare'
  label: string
  color: string
  severity: number
  is_flare: boolean
  timestamp: string
  warnings: string[]
}

export function useFlareDetection(intervalMs = 15 * 60 * 1000) {
  const [data, setData]       = useState<FlareResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const res = await fetch('/api/flare-detection')
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        setData(await res.json())
        setError(null)
      } catch (e) {
        setError(String(e))
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const id = setInterval(fetchData, intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])

  return { data, loading, error }
}
```

### Uso no componente SolarAlertFeed

```tsx
// components/helios/dashboard/SolarAlertFeed.tsx (trecho)
import { useFlareDetection } from '@/hooks/useFlareDetection'

export function SolarAlertFeed() {
  const { data, loading } = useFlareDetection()

  if (loading) return <Spinner />

  return (
    <div>
      <Badge style={{ background: data?.color }}>
        {data?.label}
      </Badge>
      <p>Probabilidade: {data?.percentage}%</p>
      {data?.warnings.length > 0 && (
        <p className="text-yellow-500">
          Canais com falha: {data.warnings.join(', ')}
        </p>
      )}
    </div>
  )
}
```

---

## Deploy em produção

### Azure VM + Docker (recomendado)

#### 1. Criar a VM na Azure

No portal Azure (ou Azure CLI), crie uma VM Linux Ubuntu 22.04:
- **CPU only:** Standard B2s (2 vCPUs, 4 GB RAM) — suficiente para inferência
- **GPU:** Standard NC6s v3 (Tesla V100) — para baixa latência em produção

```bash
# Azure CLI
az vm create \
  --resource-group helios-rg \
  --name helios-api-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username azureuser \
  --generate-ssh-keys

# Abrir a porta 8000
az vm open-port --resource-group helios-rg --name helios-api-vm --port 8000
```

#### 2. Instalar Docker na VM

```bash
# Conectar na VM
ssh azureuser@<IP-DA-VM>

# Instalar Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

#### 3. Copiar os artefatos do modelo

Do seu computador local, envie os pesos para a VM:

```bash
# Criar pasta de modelos na VM
ssh azureuser@<IP-DA-VM> "mkdir -p ~/helios-models"

# Enviar os arquivos (do Colab: baixe primeiro para o seu PC)
scp solar_deep_best.pth azureuser@<IP-DA-VM>:~/helios-models/
scp model_meta.json     azureuser@<IP-DA-VM>:~/helios-models/
```

#### 4. Subir a API

```bash
# Na VM: clonar o repositório ou copiar a pasta API/
git clone https://github.com/SEU-REPO/helios-acv.git
cd helios-acv/API

# Build e start
docker compose up -d --build

# Verificar se está rodando
docker compose logs -f
curl http://localhost:8000/health
```

A API estará disponível em `http://<IP-DA-VM>:8000`.

#### 5. (Opcional) GPU na VM

Se a VM tiver GPU NVIDIA, instale o NVIDIA Container Toolkit e descomente o bloco `deploy` no `docker-compose.yml`:

```bash
# Instalar NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

No `Dockerfile`, troque o `FROM` para a imagem com CUDA (instrução comentada no arquivo).

---

### Atualizar o modelo sem rebuild

Como os pesos ficam em um volume externo à imagem (`~/helios-models/`), atualizar o modelo é só substituir o arquivo e reiniciar o container — sem novo build:

```bash
# Enviar pesos novos para a VM
scp solar_deep_best.pth azureuser@<IP-DA-VM>:~/helios-models/

# Reiniciar o container para carregar os novos pesos
ssh azureuser@<IP-DA-VM> "cd helios-acv/API && docker compose restart"
```

---

### Railway (alternativa gratuita para projetos acadêmicos)

1. Crie um projeto em [railway.app](https://railway.app)
2. Adicione um serviço Python e aponte para a pasta `API/`
3. Configure as variáveis de ambiente (`HELIOS_WEIGHTS`, `HELIOS_META`)
4. O Railway detecta o `requirements.txt` automaticamente
5. Defina o comando de start:
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

### Render

```yaml
# render.yaml
services:
  - type: web
    name: helios-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: HELIOS_WEIGHTS
        value: solar_deep_best.pth
      - key: HELIOS_META
        value: model_meta.json
```

### CORS em produção

Antes de fazer deploy, atualize `ALLOWED_ORIGINS` em `main.py`:

```python
ALLOWED_ORIGINS = [
    "https://helius-zeta.vercel.app",   # produção
]
```

Remova o `"*"` — ele é apenas para desenvolvimento local.

---

## Arquitetura do modelo

| Propriedade | Valor |
|---|---|
| Nome | SolarDeep (CNN-B) |
| Parâmetros | 2.388.929 |
| Input | tensor (4, 256, 256) — 4 canais AIA normalizados |
| Output | logit escalar (sigmoid → probabilidade) |
| Blocos | 5 blocos conv duplos + BatchNorm + MaxPool |
| Treinamento | SDOBenchmark · 8.336 amostras · AdamW + CosineAnnealingLR |
| AUC-ROC (test) | **85.59%** |
| Accuracy (test) | 83.75% |
| F1 flare (test) | 47.83% |
| Threshold | 0.82 (calibrado por F1 no val set) |

---

## Fluxo completo

```
[Site Next.js]
  └── GET /api/flare-detection  (a cada 15 min)
        └── [API FastAPI — main.py]
              └── GET /predict/live
                    ├── fetch_sdo_channels()     → NASA SDO (4 URLs públicas)
                    ├── preprocess_array()       → tensor (1, 4, 256, 256)
                    ├── run_inference(model)     → probabilidade 0–1
                    └── build_result()           → JSON com level, label, color...
```
