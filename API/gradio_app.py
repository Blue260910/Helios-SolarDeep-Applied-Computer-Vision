"""
HELIOS — Interface de demonstração (Gradio)

UI simples para a demonstração funcional do trabalho ACV.
Reaproveita exatamente o mesmo modelo SolarDeep e o mesmo pipeline da API REST
(model.py + inference.py), apenas trocando o FastAPI por uma interface visual.

Rodar:
    python gradio_app.py
Depois abra http://127.0.0.1:7860 no navegador.
"""

import os

from PIL import Image

from inference import (
    build_result,
    fetch_sdo_channels,
    preprocess_array,
    preprocess_pil,
    run_inference,
)
from model import load_model

import gradio as gr

# ── Carga do modelo (uma vez) ──────────────────────────────────────────────────

WEIGHTS_PATH = os.getenv("HELIOS_WEIGHTS", "models/solar_deep_best.pth")
META_PATH    = os.getenv("HELIOS_META",    "models/model_meta.json")
DEVICE       = "cpu"

print("[HELIOS] Carregando modelo SolarDeep...")
_model, _meta = load_model(WEIGHTS_PATH, META_PATH, DEVICE)
THRESHOLD = _meta["threshold"]
print(f"[HELIOS] Modelo pronto — threshold={THRESHOLD} | AUC={_meta['auc_test']}%")


# ── Funções de predição ────────────────────────────────────────────────────────

def _format(result: dict) -> tuple[str, dict]:
    """Monta um texto-resumo + o JSON bruto para exibição."""
    resumo = (
        f"## {result['label']}\n\n"
        f"- **Probabilidade de flare:** {result['percentage']}%\n"
        f"- **Nível:** `{result['level']}` (severidade {result['severity']}/3)\n"
        f"- **Threshold do modelo:** {result['threshold']}\n"
        f"- **É flare?** {'Sim 🔴' if result['is_flare'] else 'Não'}\n"
        f"- **Fonte:** {result['source']}"
    )
    return resumo, result


def predict_upload(image: Image.Image):
    """Classifica uma imagem enviada pelo usuário."""
    if image is None:
        return "⚠️ Envie uma imagem solar primeiro.", {}
    tensor = preprocess_pil(image)
    prob   = run_inference(_model, tensor, DEVICE)
    return _format(build_result(prob, THRESHOLD, source="upload"))


def predict_live():
    """Baixa os 4 canais AIA da NASA SDO e classifica em tempo real."""
    array_4ch, warnings = fetch_sdo_channels(timeout=12)
    if array_4ch.max() == 0:
        return "❌ Não foi possível baixar imagens da NASA SDO. Tente novamente.", {}
    tensor = preprocess_array(array_4ch)
    prob   = run_inference(_model, tensor, DEVICE)
    return _format(build_result(prob, THRESHOLD, source="live", warnings=warnings))


# ── Interface ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="HELIOS — Detecção de Solar Flares") as demo:
    gr.Markdown(
        "# ☀️ HELIOS — Detecção de Solar Flares\n"
        "Modelo **SolarDeep (CNN-B)** treinado do zero sobre o **SDOBenchmark (NASA/FHNW)**.  \n"
        f"AUC-ROC: **{_meta['auc_test']}%** · Accuracy: **{_meta['accuracy_test']}%** · "
        f"Classes: `no_flare` / `flare`"
    )

    with gr.Tab("📤 Enviar imagem"):
        gr.Markdown("Faça upload de uma imagem solar (JPG/PNG) para classificar.")
        with gr.Row():
            inp_img = gr.Image(type="pil", label="Imagem solar")
            with gr.Column():
                out_md_up   = gr.Markdown()
                out_json_up = gr.JSON(label="Resposta completa")
        btn_up = gr.Button("🔍 Classificar imagem", variant="primary")
        btn_up.click(predict_upload, inputs=inp_img, outputs=[out_md_up, out_json_up])

    with gr.Tab("🛰️ NASA SDO ao vivo"):
        gr.Markdown(
            "Baixa os 4 canais AIA (94, 131, 171, 304 Å) atuais da NASA SDO "
            "e classifica a atividade solar em tempo real."
        )
        out_md_live   = gr.Markdown()
        out_json_live = gr.JSON(label="Resposta completa")
        btn_live = gr.Button("🛰️ Buscar e classificar (ao vivo)", variant="primary")
        btn_live.click(predict_live, inputs=None, outputs=[out_md_live, out_json_live])


if __name__ == "__main__":
    # 0.0.0.0 funciona tanto local (via localhost) quanto no Hugging Face Spaces,
    # que precisa que a app escute em todas as interfaces para o proxy alcançá-la.
    demo.launch(server_name="0.0.0.0", server_port=7860)
