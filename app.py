import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import google.generativeai as genai
import base64
import io
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# -----------------------------
# Gemini Setup
# -----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-3-flash-preview")

def analyze_with_gemini(image: Image.Image):
    # Converte imagem para bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    prompt = """
    Você é um especialista em inspeção de qualidade de produtos.    
    Analise a imagem e faça uma avaliação resumida do estado do produto.
    Responda às seguintes perguntas -- (SEM AS REPETIR, SOMENTE AS RESPOSTAS):

    1. Existem defeitos aparentes que possam comprometer o funcionamento do produto?    
    2. Se não, sugira passos recomendados para garantir o funcionamento do produto com base no seu conhecimento.
    3. Em sua análise final, sintetize um diagnóstico resumido para o cliente, destacando:
    - As principais conclusões e recomendações.
    - Caso, ainda assim, o produto não funcione adequadamente, o cliente pode solicitar a troca junto ao vendedor, apresentando o diagnóstico gerado por você.
    
    Sugestão de Saída:
    ## Condições Gerais do Produto: \n\n[Resumo do estado do produto, destacando quaisquer defeitos ou condições observadas.]
    ## Recomendações para Funcionamento adequado: \n\n[Passos recomendados para garantir o funcionamento do produto, caso não haja defeitos aparentes.]
    ## Diagnóstico Final: \n\n[Síntese do diagnóstico para o cliente, destacando as principais conclusões e recomendações, e orientações para solicitar troca, se necessário.]

    Fallback:
    Caso uma imagem não relacionada a produtos de e-commerce seja enviada,
    RESPONDA EDUCADAMENTE, informando que a análise é focada em produtos de e-commerce
    e peça para que o usuário envie a imagem correta do produto.
    """

    response = gemini_model.generate_content(
        [
            prompt,
            {"mime_type": "image/png", "data": img_bytes}
        ]
    )

    return response.text


# -----------------------------
# Background Removal Model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

rmbg_model = AutoModelForImageSegmentation.from_pretrained(
    "briaai/RMBG-2.0",
    trust_remote_code=True
).to(device).eval()

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def remove_background(image: Image.Image):
    image_rgb = image.convert("RGB")
    input_tensor = transform_image(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = rmbg_model(input_tensor)[-1].sigmoid().cpu()[0].squeeze()

    mask = transforms.ToPILImage()(pred)
    mask = mask.resize(image.size)

    output = image_rgb.copy()
    output.putalpha(mask)
    return output


# -----------------------------
# Combined Pipeline (Background Removal + Gemini Analysis)
# -----------------------------
def process_image(image):
    no_bg = remove_background(image)
    analysis = analyze_with_gemini(image)    
    return no_bg, analysis


# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="Suporte Automatizado para Produtos de E-commerce") as app:

    gr.Markdown("""
    # 🛠️ Suporte Automatizado para Produtos de E-commerce
    Envie uma imagem do produto para receber uma análise detalhada das condições e possíveis soluções para problemas técnicos.
    """)

    with gr.Row():

        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                width=300,
                height=300,
                label="📤 Envie a imagem do produto"
            )
            submit_btn = gr.Button("🔍 Analisar imagem")
            clear_btn = gr.Button("🧹 Limpar")

        with gr.Column(scale=1):
            output_image = gr.Image(
                type="pil",
                width=300,
                height=300,
                label="📸 Imagem analisada"
            )
            analysis_text = gr.Markdown(
                "A análise aparecerá aqui...",
                label="📄 Análise detalhada"
            )

    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, analysis_text]
    )

    clear_btn.click(
        fn=lambda: (None, "A análise aparecerá aqui..."),
        inputs=None,
        outputs=[output_image, analysis_text]
    )

if __name__ == "__main__":
    app.launch(share=False)

