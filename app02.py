from pandas import DataFrame
import yaml
import json
import time
from collections import Counter
from os import makedirs

import pandas as pd
import plotly.express as px
from transformers import pipeline
import psycopg2
import gradio as gr


# Carregar modelo uma única vez (otimização de performance)
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    batch_size=64,
    truncation=True,
    padding=True,
    max_length=512
)

# Ciação de view para otimizar as consultas
with open("db_config.yaml", "r") as file:
    config = yaml.safe_load(file)["database"]
conn = psycopg2.connect(
    host=config["host"],
    port=config["port"],
    dbname=config["dbname"],
    user=config["user"],
    password=config["password"]
    )
cursor = conn.cursor()    
cursor.execute("""CREATE OR REPLACE VIEW sentiment_reviews
                AS select text_review, review_date
                FROM reviews
                WHERE text_review IS NOT NULL
                ORDER BY review_date DESC""")
conn.commit()
cursor.close()
conn.close()


# Conexão com o banco de dados para consulta
def query_database(limit=500) -> DataFrame:
    """Consulta os reviews mais recentes do banco de dados, limitando a quantidade para análise.

    Args:
        limit (int, optional): Número de reviews a serem consultados. Defaults to 500.

    Returns:
        DataFrame: DataFrame contendo os reviews para análise de sentimentos.
    """
    with open("db_config.yaml", "r") as file:
        config = yaml.safe_load(file)["database"]

    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname=config["dbname"],
        user=config["user"],
        password=config["password"]
    )
    cursor = conn.cursor()    
    cursor.execute(f"SELECT text_review FROM sentiment_reviews ORDER BY RANDOM() LIMIT {limit}")
    results = cursor.fetchall()
    df = pd.DataFrame(results, columns=["review_text"])
    cursor.close()
    conn.close()
    return df

# Função para logar o tempo de execução
def log_execution_time(limit, exec_time) -> None:
    """Loga o tempo de execução em um arquivo JSON.

    Args:
        limit (_type_): Número de reviews a serem consultados.
        exec_time (_type_): Tempo de execução em segundos.
    """
    log_entry = {"review_count": limit, "execution_time": exec_time}
    
    makedirs("logs", exist_ok=True) # Criar diretório de logs se não existir
    with open("logs/execution_log.json", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")
    

# Função principal de análise de sentimentos
def sentiment_analysis(limit=500):
    """Realiza a análise de sentimentos nos reviews consultados do banco de dados, 
    calcula o tempo de execução e gera um gráfico de distribuição dos sentimentos.

    Args:
        limit (_type_): Número de reviews a serem consultados.   

    Returns:
        DataFrame: DataFrame contendo os reviews para análise de sentimentos (Top 50).
        Figure: Gráfico de distribuição dos sentimentos (PLOTLY).
        str: Tempo de execução formatado para exibição em interface Gradio.
    """
    df = query_database(limit)
    start_time = time.time()
    results = classifier(df["review_text"].tolist())
    df["sentiment"] = [res["label"] for res in results]
    df["confidence"] = [round(res["score"], 2) for res in results]
    df = df[["sentiment", "confidence", "review_text"]]
    end_time = time.time()
    exec_time = round(end_time - start_time, 2) # Tempo de execução em segundos
    log_execution_time(limit, exec_time) # Log de performance

    # Distribuição e Porcentagem
    sentiment_counts = Counter(df["sentiment"])
    total_reviews = sum(sentiment_counts.values())

    # Calcula porcentagem
    sentiment_percentages = {
        sentiment: round((count / total_reviews) * 100, 2)
        for sentiment, count in sentiment_counts.items()
    }
    
    color_map = {
        "POSITIVE": "#2ECC71",  # Verde para positivo
        "NEGATIVE": "#E74C3C"   # Vermelho para negativo
    }
    figure = px.bar(
        x=list(sentiment_counts.keys()),
        y=list(sentiment_counts.values()),
        labels={"x": "Sentimento predominante", "y": "Quantidade de reviews", 
                "text": "Porcentagem", "color": "Sentimento"},
        title="Distribuição de Sentimentos",        
        template="plotly_white",
        color=list(sentiment_counts.keys()), # Define as cores com base nos rótulos
        color_discrete_map=color_map,         # Aplica o mapa de cores personalizado  
        text=[f"{sentiment_percentages[s]}%" for s in sentiment_counts.keys()] # Adiciona porcentagem      
        )
    
    # Ajustes estéticos adicionais
    figure.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")), # Borda escura para as barras
                         textposition="inside") 
    
    figure.update_layout(
        title_font=dict(size=22, family="Arial", color="black"),
        xaxis=dict(title_font=dict(size=17), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=17), tickfont=dict(size=14)),
        bargap=0.25, # Espaçamento entre as barras
    )
        

    return df.head(50), figure, f"Tempo de execução: {exec_time} segundos"


# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 📊 Análise de Sentimentos em Reviews - Amazon Products")
    gr.Markdown("### Este painel mostra performance e distribuição de sentimentos dos Clientes.")

    with gr.Row():
        with gr.Column(scale=1):
            limit_input = gr.Slider(100, 5000, value=500, step=100, label="Número de reviews a analisar")
            run_button = gr.Button("Executar análise", variant="primary")
            time_output = gr.Markdown(label="Métrica de performance")

        with gr.Column(scale=2):            
            with gr.Tab("📈 Distribuição de Sentimentos"):
                plot_output = gr.Plot(label="Gráfico de Sentimentos")
            
            with gr.Tab("📑 Resultados"):
                df_output = gr.Dataframe(
                    headers=["sentiment","confidence", "review_text"], 
                    label="Tabela de Reviews"
                )

    run_button.click(
        sentiment_analysis,
        inputs=limit_input,
        outputs=[df_output, plot_output, time_output]
    )


if __name__ == "__main__":
    demo.launch()
