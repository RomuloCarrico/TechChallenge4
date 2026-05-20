from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

from src.retrain import retrain_model
from src.predict import predict_next_price
from src.monitoring import monitoring_middleware

# FASTAPI

app = FastAPI(
    title="PETR4 Prediction API",
    version="1.0.0"
)

app.middleware("http")(
    monitoring_middleware
)

# INPUT

class StockInput(BaseModel):

    prices: list[float]

class RetrainInput(BaseModel):

    ticker: str = "PETR4.SA"

    epochs: int = 20

    batch_size: int = 32

    window_size: int = 60

# HOME

@app.get("/",description="Rota principal da API responsável por verificar se o serviço está online e disponível para requisições.")
def home():

    return {
        "message": "API online"
    }


# Rota de Healthcheck

@app.get("/health",description="Endpoint de monitoramento utilizado para validar a saúde da aplicação e verificar se a API está operando corretamente.")
def health():

    return {
        "status": "ok"
    }

# Rota para predição
@app.post("/predict",description="Endpoint responsável por realizar previsões do próximo preço de fechamento das ações da Petrobras utilizando o modelo LSTM treinado.")
def predict(data: StockInput):

    try:

        prediction = predict_next_price(
            data.prices
        )

        return {
            "predicted_price": round(
                prediction,
                2
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }
    

# Rota para receber as métricas do modelo    
@app.get("/metrics",description="Retorna as métricas de desempenho do modelo treinado, incluindo MAE, RMSE e MAPE.")
def get_metrics():

    with open(
        "models/metrics.json",
        "r"
    ) as f:

        metrics = json.load(f)

    return metrics

@app.post("/retrain", description="Permite realizar o re-treinamento do modelo LSTM dinamicamente, possibilitando a alteração de hiperparâmetros como epochs e batch size.")
def retrain(data: RetrainInput):

    try:

        metrics = retrain_model(
            ticker=data.ticker,
            epochs=data.epochs,
            batch_size=data.batch_size,
            window_size=data.window_size
        )

        return {
            "message": "Modelo retreinado com sucesso",
            "metrics": metrics
        }

    except Exception as e:

        return {
            "error": str(e)
        }
    

@app.get(
    "/monitoring",
    description="Retorna os logs de monitoramento da API."
)
def monitoring():

    log_file = "monitoring/logs.json"

    if not os.path.exists(log_file):

        return {
            "message": "Nenhum log encontrado."
        }

    with open(log_file, "r") as f:

        logs = json.load(f)

    return {
        "total_requests": len(logs),
        "logs": logs[-20:]
    }