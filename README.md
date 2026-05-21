# Tech Challenge - Fase 4

## Previsão de Preço de Ações com LSTM + FastAPI

Projeto desenvolvido para o **Tech Challenge da Pós-Tech FIAP - Machine Learning Engineering**, com foco na construção de uma pipeline completa de Deep Learning para previsão de preços de ações utilizando redes neurais LSTM.

O projeto contempla:

* coleta e tratamento de dados financeiros;
* treinamento de modelo LSTM;
* inferência de preços futuros;
* API REST com FastAPI;
* re-treinamento dinâmico do modelo;
* monitoramento de performance da aplicação;
* persistência de métricas e logs.

---

# Objetivo

Desenvolver uma solução de Machine Learning capaz de prever o preço de fechamento das ações da Petrobras (`PETR4.SA`) utilizando redes neurais do tipo Long Short-Term Memory (LSTM), disponibilizando o modelo por meio de uma API RESTful.

---

# Tecnologias Utilizadas

* Python 3.11
* TensorFlow / Keras
* FastAPI
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* yFinance
* Uvicorn
* psutil

---

# Estrutura do Projeto

```text
TechChallenge/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── petr4.csv
│
├── models/
│   ├── lstm_model.keras
│   ├── metrics.json
│   └── scaler.pkl
│
├── monitoring/
│   └── logs.json
│
├── notebooks/
│   ├── exploracao.ipynb
│   └── .ipynb_checkpoints/
│
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── model.py
│   ├── monitoring.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── retrain.py
│   └── train.py
│
├── requirements.txt
└── render.yaml
```

---

# Dataset

Os dados históricos das ações foram obtidos utilizando a biblioteca `yfinance`.

Ativo utilizado:

```python
PETR4.SA
```

Período utilizado:

```python
2018-01-01 até 2025-01-01
```

---

# Pipeline do Projeto

## 1. Coleta de Dados

Os dados históricos são baixados via Yahoo Finance e armazenados localmente em CSV.

Arquivo responsável:

```text
src/download_data.py
```

---

## 2. Pré-processamento

Etapas realizadas:

* seleção da coluna `Close`;
* normalização com `MinMaxScaler`;
* criação de janelas temporais;
* divisão treino/teste.

Arquivo responsável:

```text
src/preprocess.py
```

---

# Modelo LSTM

A arquitetura da rede neural foi construída utilizando TensorFlow/Keras.

Estrutura utilizada:

* 2 camadas LSTM;
* Dropout para redução de overfitting;
* camada Dense para previsão final.

Arquivo responsável:

```text
src/model.py
```

---

# Treinamento

O treinamento do modelo realiza:

* carregamento dos dados;
* pré-processamento;
* treinamento da rede;
* avaliação;
* persistência do modelo e métricas.

Arquivo responsável:

```text
src/train.py
```

---

# Métricas Utilizadas

Foram utilizadas as seguintes métricas:

## MAE — Mean Absolute Error

Avalia o erro absoluto médio das previsões.

## RMSE — Root Mean Squared Error

Penaliza erros maiores e mede a dispersão dos erros.

## MAPE — Mean Absolute Percentage Error

Mede o erro percentual médio das previsões.

Exemplo de resultados obtidos:

```text
MAE: 0.0506
RMSE: 0.0564
MAPE: 0.0607
```

---

# API REST

A aplicação disponibiliza uma API REST utilizando FastAPI.

Arquivo responsável:

```text
api/main.py
```

---

# Rotas Disponíveis

## GET `/`

Verifica se a API está online.

---

## GET `/health`

Endpoint de health check da aplicação.

---

## GET `/metrics`

Retorna as métricas do modelo treinado.

---

## GET `/model-info`

Retorna informações sobre o modelo utilizado.

---

## GET `/monitoring`

Retorna os logs de monitoramento da API.

---

## POST `/predict`

Realiza previsão do próximo preço da ação utilizando os últimos 60 preços históricos.

Exemplo de payload:

```json
{
  "prices": [32.1, 32.4, 32.2, ...]
}
```

---

## POST `/retrain`

Permite realizar o re-treinamento do modelo dinamicamente.

Exemplo de payload:

```json
{
  "ticker": "PETR4.SA",
  "epochs": 30,
  "batch_size": 16,
  "window_size": 60
}
```

---

# Monitoramento

Foi implementado um sistema de monitoramento utilizando middleware do FastAPI.

O sistema registra:

* tempo de resposta;
* utilização de CPU;
* utilização de memória RAM;
* endpoint acessado;
* timestamp das requisições.

Arquivo responsável:

```text
src/monitoring.py
```

Logs armazenados em:

```text
monitoring/logs.json
```

---

# Como Executar o Projeto

## 1. Clonar repositório

```bash
git clone <url-do-repositorio>
```

---

## 2. Criar ambiente virtual

```bash
python -m venv venv
```

---

## 3. Ativar ambiente virtual

### Windows

```bash
venv\Scripts\activate
```

### Linux/MacOS

```bash
source venv/bin/activate
```

---

## 4. Instalar dependências

```bash
pip install -r requirements.txt
```

---

# Download dos Dados

```bash
python src/download_data.py
```

---

# Treinamento do Modelo

```bash
python src/train.py
```

---

# Executar API

```bash
uvicorn api.main:app --reload
```

---

# Documentação Swagger

Após iniciar a API:

```text
http://127.0.0.1:8000/docs
```

---

# Monitoramento da API

Endpoint:

```text
GET /monitoring
```

Exemplo de resposta:

```json
{
  "total_requests": 10,
  "logs": [
    {
      "endpoint": "/predict",
      "response_time_seconds": 0.4231,
      "cpu_usage_percent": 21.4,
      "memory_usage_percent": 58.7
    }
  ]
}
```

---

# Melhorias Futuras

Possíveis evoluções do projeto:

* integração com Prometheus/Grafana;
* uso de múltiplas features financeiras;
* tuning automatizado de hiperparâmetros;
* versionamento de modelos;
* pipelines de CI/CD;
* uso de MLflow.

---

# Autor

Projeto desenvolvido para fins acadêmicos na Pós-Tech FIAP — Machine Learning Engineering.
