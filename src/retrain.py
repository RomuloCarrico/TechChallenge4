import os
import json
import joblib

#Script para retreinar do modelo 
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from src.preprocess import (
    load_data,
    preprocess_data,
    create_sequences,
    split_data
)

from src.model import build_lstm_model


def retrain_model(
    ticker="PETR4.SA",
    epochs=20,
    batch_size=32,
    window_size=60
):

    # Carregando os dados

    df = load_data(
        ticker=ticker
    )

    # Pré processamento

    scaled_data, scaler = preprocess_data(df)

    # Criando as sequencias de dias

    X, y = create_sequences(
        scaled_data,
        window_size=window_size
    )

    # Dividindo os dados em treino e teste

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # Montando o modelo

    model = build_lstm_model(
        input_shape=(X_train.shape[1], 1)
    )

    # Fazendo o treinamento do modelo com os dados de treino

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Utilizando o modelo treinado com os dados de treino para fazer inferências nos dados de teste.

    predictions = model.predict(X_test)

    # Comparando as previsões geradas com os dados de teste e verificando as métricas do modelo treinado

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions,
        squared=False
    )

    mape = mean_absolute_percentage_error(
        y_test,
        predictions
    )

    # Salvando as métricas para expor via API
    metrics = {
        "ticker": ticker,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "epochs": epochs,
        "batch_size": batch_size,
        "window_size": window_size
    }

    # Salvando o modelo no formato pickle

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        "models/lstm_model.keras"
    )

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    with open(
        "models/metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    return metrics