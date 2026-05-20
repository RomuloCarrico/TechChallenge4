import os
import joblib
import json

#Script de treinamento do modelo 

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from preprocess import (
    load_data,
    preprocess_data,
    create_sequences,
    split_data
)

from model import build_lstm_model


# Importando os dados

df = load_data()

# Fazendo o pré processamento dos dados

scaled_data, scaler = preprocess_data(df)

# Criando as sequencias de dias

X, y = create_sequences(
    scaled_data,
    window_size=60
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
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
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

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")

# Salvando as métricas para expor via API
metrics = {
    "mae": float(mae),
    "rmse": float(rmse),
    "mape": float(mape),
    "epochs": 20,
    "batch_size": 32
}
with open("models/metrics.json", "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# Salvando o modelo no formato pickle

os.makedirs("models", exist_ok=True)

model.save(
    "models/lstm_model.keras"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Modelo salvo com sucesso.")