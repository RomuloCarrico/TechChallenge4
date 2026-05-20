import numpy as np
import joblib

from tensorflow.keras.models import load_model

# CARREGAMENTO

model = load_model(
    "models/lstm_model.keras"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

WINDOW_SIZE = 60


# FUNÇÃO DE PREVISÃO

def predict_next_price(prices):

    # VALIDAÇÃO

    if len(prices) != WINDOW_SIZE:

        raise ValueError(
            f"São necessários exatamente "
            f"{WINDOW_SIZE} preços."
        )

    # PREPARAÇÃO

    data = np.array(prices)

    data = data.reshape(-1, 1)

    scaled_data = scaler.transform(data)

    X = np.array([
        scaled_data[:, 0]
    ])

    X = np.reshape(
        X,
        (X.shape[0], X.shape[1], 1)
    )

    # PREDIÇÃO

    prediction = model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    return float(prediction[0][0])