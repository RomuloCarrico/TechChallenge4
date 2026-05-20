from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout
)

#Responsável por construir a arquitetura LSTM.

def build_lstm_model(input_shape):

    model = Sequential()

    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        LSTM(
            units=50
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(units=1)
    )

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    return model