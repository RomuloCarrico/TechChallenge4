import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Função de preprocessamento dos dados, carregando os dados da Petrobras, usando o yfinance, entre os dias 01/01/2018 e 01/01/2025.
# Então é feita a normalização dos dados usando MinMaxScaler. É criado sequências usando 60 dias como padrão para previsões do 61º dia, e feita a formatação para uma matriz de 3 dimensões, para ser usado no LTSM
# E então a divisão dos dados entre treino e teste, retirando o shuffle da função para não ter problemas de janelas aleatórias do passado e do futuro misturadas.

def load_data(
    filepath="data/petr4.csv"
):

    df = pd.read_csv(
        filepath,
    )

    # Remove linhas nulas
    df = df.dropna()

    if df.empty:

        raise ValueError(
            "Dataset vazio."
        )

    return df


def preprocess_data(df):

    data = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler


def create_sequences(data, window_size=60):

    X = []
    y = []

    for i in range(window_size, len(data)):

        X.append(
            data[i-window_size:i, 0]
        )

        y.append(
            data[i, 0]
        )

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(
        X,
        (X.shape[0], X.shape[1], 1)
    )

    return X, y


def split_data(X, y, pct_teste=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pct_teste, shuffle=False)

    return X_train, X_test, y_train, y_test