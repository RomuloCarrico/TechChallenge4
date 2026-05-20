import os
import yfinance as yf

ticker = "PETR4.SA"

start_date = "2018-01-01"
end_date = "2025-01-01"

# Baixando os dados

df = yf.download(
    "PETR4.SA",
    start="2018-01-01",
    end="2025-01-01",
    auto_adjust=False,
    progress=False
)
# Salvando os dados localmente

os.makedirs(
    "data",
    exist_ok=True
)

df.columns = df.columns.map(
    lambda x: x[0] if isinstance(x, tuple) else x
)

df.to_csv(
    "data/petr4.csv"
)

print("Dados salvos com sucesso.")