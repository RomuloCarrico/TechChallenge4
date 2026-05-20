import time
import json
import os
import psutil

from datetime import datetime
from fastapi import Request

os.makedirs(
    "monitoring",
    exist_ok=True
)

# Monitoramento

async def monitoring_middleware(
    request: Request,
    call_next
):

    start_time = time.time()

    # Executa a request

    response = await call_next(request)

    # Verifica o tempo

    process_time = time.time() - start_time

    # Verifica os recursos utilizados

    cpu_usage = psutil.cpu_percent()

    memory_usage = psutil.virtual_memory().percent

    # Cria o log

    log_data = {
        "timestamp": str(datetime.now()),
        "endpoint": request.url.path,
        "method": request.method,
        "response_time_seconds": round(
            process_time,
            4
        ),
        "cpu_usage_percent": cpu_usage,
        "memory_usage_percent": memory_usage
    }

    # Gera o arquivo de log

    log_file = "monitoring/logs.json"

    logs = []

    if os.path.exists(log_file):

        with open(log_file, "r") as f:

            try:
                logs = json.load(f)
            except:
                logs = []

    logs.append(log_data)

    # Salva o arquivo de log

    with open(log_file, "w") as f:

        json.dump(
            logs,
            f,
            indent=4
        )

    return response