#FROM tensorflow/tensorflow:2.21.0
FROM tensorflow/tensorflow:2.21.0-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Bioalgoritmo/ ./Bioalgoritmo/
COPY ManejoDeDatos/ ./ManejoDeDatos/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app/Bioalgoritmo
CMD ["python", "main.py"]
