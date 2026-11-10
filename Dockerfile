# Imagen ligera de la API para desplegar: sirve el modelo exportado en modelo/
# (python -m aforo.exportar) sin MLflow. La pila completa con MLflow usa Dockerfile.mlflow.
FROM python:3.12-slim

# LightGBM necesita OpenMP en tiempo de ejecución.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/modelo \
    PORT=8000

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY modelo ./modelo

RUN useradd --create-home aforo && mkdir -p /app/logs && chown aforo /app/logs
USER aforo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"

# Render, Railway y similares indican el puerto en $PORT.
CMD ["sh", "-c", "uvicorn aforo.api:app --host 0.0.0.0 --port ${PORT}"]
