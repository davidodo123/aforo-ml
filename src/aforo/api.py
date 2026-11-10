"""API de predicción de afluencia (FastAPI).

Carga el modelo con alias "champion" del registro de MLflow al arrancar.
Cada predicción se apunta en un log JSONL para poder vigilar el modelo después
(drift, comparar con los clientes reales).

Arranque local:
    uvicorn aforo.api:app --reload
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from aforo import __version__
from aforo.features import HORA_APERTURA, HORA_CIERRE, construir_features
from aforo.festivos import ANIOS_CUBIERTOS

log = logging.getLogger("aforo.api")

MODEL_URI = os.getenv("MODEL_URI", "models:/aforo@champion")
# Si está definida, se carga el modelo exportado (python -m aforo.exportar) sin MLflow.
MODEL_DIR = os.getenv("MODEL_DIR")
RUTA_LOG = Path(os.getenv("RUTA_LOG_PREDICCIONES", "logs/predicciones.jsonl"))


class Hora(BaseModel):
    fecha_hora: datetime = Field(examples=["2026-11-14T21:00:00"])
    temperatura: float = Field(ge=-15, le=50, description="ºC previstos", examples=[17.5])
    lluvia_mm: float = Field(0.0, ge=0, le=100, description="mm/h previstos", examples=[0.0])
    evento: bool = Field(False, description="Partido, concierto u otro evento esa noche")

    @field_validator("fecha_hora")
    @classmethod
    def dentro_de_horario(cls, v: datetime) -> datetime:
        if not HORA_APERTURA <= v.hour < HORA_CIERRE:
            raise ValueError(f"El local abre de {HORA_APERTURA}:00 a {HORA_CIERRE - 1}:59")
        if v.year not in ANIOS_CUBIERTOS:
            raise ValueError(
                f"No hay festivos cargados para {v.year} (años disponibles: "
                f"{min(ANIOS_CUBIERTOS)}-{max(ANIOS_CUBIERTOS)}); añádelos en festivos.py"
            )
        return v


class Peticion(BaseModel):
    horas: list[Hora] = Field(min_length=1, max_length=24 * 14)


class Prediccion(BaseModel):
    fecha_hora: datetime
    clientes: float


class Respuesta(BaseModel):
    modelo: str
    predicciones: list[Prediccion]
    avisos: list[str] = []


def cargar_modelo(uri: str) -> tuple[Any, dict]:
    if MODEL_DIR:
        from aforo.servir import ModeloExportado

        modelo = ModeloExportado(MODEL_DIR)
        claves = ("uri", "run_id", "version", "train_hasta")
        return modelo, {k: modelo.meta.get(k) for k in claves}

    import mlflow  # solo en el modo MLflow: la imagen ligera no lo instala

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    modelo = mlflow.pyfunc.load_model(uri)
    run_id = modelo.metadata.run_id
    meta = {"uri": uri, "run_id": run_id}
    try:  # hasta qué fecha vio datos el modelo, para avisar de extrapolaciones
        meta["train_hasta"] = mlflow.MlflowClient().get_run(run_id).data.params.get("train_hasta")
    except Exception as e:
        log.warning("No se pudo leer train_hasta del run %s: %s", run_id, e)
    return modelo, meta


def avisos_extrapolacion(horas: list[datetime], meta: dict) -> list[str]:
    """Avisa si se pide más allá de los datos de entrenamiento (extrapolación)."""
    if not meta.get("train_hasta"):
        return []
    limite = pd.Timestamp(meta["train_hasta"])
    fuera = sum(pd.Timestamp(h).replace(tzinfo=None) > limite for h in horas)
    if not fuera:
        return []
    return [
        f"{fuera} de {len(horas)} horas son posteriores a los datos de entrenamiento "
        f"({limite:%Y-%m-%d}): la tendencia de crecimiento se extrapola y la "
        "incertidumbre es mayor cuanto más lejos esté la fecha."
    ]


def crear_app(modelo: Any = None, meta: dict | None = None) -> FastAPI:
    """`modelo` permite inyectar uno falso en los tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.modelo, app.state.meta = modelo, meta or {}
        if app.state.modelo is None:
            try:
                app.state.modelo, app.state.meta = cargar_modelo(MODEL_URI)
                log.info("Modelo cargado: %s", app.state.meta)
            except Exception as e:  # la API arranca igual; /health avisa
                log.error("No se pudo cargar el modelo %s: %s", MODEL_URI, e)
        yield

    app = FastAPI(title="Aforo", version=__version__, lifespan=lifespan,
                  description="Predicción de clientes por hora de un local.")

    @app.get("/", include_in_schema=False)
    def inicio():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        if app.state.modelo is None:
            raise HTTPException(503, "Modelo no cargado")
        return {"estado": "ok"}

    @app.get("/modelo")
    def info_modelo():
        return {"version_api": __version__, **app.state.meta}

    @app.post("/recargar")
    def recargar():
        """Vuelve a cargar el modelo (el "champion" de MLflow o la carpeta exportada) sin reiniciar."""
        try:
            app.state.modelo, app.state.meta = cargar_modelo(MODEL_URI)
        except Exception as e:
            raise HTTPException(503, f"No se pudo cargar {MODEL_URI}: {e}") from e
        return app.state.meta

    @app.post("/predecir", response_model=Respuesta)
    def predecir(p: Peticion):
        if app.state.modelo is None:
            raise HTTPException(503, "Modelo no cargado")
        entrada = pd.DataFrame([h.model_dump() for h in p.horas])
        t0 = time.perf_counter()
        pred = app.state.modelo.predict(construir_features(entrada))
        ms = (time.perf_counter() - t0) * 1000

        salida = [
            Prediccion(fecha_hora=h.fecha_hora, clientes=round(max(float(c), 0.0), 1))
            for h, c in zip(p.horas, pred)
        ]
        _guardar_log(p, salida, app.state.meta, ms)
        return Respuesta(
            modelo=app.state.meta.get("uri", "desconocido"),
            predicciones=salida,
            avisos=avisos_extrapolacion([h.fecha_hora for h in p.horas], app.state.meta),
        )

    return app


def _guardar_log(p: Peticion, salida: list[Prediccion], meta: dict, ms: float) -> None:
    try:
        RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
        ahora = datetime.now(timezone.utc).isoformat()
        with RUTA_LOG.open("a", encoding="utf-8") as f:
            for h, s in zip(p.horas, salida):
                fila = {"ts": ahora, "run_id": meta.get("run_id"), "ms": round(ms, 2),
                        **h.model_dump(mode="json"), "pred": s.clientes}
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("No se pudo escribir el log de predicciones: %s", e)


app = crear_app()
