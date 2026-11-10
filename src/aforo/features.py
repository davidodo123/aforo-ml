"""Construcción de features.

El MISMO código se usa al entrenar y al servir en la API: así evitamos el
"training-serving skew" (que el modelo vea datos distintos en producción).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from aforo.festivos import es_festivo

HORA_APERTURA = 8
HORA_CIERRE = 24  # abierto de 8:00 a 23:59

COLUMNAS_ENTRADA = ["fecha_hora", "temperatura", "lluvia_mm", "evento"]

FEATURES = [
    "hora",
    "dia_semana",
    "mes",
    "dia_del_anio_sin",
    "dia_del_anio_cos",
    "fin_de_semana",
    "festivo",
    "vispera_festivo",
    "agosto",
    "temperatura",
    "lluvia_mm",
    "llueve",
    "evento",
    "dias_desde_inicio",
]

# Fecha de referencia para la tendencia (apertura del local en los datos sintéticos).
FECHA_INICIO = pd.Timestamp("2024-01-01")


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe columnas COLUMNAS_ENTRADA y devuelve solo FEATURES, en orden."""
    faltan = set(COLUMNAS_ENTRADA) - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas de entrada: {sorted(faltan)}")

    fh = pd.to_datetime(df["fecha_hora"])
    fechas = fh.dt.date
    dia_anio = fh.dt.dayofyear

    out = pd.DataFrame(index=df.index)
    out["hora"] = fh.dt.hour.astype("int64")
    out["dia_semana"] = fh.dt.dayofweek.astype("int64")  # 0 = lunes
    out["mes"] = fh.dt.month.astype("int64")
    out["dia_del_anio_sin"] = np.sin(2 * np.pi * dia_anio / 365.25)
    out["dia_del_anio_cos"] = np.cos(2 * np.pi * dia_anio / 365.25)
    out["fin_de_semana"] = (out["dia_semana"] >= 5).astype("int64")
    out["festivo"] = fechas.map(es_festivo).astype("int64")
    out["vispera_festivo"] = fechas.map(lambda d: es_festivo(d + timedelta(days=1))).astype("int64")
    out["agosto"] = (out["mes"] == 8).astype("int64")
    out["temperatura"] = df["temperatura"].astype("float64")
    out["lluvia_mm"] = df["lluvia_mm"].astype("float64").clip(lower=0)
    out["llueve"] = (out["lluvia_mm"] > 0.5).astype("int64")
    out["evento"] = df["evento"].astype("int64")
    out["dias_desde_inicio"] = (fh - FECHA_INICIO).dt.days.astype("float64")
    # Todo float64: el esquema de MLflow acepta nulos y evita errores de tipo al servir.
    return out[FEATURES].astype("float64")


def en_horario(fecha_hora: pd.Series) -> pd.Series:
    horas = pd.to_datetime(fecha_hora).dt.hour
    return (horas >= HORA_APERTURA) & (horas < HORA_CIERRE)
