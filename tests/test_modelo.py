import numpy as np
import pandas as pd

from aforo.datos import generar
from aforo.features import construir_features
from aforo.modelo import ModeloConTendencia, estimar_pendiente


def _datos(desde, hasta):
    df = generar(desde, hasta, semilla=5, con_ruido_calidad=False)
    return construir_features(df), df["clientes"]


def test_recupera_la_tendencia_del_generador():
    X, y = _datos("2024-01-01", "2025-12-31")
    crecimiento = np.exp(estimar_pendiente(X, y)) - 1
    assert 0.06 < crecimiento < 0.14  # el generador crece ~10 %/año


def test_sin_tendencia_con_menos_de_un_anio():
    X, y = _datos("2024-01-01", "2024-08-31")
    assert estimar_pendiente(X, y) == 0.0


def test_extrapola_el_crecimiento():
    X, y = _datos("2024-01-01", "2025-06-30")
    modelo = ModeloConTendencia.entrenar(X, y, {"objective": "poisson", "n_estimators": 100, "verbose": -1})
    entrada = pd.DataFrame({
        "fecha_hora": pd.to_datetime(["2025-03-15 21:00", "2026-03-14 21:00"]),  # sábados
        "temperatura": [15.0, 15.0], "lluvia_mm": [0.0, 0.0], "evento": [False, False],
    })
    hoy, dentro_de_un_anio = modelo.predecir(construir_features(entrada))
    assert dentro_de_un_anio > hoy * 1.05
