"""Modelo exportado para servir SIN MLflow ni scikit-learn.

`python -m aforo.exportar` guarda el champion en una carpeta con:
- arbol.txt: el booster de LightGBM en su formato nativo.
- meta.json: pendiente de la tendencia, features del árbol y de dónde salió.

Así la imagen de la API solo necesita LightGBM, pandas y FastAPI (unos 300 MB
en lugar de más de 1 GB con MLflow) y no depende de un servidor MLflow.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ARCHIVO_ARBOL = "arbol.txt"
ARCHIVO_META = "meta.json"


class ModeloExportado:
    def __init__(self, carpeta: str | Path):
        carpeta = Path(carpeta)
        self.meta = json.loads((carpeta / ARCHIVO_META).read_text(encoding="utf-8"))
        self.arbol = lgb.Booster(model_file=str(carpeta / ARCHIVO_ARBOL))
        self.features_arbol = self.meta["features_arbol"]
        self.pendiente = float(self.meta["pendiente"])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        crudo = self.arbol.predict(X[self.features_arbol], raw_score=True)
        anios = X["dias_desde_inicio"].to_numpy(dtype=float) / 365.25
        return np.exp(crudo + self.pendiente * anios)
