"""Modelo de afluencia = tendencia (log-lineal) × LightGBM.

Los árboles no extrapolan: si `dias_desde_inicio` es una feature del árbol, cualquier
fecha posterior al entrenamiento cae en la última hoja y el crecimiento se queda
plano. Por eso la tendencia se separa:

1. Una regresión de Poisson estima la pendiente de crecimiento controlando hora,
   día, mes, festivos, clima y eventos (si no, la estacionalidad se confunde con
   la tendencia).
2. LightGBM (objetivo Poisson, enlace log) aprende el resto, con la tendencia como
   `init_score`: log(clientes) = árbol(x) + pendiente · años.
3. Al predecir se suman ambas partes, así que la tendencia sí se extrapola.

El modelo recibe las mismas FEATURES que antes, así la API no cambia.
"""

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from aforo.features import FEATURES

FEATURES_ARBOL = [f for f in FEATURES if f != "dias_desde_inicio"]

# Con menos de un año de datos la pendiente se confunde con la estacionalidad.
DIAS_MINIMOS_TENDENCIA = 365


def _anios(X: pd.DataFrame) -> np.ndarray:
    return X["dias_desde_inicio"].to_numpy(dtype=float) / 365.25


def _diseno_tendencia(X: pd.DataFrame) -> pd.DataFrame:
    """Variables de control para que la pendiente mida solo el crecimiento."""
    temp = pd.cut(X["temperatura"], [-np.inf, 6, 18, 28, 33, np.inf]).astype(str)
    controles = pd.get_dummies(
        pd.DataFrame({
            "hora": X["hora"].astype(int).astype(str),
            "dia": X["dia_semana"].astype(int).astype(str),
            "mes": X["mes"].astype(int).astype(str),
            "temp": temp,
        }),
        drop_first=True,
        dtype=float,
    )
    for c in ["festivo", "vispera_festivo", "llueve", "evento"]:
        controles[c] = X[c].to_numpy(dtype=float)
    controles["anios"] = _anios(X)
    return controles


def estimar_pendiente(X: pd.DataFrame, y: pd.Series) -> float:
    """Crecimiento anual en escala log (0,095 ≈ +10 % al año)."""
    if X["dias_desde_inicio"].max() - X["dias_desde_inicio"].min() < DIAS_MINIMOS_TENDENCIA:
        return 0.0
    D = _diseno_tendencia(X)
    glm = PoissonRegressor(alpha=1e-6, max_iter=1000).fit(D, y)
    return float(glm.coef_[list(D.columns).index("anios")])


class ModeloConTendencia(mlflow.pyfunc.PythonModel):
    def __init__(self, arbol: lgb.LGBMRegressor, pendiente: float):
        self.arbol = arbol
        self.pendiente = pendiente

    @classmethod
    def entrenar(cls, X: pd.DataFrame, y: pd.Series, params: dict) -> "ModeloConTendencia":
        pendiente = estimar_pendiente(X, y)
        arbol = lgb.LGBMRegressor(**params)
        arbol.fit(X[FEATURES_ARBOL], y, init_score=pendiente * _anios(X))
        return cls(arbol, pendiente)

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        crudo = self.arbol.predict(X[FEATURES_ARBOL], raw_score=True)
        return np.exp(crudo + self.pendiente * _anios(X))

    def predict(self, context, model_input, params=None):
        return self.predecir(model_input)

    def importancias(self) -> pd.Series:
        return pd.Series(self.arbol.feature_importances_, index=FEATURES_ARBOL).sort_values(ascending=False)
