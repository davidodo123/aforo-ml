import mlflow
import numpy as np
import pandas as pd

from aforo.datos import generar
from aforo.entrenar import NOMBRE_MODELO, entrenar
from aforo.exportar import exportar
from aforo.features import construir_features
from aforo.servir import ModeloExportado


def test_exportado_predice_igual_que_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    ruta = tmp_path / "a.csv"
    generar("2024-01-01", "2024-12-31", semilla=7).to_csv(ruta, index=False)
    entrenar(str(ruta), dias_test=30, params={"n_estimators": 150})

    meta = exportar(tmp_path / "modelo")
    exportado = ModeloExportado(tmp_path / "modelo")
    original = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@champion")

    X = construir_features(pd.read_csv(ruta, parse_dates=["fecha_hora"]).tail(200))
    np.testing.assert_allclose(exportado.predict(X), original.predict(X), rtol=1e-9)
    assert meta["train_hasta"].startswith("2024-12-31")
