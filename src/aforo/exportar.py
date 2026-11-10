"""Exporta el champion de MLflow a una carpeta que la API carga sin MLflow.

Uso:
    python -m aforo.exportar --salida modelo
"""

import argparse
import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from aforo.entrenar import NOMBRE_MODELO
from aforo.modelo import ModeloConTendencia
from aforo.servir import ARCHIVO_ARBOL, ARCHIVO_META


def exportar(salida: str | Path, alias: str = "champion") -> dict:
    uri = f"models:/{NOMBRE_MODELO}@{alias}"
    modelo = mlflow.pyfunc.load_model(uri).unwrap_python_model()
    if not isinstance(modelo, ModeloConTendencia):
        raise TypeError(f"{uri} no es un ModeloConTendencia; reentrena con la versión actual")

    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, alias)
    params = MlflowClient().get_run(version.run_id).data.params

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    modelo.arbol.booster_.save_model(str(salida / ARCHIVO_ARBOL))
    meta = {
        "uri": f"{uri} (v{version.version}, exportado)",
        "run_id": version.run_id,
        "version": str(version.version),
        "train_hasta": params.get("train_hasta"),
        "pendiente": modelo.pendiente,
        "features_arbol": modelo.arbol.booster_.feature_name(),
    }
    (salida / ARCHIVO_META).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta el champion para servirlo sin MLflow")
    p.add_argument("--salida", default="modelo")
    p.add_argument("--alias", default="champion")
    a = p.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    meta = exportar(a.salida, a.alias)
    print(f"Exportado {meta['uri']} -> {a.salida}/")


if __name__ == "__main__":
    main()
