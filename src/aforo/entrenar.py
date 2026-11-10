"""Entrena el modelo de afluencia y lo registra en MLflow.

Pasos:
1. Carga y limpia los datos.
2. Split TEMPORAL (nunca aleatorio en series temporales): los últimos N días son test.
3. Baseline: media histórica por (día de la semana, hora). Si el modelo no la
   supera, no merece producción.
4. LightGBM con objetivo Poisson (adecuado para conteos de clientes).
5. Registra parámetros, métricas y modelo en MLflow. Si mejora al baseline y al
   campeón actual, lo marca con el alias "champion" (el que sirve la API).

Uso:
    python -m aforo.entrenar --datos datos/afluencia.csv
"""

import argparse
import os

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from aforo.datos import limpiar
from aforo.features import FEATURES, construir_features

NOMBRE_MODELO = "aforo"
EXPERIMENTO = "aforo-demanda-local"


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_real
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        # WAPE: error absoluto total / clientes totales. Más estable que MAPE con valores bajos.
        "wape": float(np.sum(np.abs(err)) / np.sum(y_real)),
        "sesgo": float(np.mean(err)),
    }


def split_temporal(df: pd.DataFrame, dias_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    corte = df["fecha_hora"].max().normalize() - pd.Timedelta(days=dias_test - 1)
    return df[df["fecha_hora"] < corte], df[df["fecha_hora"] >= corte]


def predecir_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    clave_tr = [train["fecha_hora"].dt.dayofweek, train["fecha_hora"].dt.hour]
    medias = train.groupby(clave_tr)["clientes"].mean()
    claves = zip(test["fecha_hora"].dt.dayofweek, test["fecha_hora"].dt.hour)
    return np.array([medias.get(k, train["clientes"].mean()) for k in claves])


def mae_campeon(cliente: MlflowClient) -> float | None:
    try:
        version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    run = cliente.get_run(version.run_id)
    return run.data.metrics.get("test_mae")


def entrenar(ruta_datos: str, dias_test: int = 90, params: dict | None = None) -> dict:
    df = pd.read_csv(ruta_datos, parse_dates=["fecha_hora"])
    df = limpiar(df)
    train, test = split_temporal(df, dias_test)

    params = {
        "objective": "poisson",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_estimators": 600,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbose": -1,
        **(params or {}),
    }

    X_tr, y_tr = construir_features(train), train["clientes"]
    X_te, y_te = construir_features(test), test["clientes"]

    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_params({
            "dias_test": dias_test,
            "filas_train": len(train),
            "filas_test": len(test),
            "train_hasta": str(train["fecha_hora"].max()),
            "features": ",".join(FEATURES),
        })

        base = metricas(y_te, predecir_baseline(train, test))
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base.items()})

        modelo = lgb.LGBMRegressor(**params)
        modelo.fit(X_tr, y_tr)
        pred = modelo.predict(X_te)
        res = metricas(y_te, pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in res.items()})
        mlflow.log_metric("mejora_vs_baseline_pct", 100 * (1 - res["mae"] / base["mae"]))

        importancias = pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)
        mlflow.log_dict(importancias.to_dict(), "importancia_features.json")

        info = mlflow.lightgbm.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(X_te, pred),
            input_example=X_te.head(3),
            registered_model_name=NOMBRE_MODELO,
        )

        cliente = MlflowClient()
        version = info.registered_model_version
        previo = mae_campeon(cliente)
        supera_baseline = res["mae"] < base["mae"]
        supera_campeon = previo is None or res["mae"] < previo
        promovido = supera_baseline and supera_campeon
        if promovido:
            cliente.set_registered_model_alias(NOMBRE_MODELO, "champion", version)
        mlflow.set_tag("promovido", str(promovido))

    resumen = {
        "run_id": run.info.run_id,
        "version": version,
        "baseline_mae": round(base["mae"], 2),
        "test_mae": round(res["mae"], 2),
        "test_wape": round(res["wape"], 4),
        "mae_campeon_anterior": previo,
        "promovido": promovido,
    }
    return resumen


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena y registra el modelo de afluencia")
    p.add_argument("--datos", default="datos/afluencia.csv")
    p.add_argument("--dias-test", type=int, default=90)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--num-leaves", type=int)
    a = p.parse_args()

    extra = {k: v for k, v in {"learning_rate": a.learning_rate, "num_leaves": a.num_leaves}.items() if v}
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    r = entrenar(a.datos, a.dias_test, extra)
    print(
        f"v{r['version']}: MAE {r['test_mae']} (baseline {r['baseline_mae']}), "
        f"WAPE {r['test_wape']:.1%}, champion={'sí' if r['promovido'] else 'no'}"
    )


if __name__ == "__main__":
    main()
