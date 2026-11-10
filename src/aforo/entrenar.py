"""Entrena el modelo de afluencia y lo registra en MLflow.

Pasos:
1. Carga y limpia los datos.
2. Split TEMPORAL (nunca aleatorio en series temporales): los últimos N días son test.
3. Baseline: media histórica por (día de la semana, hora). Si el modelo no la
   supera, no merece producción.
4. Tendencia log-lineal × LightGBM Poisson (ver aforo.modelo), evaluado en test.
5. Compara con el champion actual de forma justa (ver comparar_con_campeon).
6. Reentrena el modelo final con TODOS los datos y lo registra en MLflow. Si el
   candidato mejora al baseline y al champion, lo marca con el alias "champion"
   (el que sirve la API).

Uso:
    python -m aforo.entrenar --datos datos/afluencia.csv
"""

import argparse
import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from aforo.datos import limpiar
from aforo.features import FEATURES, construir_features
from aforo.modelo import ModeloConTendencia

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
    clave_tr = [train["fecha_hora"].dt.day_of_week, train["fecha_hora"].dt.hour]
    medias = train.groupby(clave_tr)["clientes"].mean()
    claves = zip(test["fecha_hora"].dt.day_of_week, test["fecha_hora"].dt.hour)
    return np.array([medias.get(k, train["clientes"].mean()) for k in claves])


def huella_datos(df: pd.DataFrame) -> str:
    """Identifica el dataset: dos runs con la misma huella usaron los mismos datos."""
    return format(int(pd.util.hash_pandas_object(df, index=False).sum()) & (2**64 - 1), "016x")


def comparar_con_campeon(
    test: pd.DataFrame, X_te: pd.DataFrame, pred_candidato: np.ndarray, huella: str
) -> dict:
    """Compara el candidato con el "champion" actual sin dar ventaja a ninguno.

    El champion se reentrenó con todos sus datos, así que puede haber visto parte
    de nuestro test. Se compara así:
    - Si al menos la mitad del test es posterior a lo que vio el champion, ambos se
      evalúan sobre esas horas no vistas.
    - Si no, pero el champion se evaluó con los mismos datos y la misma ventana de
      test, se usa su métrica guardada (caso típico: probar hiperparámetros).
    - Si no hay forma justa de comparar, no se promueve automáticamente.
    """
    cliente = MlflowClient()
    try:
        version = cliente.get_model_version_by_alias(NOMBRE_MODELO, "champion")
    except mlflow.exceptions.MlflowException:
        return {"estado": "sin_campeon"}
    run = cliente.get_run(version.run_id)
    p, m = run.data.params, run.data.metrics

    if p.get("train_hasta"):
        no_visto = (test["fecha_hora"] > pd.Timestamp(p["train_hasta"])).to_numpy()
    else:
        no_visto = np.ones(len(test), dtype=bool)

    if no_visto.mean() >= 0.5:
        try:
            campeon = mlflow.pyfunc.load_model(f"models:/{NOMBRE_MODELO}@champion")
            pred_campeon = campeon.predict(X_te[no_visto])
        except Exception as e:
            print(f"Aviso: el champion no puede predecir con las features actuales ({e}).")
            return {"estado": "campeon_incompatible"}
        y = test["clientes"].to_numpy()[no_visto]
        return {
            "estado": "comparable",
            "criterio": f"reevaluados en {int(no_visto.sum())} horas que el champion no vio",
            "campeon_mae": metricas(y, pred_campeon)["mae"],
            "candidato_mae": metricas(y, pred_candidato[no_visto])["mae"],
        }

    mismo_holdout = (
        p.get("huella_datos") == huella
        and p.get("test_desde") == str(test["fecha_hora"].min())
        and "test_mae" in m
    )
    if mismo_holdout:
        return {
            "estado": "comparable",
            "criterio": "mismos datos y mismo test: se usa el MAE guardado del champion",
            "campeon_mae": m["test_mae"],
            "candidato_mae": metricas(test["clientes"], pred_candidato)["mae"],
        }
    return {"estado": "no_comparable"}


def entrenar(ruta_datos: str, dias_test: int = 90, params: dict | None = None) -> dict:
    df = pd.read_csv(ruta_datos, parse_dates=["fecha_hora"])
    df = limpiar(df)
    train, test = split_temporal(df, dias_test)
    huella = huella_datos(df)

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
            "test_desde": str(test["fecha_hora"].min()),
            "eval_train_hasta": str(train["fecha_hora"].max()),
            # El modelo registrado se reentrena con TODO: esto es lo que ha visto.
            "train_hasta": str(df["fecha_hora"].max()),
            "huella_datos": huella,
            "features": ",".join(FEATURES),
        })

        # 1. Evaluación honesta: entrenar sin el test.
        base = metricas(y_te, predecir_baseline(train, test))
        mlflow.log_metrics({f"baseline_{k}": v for k, v in base.items()})

        candidato = ModeloConTendencia.entrenar(X_tr, y_tr, params)
        pred = candidato.predecir(X_te)
        res = metricas(y_te, pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in res.items()})
        mlflow.log_metric("mejora_vs_baseline_pct", 100 * (1 - res["mae"] / base["mae"]))

        # 2. Comparación justa con el champion.
        comp = comparar_con_campeon(test, X_te, pred, huella)
        mlflow.set_tag("comparacion", comp["estado"])
        if comp["estado"] == "comparable":
            mlflow.log_metrics({
                "campeon_mae": comp["campeon_mae"],
                "candidato_mae_vs_campeon": comp["candidato_mae"],
            })
            mlflow.set_tag("criterio_comparacion", comp["criterio"])

        # 3. Modelo final: mismo método, con todos los datos.
        final = ModeloConTendencia.entrenar(construir_features(df), df["clientes"], params)
        crecimiento = 100 * (np.exp(final.pendiente) - 1)
        mlflow.log_metrics({"pendiente_anual_log": final.pendiente, "crecimiento_anual_pct": crecimiento})
        mlflow.log_dict(final.importancias().to_dict(), "importancia_features.json")

        info = mlflow.pyfunc.log_model(
            name="modelo",
            python_model=final,
            signature=infer_signature(X_te, pred),
            input_example=X_te.head(3),
            registered_model_name=NOMBRE_MODELO,
        )

        version = info.registered_model_version
        supera_baseline = res["mae"] < base["mae"]
        if comp["estado"] in ("sin_campeon", "campeon_incompatible"):
            promovido = supera_baseline
        elif comp["estado"] == "comparable":
            promovido = supera_baseline and comp["candidato_mae"] < comp["campeon_mae"]
        else:
            print("Aviso: no hay una comparación justa con el champion; no se promueve automáticamente.")
            promovido = False
        if promovido:
            MlflowClient().set_registered_model_alias(NOMBRE_MODELO, "champion", version)
        mlflow.set_tag("promovido", str(promovido))

    return {
        "run_id": run.info.run_id,
        "version": version,
        "baseline_mae": round(base["mae"], 2),
        "test_mae": round(res["mae"], 2),
        "test_wape": round(res["wape"], 4),
        "crecimiento_anual_pct": round(crecimiento, 1),
        "comparacion": comp["estado"],
        "mae_campeon_anterior": round(comp["campeon_mae"], 2) if "campeon_mae" in comp else None,
        "mae_candidato_vs_campeon": round(comp["candidato_mae"], 2) if "candidato_mae" in comp else None,
        "promovido": promovido,
    }


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
        f"WAPE {r['test_wape']:.1%}, crecimiento {r['crecimiento_anual_pct']} %/año, "
        f"champion={'sí' if r['promovido'] else 'no'} ({r['comparacion']})"
    )


if __name__ == "__main__":
    main()
