import mlflow

from aforo.datos import generar
from aforo.entrenar import NOMBRE_MODELO, entrenar


def test_entrena_supera_baseline_y_promueve(tmp_path):
    ruta = tmp_path / "afluencia.csv"
    generar("2024-01-01", "2024-12-31", semilla=7).to_csv(ruta, index=False)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    r = entrenar(str(ruta), dias_test=30, params={"n_estimators": 150})

    assert r["test_mae"] < r["baseline_mae"]
    assert r["promovido"] is True
    version = mlflow.MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion")
    assert str(version.version) == str(r["version"])


def test_compara_con_campeon_en_el_mismo_test(tmp_path):
    """El campeón se reevalúa sobre el test del modelo nuevo, no con su MAE guardado."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    ruta_a = tmp_path / "a.csv"
    generar("2024-01-01", "2024-12-31", semilla=7).to_csv(ruta_a, index=False)
    r1 = entrenar(str(ruta_a), dias_test=30, params={"n_estimators": 150})
    assert r1["promovido"] is True

    # Datos más largos: el test del segundo entrenamiento es otro periodo.
    ruta_b = tmp_path / "b.csv"
    generar("2024-01-01", "2025-03-31", semilla=7).to_csv(ruta_b, index=False)
    r2 = entrenar(str(ruta_b), dias_test=30, params={"n_estimators": 3})

    assert r2["mae_campeon_anterior"] is not None
    assert r2["mae_campeon_anterior"] != r1["test_mae"]  # reevaluado, no copiado
    assert r2["promovido"] is False  # 3 árboles no ganan al campeón
    version = mlflow.MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion")
    assert str(version.version) == str(r1["version"])
