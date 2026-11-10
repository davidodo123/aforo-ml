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
