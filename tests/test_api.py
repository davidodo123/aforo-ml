import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import aforo.api as api_mod
from aforo.features import FEATURES


class ModeloFalso:
    """Devuelve la hora como predicción y comprueba que recibe las features correctas."""

    def predict(self, X):
        assert list(X.columns) == FEATURES
        return np.asarray(X["hora"], dtype=float)


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "RUTA_LOG", tmp_path / "pred.jsonl")
    app = api_mod.crear_app(ModeloFalso(), {"uri": "falso", "run_id": "test"})
    with TestClient(app) as c:
        yield c


def test_health(cliente):
    assert cliente.get("/health").json() == {"estado": "ok"}


def test_predecir_y_log(cliente, tmp_path):
    r = cliente.post("/predecir", json={"horas": [
        {"fecha_hora": "2026-11-14T21:00:00", "temperatura": 15, "evento": True},
        {"fecha_hora": "2026-11-14T09:00:00", "temperatura": 9, "lluvia_mm": 2.5},
    ]})
    assert r.status_code == 200
    preds = r.json()["predicciones"]
    assert [p["clientes"] for p in preds] == [21.0, 9.0]
    lineas = (tmp_path / "pred.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 2 and json.loads(lineas[0])["run_id"] == "test"


def test_rechaza_hora_cerrado(cliente):
    r = cliente.post("/predecir", json={"horas": [
        {"fecha_hora": "2026-11-14T04:00:00", "temperatura": 10},
    ]})
    assert r.status_code == 422


def test_rechaza_peticion_vacia(cliente):
    assert cliente.post("/predecir", json={"horas": []}).status_code == 422


def test_sin_modelo_da_503(monkeypatch):
    def falla(uri):
        raise RuntimeError("no hay champion")

    monkeypatch.setattr(api_mod, "cargar_modelo", falla)
    with TestClient(api_mod.crear_app()) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/predecir", json={"horas": [{"fecha_hora": "2026-11-14T12:00:00", "temperatura": 20}]})
        assert r.status_code == 503
