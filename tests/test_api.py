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


def test_rechaza_anio_sin_festivos(cliente):
    r = cliente.post("/predecir", json={"horas": [
        {"fecha_hora": "2031-05-10T12:00:00", "temperatura": 20},
    ]})
    assert r.status_code == 422
    assert "festivos" in r.text


def test_avisa_si_extrapola(tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "RUTA_LOG", tmp_path / "pred.jsonl")
    meta = {"uri": "falso", "run_id": "test", "train_hasta": "2025-12-31 23:00:00"}
    with TestClient(api_mod.crear_app(ModeloFalso(), meta)) as c:
        r = c.post("/predecir", json={"horas": [
            {"fecha_hora": "2025-06-14T21:00:00", "temperatura": 25},
            {"fecha_hora": "2026-06-14T21:00:00", "temperatura": 25},
        ]})
    assert r.status_code == 200
    avisos = r.json()["avisos"]
    assert len(avisos) == 1 and avisos[0].startswith("1 de 2 horas")


def test_sin_aviso_dentro_del_entrenamiento(cliente):
    r = cliente.post("/predecir", json={"horas": [
        {"fecha_hora": "2026-11-14T21:00:00", "temperatura": 15},
    ]})
    assert r.json()["avisos"] == []


def test_cors_permite_la_web(cliente):
    r = cliente.options("/predecir", headers={
        "Origin": "https://aforo.vercel.app",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "https://aforo.vercel.app")
