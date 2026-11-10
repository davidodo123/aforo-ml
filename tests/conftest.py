import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest


@pytest.fixture(autouse=True)
def _directorio_temporal(tmp_path, monkeypatch):
    """MLflow guarda artefactos en ./mlruns: que los tests no ensucien el proyecto."""
    monkeypatch.chdir(tmp_path)
