# aforo-ml

Predicción de **clientes por hora** de un bar-cafetería, con un ciclo de ML en producción completo:
datos → entrenamiento → registro de modelos → API → Docker → tests y CI.

> Los datos son **sintéticos**: los genera `aforo.datos` con patrones realistas (horas punta,
> fines de semana, festivos de Granada, clima, eventos, agosto flojo, tendencia de crecimiento
> y ruido). Así se puede comprobar que el modelo aprende lo que debe.

## Arquitectura

```
aforo.datos ──► datos/afluencia.csv ──► aforo.entrenar ──► MLflow (runs + registro)
                                                               │  alias "champion"
                                       cliente ──► FastAPI ◄───┘
                                                     │
                                                     └──► logs/predicciones.jsonl
```

| Pieza | Qué hace |
|---|---|
| `src/aforo/datos.py` | Genera y limpia los datos sintéticos |
| `src/aforo/features.py` | Features compartidas por entrenamiento y API |
| `src/aforo/entrenar.py` | Baseline + LightGBM Poisson, split temporal, registro en MLflow y promoción a `champion` |
| `src/aforo/api.py` | API FastAPI que sirve el `champion` |

## Resultados (test: últimos 90 días de 2025)

| Modelo | MAE | WAPE |
|---|---|---|
| Baseline (media por día y hora) | 5,97 | — |
| LightGBM Poisson | 4,82 | 15,2 % |
| LightGBM Poisson, `num_leaves=8` | 4,63 | 14,6 % |

## Arrancar en local (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .

python -m aforo.datos                 # genera datos/afluencia.csv
python -m aforo.entrenar              # entrena y registra en mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
uvicorn aforo.api:app --reload        # http://localhost:8000/docs
pytest -q
```

## Arrancar con Docker

```powershell
docker compose up -d mlflow                          # MLflow en http://localhost:5000
docker compose --profile entrenar run --rm entrenar  # genera datos, entrena y registra
docker compose up -d api                             # API en http://localhost:8000/docs
```

Tras reentrenar, la API coge el nuevo `champion` sin reiniciar:

```powershell
curl -X POST http://localhost:8000/recargar
```

## Ejemplo de petición

```json
POST /predecir
{
  "horas": [
    {"fecha_hora": "2026-11-14T21:00:00", "temperatura": 15, "evento": true},
    {"fecha_hora": "2026-11-10T21:00:00", "temperatura": 12, "lluvia_mm": 3}
  ]
}
```

## Limitaciones conocidas

- Los árboles no extrapolan la tendencia: para fechas posteriores a los datos, el crecimiento se queda plano.
- Los festivos están en `festivos.py` solo para 2024-2026.

Plan completo en [`diario/Roadmap.md`](diario/Roadmap.md).
