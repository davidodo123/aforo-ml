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
| `src/aforo/modelo.py` | Tendencia log-lineal (regresión de Poisson) × LightGBM Poisson, para que el crecimiento se extrapole |
| `src/aforo/entrenar.py` | Baseline, evaluación con split temporal, comparación justa con el `champion`, reentrenamiento final con todos los datos y registro en MLflow |
| `src/aforo/api.py` | API FastAPI que sirve el `champion` (desde MLflow o desde `modelo/`) |
| `src/aforo/exportar.py` | Exporta el `champion` a `modelo/` (árbol LightGBM + `meta.json`) para servirlo sin MLflow |

## Resultados (test: últimos 90 días de 2025)

| Modelo | MAE | WAPE |
|---|---|---|
| Baseline (media por día y hora) | 5,97 | — |
| LightGBM Poisson | 4,82 | 15,2 % |
| LightGBM Poisson, `num_leaves=8` | 4,63 | 14,6 % |
| Tendencia × LightGBM Poisson | 4,74 | 15,0 % |

El modelo con tendencia separada estima un crecimiento del 9,8 % anual (el generador usa un 10 %).
Ejemplo, sábado a las 21:00 con evento: el modelo anterior predecía 92 → 91 → 91 clientes en
2024 → 2025 → 2026; el nuevo, 88 → 97 → 106.

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

## Desplegar (imagen ligera, sin MLflow)

Hay dos imágenes:

| Archivo | Para qué | Tamaño aprox. de dependencias |
|---|---|---|
| `Dockerfile` | API pública: sirve el modelo exportado en `modelo/` | ~300 MB |
| `Dockerfile.mlflow` | Pila completa de `docker-compose` (MLflow, entrenar, API) | ~730 MB |

Tras entrenar un modelo mejor, se exporta y se sube al repo:

```powershell
python -m aforo.exportar --salida modelo
git add modelo && git commit -m "Nuevo champion" && git push
```

**Render**: New → Blueprint → elige el repo. `render.yaml` crea el servicio desde el `Dockerfile`
y cada push a `main` lo redespliega. La raíz `/` redirige a `/docs`.

En el plan gratuito el servicio se duerme tras un rato sin visitas (la primera petición tarda
unos segundos) y `logs/predicciones.jsonl` se pierde en cada redespliegue.

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

- La tendencia se modela como log-lineal (crecimiento porcentual constante). Para fechas posteriores
  a los datos se extrapola, y la API lo indica en el campo `avisos` de la respuesta.
- Con menos de un año de datos no se estima tendencia (se confundiría con la estacionalidad).
- Los festivos están en `festivos.py` solo para 2024-2026; la API rechaza (422) fechas de otros años.
- El modelo registrado se reentrena con todos los datos tras evaluarlo. Para decidir si pasa a
  `champion`, se compara con el actual solo en horas que este no vio; si no las hay, con su MAE
  guardado cuando los datos y el test son los mismos; y si tampoco, no se promueve automáticamente.

Plan completo en [`diario/Roadmap.md`](diario/Roadmap.md).
