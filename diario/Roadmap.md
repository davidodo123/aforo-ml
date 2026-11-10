# Roadmap · aforo-ml

Predicción de clientes por hora de un bar-cafetería (datos sintéticos) con un ciclo MLOps completo.
Leyenda: [x] hecho · [/] a medias · [ ] pendiente · [-] descartado

## Fase 1 · Base (MVP)
- [x] Generador de datos sintéticos con patrones realistas (hora, día, festivos, clima, eventos, agosto, tendencia)
- [x] Limpieza de datos (huecos de sensor y lecturas imposibles)
- [x] Features compartidas entre entrenamiento y API (sin training-serving skew)
- [x] Baseline (media por día y hora) y LightGBM Poisson con split temporal
- [x] Registro en MLflow (parámetros, métricas, modelo, alias `champion`)
- [x] API FastAPI: `/health`, `/modelo`, `/predecir`, `/recargar`
- [x] Log de predicciones en JSONL
- [x] Dockerfile y docker-compose (MLflow + entrenar + API)
- [x] Tests con pytest y CI en GitHub Actions
- [ ] Probar la pila con Docker en el PC (`docker compose`)
- [x] Subir el repo a GitHub

## Fase 2 · Calidad del modelo
- [x] Promoción justa: el `champion` se reevalúa sobre el mismo test que el modelo nuevo
- [x] La API rechaza años sin festivos cargados y avisa cuando extrapola más allá del entrenamiento
- [x] Reentrenar el modelo final con todos los datos, con comparación justa frente al `champion`
- [ ] Validación cruzada temporal (TimeSeriesSplit) en lugar de un solo corte
- [ ] Búsqueda de hiperparámetros (Optuna) registrada en MLflow
- [ ] Features de retardo (misma hora hace 7 días, media móvil) y cómo servirlas en la API
- [x] Arreglar la tendencia: tendencia log-lineal aparte y LightGBM sobre el resto (MAE 4,82 → 4,74)
- [ ] Intervalos de predicción (cuantiles) además del valor medio

## Fase 3 · Producción de verdad
- [x] Imagen ligera de la API sin MLflow (modelo exportado en `modelo/`) y `render.yaml`
- [ ] Desplegar la API en Render (falta conectar el repo)
- [ ] Previsión meteorológica real (AEMET u Open-Meteo) para predecir los próximos 7 días
- [ ] Endpoint `/predecir/dia` que construya las 16 horas de un día a partir de la previsión
- [ ] Autenticación por API key y rate limiting

## Fase 4 · Monitorización y reentrenamiento
- [ ] Endpoint para registrar los clientes reales y calcular el error en producción
- [ ] Detección de drift en las entradas (Evidently) sobre el log de predicciones
- [ ] Reentrenamiento programado (GitHub Actions cron) que solo promueve si mejora al `champion`
- [ ] Panel con métricas del modelo en producción

## Fase 5 · Portfolio
- [ ] README con capturas de MLflow y `/docs`
- [ ] Post de LinkedIn explicando el ciclo MLOps
- [ ] Usarlo como base para las prácticas del módulo de Despliegue
