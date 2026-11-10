# Decisiones · aforo-ml

## 2026-11-10 · Tema y stack
- **Tema:** demanda (clientes por hora) de un local, en vez de datos del desguace o del olivar.
- **Datos:** sintéticos, generados con `aforo.datos`, con patrones conocidos para poder comprobar que el modelo los aprende. En el README se indica que son inventados.
- **Stack:** FastAPI + Docker + MLflow (opción estándar de la industria y alineada con el módulo de Despliegue).

## 2026-11-10 · Modelo
- **LightGBM con objetivo Poisson:** los clientes son conteos (enteros, no negativos, varianza que crece con la media).
- **Split temporal, nunca aleatorio:** los últimos 90 días son test. Un split aleatorio filtraría el futuro al entrenamiento.
- **Baseline obligatorio:** media por (día de la semana, hora). Un modelo que no la supere no se promueve.
- **Métrica principal: MAE**, y además WAPE para comparar entre locales de distinto tamaño.

## 2026-11-10 · Servir el modelo
- La API carga `models:/aforo@champion`. Se cambia de modelo moviendo el alias en MLflow y llamando a `/recargar`, sin redesplegar.
- Las features se calculan con el mismo módulo (`aforo.features`) al entrenar y al servir.
- Sin features de retardo en el MVP: así la API solo necesita fecha, clima y evento.
