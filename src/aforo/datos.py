"""Generador de datos SINTÉTICOS de afluencia de un bar-cafetería en Granada.

Los datos son inventados pero con patrones realistas que el modelo debe descubrir:
- Perfil por hora: desayunos (9-10h), comidas (14-15h) y noche (20-22h).
- Viernes y sábado más fuertes; domingo fuerte a mediodía.
- Festivos se comportan como sábado; vísperas, noche más fuerte.
- Agosto baja (Granada se vacía de estudiantes).
- Clima: terraza buena entre 18 y 28 ºC, el calor extremo hunde las tardes, la lluvia resta.
- Eventos (partidos, conciertos): suben la noche.
- Tendencia: el local crece ~10 % al año.
- Ruido Poisson + problemas de calidad (huecos de sensor, picos erróneos) para practicar validación.

Uso:
    python -m aforo.datos --desde 2024-01-01 --hasta 2025-12-31 --salida datos/afluencia.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aforo.features import construir_features, en_horario
from aforo.festivos import esta_cerrado

# Clientes medios por hora en un martes normal de primavera.
PERFIL_HORA = {
    8: 14, 9: 30, 10: 26, 11: 16, 12: 14, 13: 24, 14: 40, 15: 34, 16: 15,
    17: 12, 18: 16, 19: 22, 20: 32, 21: 38, 22: 30, 23: 16,
}
MULT_DIA = {0: 0.85, 1: 0.90, 2: 0.95, 3: 1.05, 4: 1.30, 5: 1.45, 6: 1.10}


def _clima(horas: pd.DatetimeIndex, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Temperatura (ºC) y lluvia (mm/h) plausibles para Granada."""
    dia = horas.dayofyear.to_numpy()
    hora = horas.hour.to_numpy()
    # Media anual ~16 ºC, mínimo en enero, máximo a finales de julio.
    estacional = 16 - 10 * np.cos(2 * np.pi * (dia - 20) / 365.25)
    diario = 6 * np.sin(2 * np.pi * (hora - 9) / 24)  # máximo hacia las 15-16h
    # Anomalía diaria persistente (rachas de días frescos o calurosos).
    n_dias = (horas.normalize().unique()).size
    anom = np.cumsum(rng.normal(0, 0.8, n_dias))
    anom = anom - pd.Series(anom).rolling(10, min_periods=1).mean().to_numpy()
    idx_dia = (horas.normalize() - horas.normalize()[0]).days
    temp = estacional + diario + anom[idx_dia] * 1.5 + rng.normal(0, 0.7, len(horas))

    # Lluvia: días lluviosos más probables en invierno, casi nunca en verano.
    mes_dia = pd.DatetimeIndex(horas.normalize().unique()).month
    p_lluvia = {1: .22, 2: .2, 3: .2, 4: .2, 5: .14, 6: .06, 7: .02, 8: .03,
                9: .1, 10: .18, 11: .22, 12: .22}
    dia_lluvioso = rng.random(n_dias) < np.array([p_lluvia[m] for m in mes_dia])
    lluvia = np.where(
        dia_lluvioso[idx_dia] & (rng.random(len(horas)) < 0.45),
        rng.gamma(1.2, 1.5, len(horas)),
        0.0,
    )
    return temp.round(1), lluvia.round(1)


def _eventos(horas: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """~3 eventos al mes, de 19 a 23h."""
    dias = horas.normalize().unique()
    dias_evento = set(dias[rng.random(len(dias)) < 0.1])
    return np.array(
        [int(h.normalize() in dias_evento and 19 <= h.hour <= 23) for h in horas]
    )


def _clientes_esperados(entrada: pd.DataFrame) -> np.ndarray:
    f = construir_features(entrada)
    base = f["hora"].map(PERFIL_HORA).to_numpy(dtype=float)

    dia = f["dia_semana"].to_numpy()
    mult = np.array([MULT_DIA[d] for d in dia])
    mult = np.where(f["festivo"] == 1, np.maximum(mult, MULT_DIA[5]), mult)
    noche = f["hora"].between(20, 23).to_numpy()
    mult = np.where((f["vispera_festivo"] == 1) & noche, mult * 1.25, mult)
    mult = np.where(f["agosto"] == 1, mult * 0.72, mult)

    t = f["temperatura"].to_numpy()
    terraza = np.where((t >= 18) & (t <= 28), 1.15, 1.0)
    calor = np.where((t > 33) & f["hora"].between(14, 19).to_numpy(), 0.6, 1.0)
    frio = np.where(t < 6, 0.85, 1.0)
    lluvia = np.where(f["lluvia_mm"] > 0.5, 0.7, 1.0)
    evento = np.where(f["evento"] == 1, 1.6, 1.0)
    tendencia = 1 + 0.10 * f["dias_desde_inicio"].to_numpy() / 365.25

    return base * mult * terraza * calor * frio * lluvia * evento * tendencia


def generar(desde: str, hasta: str, semilla: int = 42, con_ruido_calidad: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    horas = pd.date_range(desde, f"{hasta} 23:00", freq="h")
    temp, lluvia = _clima(horas, rng)
    df = pd.DataFrame({
        "fecha_hora": horas,
        "temperatura": temp,
        "lluvia_mm": lluvia,
        "evento": _eventos(horas, rng),
    })
    df = df[en_horario(df["fecha_hora"])]
    df = df[~df["fecha_hora"].dt.date.map(esta_cerrado)].reset_index(drop=True)

    df["clientes"] = rng.poisson(_clientes_esperados(df))

    if con_ruido_calidad:
        # 0,5 % de horas sin dato (sensor caído) y 0,1 % de lecturas absurdas.
        n = len(df)
        huecos = rng.choice(n, int(n * 0.005), replace=False)
        df.loc[huecos, "clientes"] = np.nan
        picos = rng.choice(n, int(n * 0.001), replace=False)
        df.loc[picos, "clientes"] = df.loc[picos, "clientes"] * 10
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita huecos y lecturas imposibles (más de 3x el p99,9 o negativas)."""
    df = df.dropna(subset=["clientes"])
    limite = df["clientes"].quantile(0.999) * 3
    return df[(df["clientes"] >= 0) & (df["clientes"] <= limite)].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Genera datos sintéticos de afluencia")
    p.add_argument("--desde", default="2024-01-01")
    p.add_argument("--hasta", default="2025-12-31")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="datos/afluencia.csv")
    a = p.parse_args()

    df = generar(a.desde, a.hasta, a.semilla)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.salida, index=False)
    print(f"{len(df)} filas -> {a.salida} ({df['clientes'].isna().sum()} huecos)")


if __name__ == "__main__":
    main()
