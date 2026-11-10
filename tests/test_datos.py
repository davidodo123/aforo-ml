import pandas as pd

from aforo.datos import generar, limpiar
from aforo.features import FEATURES, construir_features


def test_generador_es_reproducible():
    a = generar("2024-03-01", "2024-03-31", semilla=1)
    b = generar("2024-03-01", "2024-03-31", semilla=1)
    pd.testing.assert_frame_equal(a, b)


def test_solo_horario_de_apertura_y_dias_abiertos():
    df = generar("2024-12-20", "2025-01-05", con_ruido_calidad=False)
    horas = df["fecha_hora"].dt.hour
    assert horas.min() == 8 and horas.max() == 23
    fechas = set(df["fecha_hora"].dt.strftime("%m-%d"))
    assert "12-25" not in fechas and "01-01" not in fechas


def test_patrones_que_debe_aprender_el_modelo():
    df = generar("2024-01-01", "2024-12-31", con_ruido_calidad=False)
    por_dia = df.groupby(df["fecha_hora"].dt.dayofweek)["clientes"].mean()
    assert por_dia[5] > por_dia[0]  # sábado > lunes
    por_mes = df.groupby(df["fecha_hora"].dt.month)["clientes"].mean()
    assert por_mes[8] < por_mes[9]  # agosto flojo


def test_limpiar_quita_huecos_y_picos():
    df = generar("2024-01-01", "2024-06-30", semilla=3)
    limpio = limpiar(df)
    assert limpio["clientes"].isna().sum() == 0
    assert limpio["clientes"].max() < df["clientes"].max()


def test_features_columnas_y_sin_nulos():
    df = generar("2024-05-01", "2024-05-07", con_ruido_calidad=False)
    X = construir_features(df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()


def test_features_festivo_y_vispera():
    df = pd.DataFrame({
        "fecha_hora": ["2025-12-24 21:00", "2025-12-25 13:00"],
        "temperatura": [10.0, 12.0],
        "lluvia_mm": [0.0, 0.0],
        "evento": [0, 0],
    })
    X = construir_features(df)
    assert X["vispera_festivo"].tolist() == [1, 0]
    assert X["festivo"].tolist() == [0, 1]
