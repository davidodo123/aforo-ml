"""Festivos de Granada (nacionales, Andalucía y locales) usados por el generador y las features.

Fechas fijas más las móviles de Semana Santa y Corpus de 2024-2026.
Si amplías el rango de datos, añade aquí los festivos del año nuevo.
"""

from datetime import date

FESTIVOS: set[date] = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 6), date(2024, 2, 28),
    date(2024, 3, 28), date(2024, 3, 29), date(2024, 5, 1), date(2024, 5, 30),
    date(2024, 8, 15), date(2024, 10, 12), date(2024, 11, 1), date(2024, 12, 6),
    date(2024, 12, 9), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 6), date(2025, 2, 28),
    date(2025, 4, 17), date(2025, 4, 18), date(2025, 5, 1), date(2025, 6, 19),
    date(2025, 8, 15), date(2025, 10, 13), date(2025, 11, 1), date(2025, 12, 6),
    date(2025, 12, 8), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 6), date(2026, 2, 28),
    date(2026, 4, 2), date(2026, 4, 3), date(2026, 5, 1), date(2026, 6, 4),
    date(2026, 8, 15), date(2026, 10, 12), date(2026, 11, 2), date(2026, 12, 7),
    date(2026, 12, 8), date(2026, 12, 25),
}

# Días en que el local cierra.
CERRADO: set[tuple[int, int]] = {(12, 25), (1, 1)}


def es_festivo(d: date) -> bool:
    return d in FESTIVOS


def esta_cerrado(d: date) -> bool:
    return (d.month, d.day) in CERRADO
