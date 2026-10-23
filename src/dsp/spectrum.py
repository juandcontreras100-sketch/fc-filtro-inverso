"""Utilidades de analisis espectral y de nivel usadas por la interfaz."""

from __future__ import annotations

import numpy as np

DB_FLOOR = -120.0
DB_CEILING = 6.0


def spectrum_db(
    samples: np.ndarray,
    fft_size: int,
    db_floor: float = DB_FLOOR,
) -> np.ndarray:
    """Magnitud RFFT con ventana Hann y suelo en dB para una visualizacion estable."""
    if samples.size == 0:
        return np.full(fft_size // 2 + 1, db_floor, dtype=np.float64)
    window = np.hanning(samples.size)
    coherent_gain = np.sum(window)
    magnitude = np.abs(np.fft.rfft(samples * window, n=fft_size))
    magnitude /= max(coherent_gain, np.finfo(np.float64).eps)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, 1e-12)), db_floor)


def level_db(
    samples: np.ndarray,
    db_floor: float = DB_FLOOR,
) -> tuple[float, float]:
    """Devuelve ``(rms_db, peak_db)`` de un bloque de audio."""
    if samples.size == 0:
        return db_floor, db_floor
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return _to_db(rms, db_floor), _to_db(peak, db_floor)


def _to_db(value: float, db_floor: float) -> float:
    return max(20.0 * np.log10(max(value, 1e-12)), db_floor)


def gain_db(transfer: np.ndarray, db_floor: float = DB_FLOOR) -> np.ndarray:
    """Magnitud de una respuesta compleja expresada en dB."""
    magnitude = np.abs(np.asarray(transfer))
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, 1e-12)), db_floor)
