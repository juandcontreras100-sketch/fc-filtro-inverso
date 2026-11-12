"""Utilidades de analisis espectral y de nivel usadas por la interfaz."""

from __future__ import annotations

import numpy as np

DB_FLOOR = -120.0
DB_CEILING = 6.0
CLIP_DB = -0.1
CLIP_THRESHOLD = 10.0 ** (CLIP_DB / 20.0)


def dbfs_to_spl(
    values_db: float | np.ndarray,
    offset_db: float,
) -> float | np.ndarray:
    """Convierte niveles dBFS a dB SPL con el offset de calibracion."""
    values_spl = np.asarray(values_db, dtype=np.float64) + float(offset_db)
    return float(values_spl) if values_spl.ndim == 0 else values_spl


def a_weighting_db(frequencies_hz: float | np.ndarray) -> float | np.ndarray:
    """Ponderacion A de IEC 61672, normalizada a 0 dB en 1 kHz."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    frequency_2 = np.square(frequencies)
    frequency_4 = np.square(frequency_2)
    reference_2 = 1_000.0**2

    denominator = (
        (frequency_2 + 20.6**2)
        * np.sqrt((frequency_2 + 107.7**2) * (frequency_2 + 737.9**2))
        * (frequency_2 + 12_194.0**2)
    )
    response = (12_194.0**2 * frequency_4) / denominator
    reference_response = (12_194.0**2 * reference_2**2) / (
        (reference_2 + 20.6**2)
        * np.sqrt((reference_2 + 107.7**2) * (reference_2 + 737.9**2))
        * (reference_2 + 12_194.0**2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values_db = 20.0 * np.log10(response) - 20.0 * np.log10(reference_response)
    return float(values_db) if values_db.ndim == 0 else values_db


def total_level_db(
    samples: np.ndarray,
    weighting: str = "Z",
    sample_rate: float = 48_000.0,
) -> float:
    """Nivel RMS total en dBFS por Parseval, con ponderacion Z o A.

    ``sample_rate`` solo interviene al calcular la ponderacion A; se usa 48 kHz
    por defecto y la interfaz pasa siempre la frecuencia real del stream.
    """
    if weighting not in ("Z", "A"):
        raise ValueError("weighting debe ser 'Z' o 'A'")

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return DB_FLOOR

    spectrum = np.fft.rfft(samples)
    bin_weights = np.full(spectrum.size, 2.0, dtype=np.float64)
    bin_weights[0] = 1.0
    if samples.size % 2 == 0:
        bin_weights[-1] = 1.0

    power = np.square(np.abs(spectrum))
    if weighting == "A":
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / float(sample_rate))
        weighting_db = a_weighting_db(frequencies)
        power *= np.power(10.0, weighting_db / 10.0)

    mean_square = float(np.sum(power * bin_weights) / samples.size**2)
    return _to_db(np.sqrt(mean_square), DB_FLOOR)


def spectrum_db(
    samples: np.ndarray,
    fft_size: int,
    db_floor: float = DB_FLOOR,
) -> np.ndarray:
    """Magnitud RFFT con ventana Hann y suelo en dB para una visualizacion estable.

    ``rfft`` devuelve solo la mitad no negativa del espectro: cada bin interior
    representa un par de frecuencias conjugadas, por eso su magnitud se
    duplica. DC y Nyquist (si existe) son bins unicos y no se duplican.
    """
    if samples.size == 0:
        return np.full(fft_size // 2 + 1, db_floor, dtype=np.float64)
    window = np.hanning(samples.size)
    coherent_gain = np.sum(window)
    magnitude = np.abs(np.fft.rfft(samples * window, n=fft_size))
    magnitude /= max(coherent_gain, np.finfo(np.float64).eps)
    # El ultimo bin es Nyquist solo cuando fft_size es par.
    interior_end = -1 if fft_size % 2 == 0 else magnitude.size
    magnitude[1:interior_end] *= 2.0
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
