"""Utilidades de analisis espectral y de nivel usadas por la interfaz."""

from __future__ import annotations

import numpy as np

from src.config import (
    CLIP_THRESHOLD_LINEAR,
    DBFS_CEILING_DB,
    DBFS_FLOOR_DB,
    DEFAULT_SAMPLE_RATE_HZ,
)

DB_FLOOR = DBFS_FLOOR_DB
DB_CEILING = DBFS_CEILING_DB
CLIP_THRESHOLD = CLIP_THRESHOLD_LINEAR
CLIP_DB = 20.0 * np.log10(CLIP_THRESHOLD)

# Centros nominales ISO 266 de 25 Hz a 20 kHz.
_THIRD_OCTAVE_CENTERS_HZ = np.array(
    [
        25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,
        250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
        2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
        12500.0, 16000.0, 20000.0,
    ]
)


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
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequencies_hz debe ser finita y no negativa")
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
    values_db = np.maximum(values_db, DB_FLOOR)
    return float(values_db) if values_db.ndim == 0 else values_db


def c_weighting_db(frequencies_hz: float | np.ndarray) -> float | np.ndarray:
    """C ponderacion IEC 61672 en dB, normalizada a 0 dB en 1 kHz."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequencies_hz debe ser finita y no negativa")

    frequency_2 = np.square(frequencies)
    reference_2 = 1_000.0**2

    def response(frequency_squared: np.ndarray) -> np.ndarray:
        return (
            12_194.0**2 * frequency_squared
            / ((frequency_squared + 20.6**2) * (frequency_squared + 12_194.0**2))
        )

    reference_response = response(np.asarray(reference_2))
    values_db = 20.0 * np.log10(
        np.maximum(response(frequency_2), np.finfo(np.float64).tiny)
        / reference_response
    )
    values_db = np.maximum(values_db, DB_FLOOR)
    return float(values_db) if values_db.ndim == 0 else values_db


def equivalent_noise_bandwidth_hz(sample_rate_hz: float, fft_size: int) -> float:
    """ENBW de ventana Hann periódica en Hz: fs * sum(w²) / sum(w)²."""
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz debe ser finita y positiva")
    if not isinstance(fft_size, (int, np.integer)) or fft_size < 2:
        raise ValueError("fft_size debe ser un entero de al menos 2")
    window = np.hanning(fft_size + 1)[:-1]
    coherent_gain = float(np.sum(window))
    if coherent_gain <= 0.0:
        raise ValueError("la ventana no tiene ganancia coherente positiva")
    return float(sample_rate_hz * np.sum(window**2) / coherent_gain**2)


def total_level_db(
    samples: np.ndarray,
    weighting: str = "Z",
    sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
) -> float:
    """Nivel RMS total en dBFS por Parseval, con ponderacion Z, A o C.

    ``sample_rate`` solo interviene al calcular la ponderacion A/C; se usa
    48 kHz por defecto y la interfaz pasa la frecuencia real del stream.
    """
    if weighting not in ("Z", "A", "C"):
        raise ValueError("weighting debe ser 'Z', 'A' o 'C'")
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate debe ser finita y positiva")

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples debe ser un vector 1-D")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples debe contener solo valores finitos")
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
    elif weighting == "C":
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / float(sample_rate))
        weighting_db = c_weighting_db(frequencies)
        power *= np.power(10.0, weighting_db / 10.0)

    mean_square = float(np.sum(power * bin_weights) / samples.size**2)
    return _to_db(np.sqrt(mean_square), DB_FLOOR)


def spectrum_db(
    samples: np.ndarray,
    fft_size: int,
    db_floor: float = DB_FLOOR,
) -> np.ndarray:
    """Magnitud de pico dBFS en RFFT, corregida por ganancia coherente Hann.

    0 dBFS corresponde a amplitud de pico 1.0; un seno de amplitud pico 1.0
    tiene RMS de tiempo -3.0103 dBFS. El ENBW para ruido esta disponible en
    :func:`equivalent_noise_bandwidth_hz`.

    ``rfft`` devuelve solo la mitad no negativa del espectro: cada bin interior
    representa un par de frecuencias conjugadas, por eso su magnitud se
    duplica. DC y Nyquist (si existe) son bins unicos y no se duplican.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples debe ser un vector 1-D")
    if not isinstance(fft_size, (int, np.integer)) or fft_size < 1:
        raise ValueError("fft_size debe ser un entero positivo")
    if not np.isfinite(db_floor):
        raise ValueError("db_floor debe ser finito")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples debe contener solo valores finitos")
    if samples.size == 0:
        return np.full(fft_size // 2 + 1, db_floor, dtype=np.float64)
    window = np.hanning(samples.size + 1)[:-1]
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
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("samples debe ser un vector 1-D")
    if not np.isfinite(db_floor):
        raise ValueError("db_floor debe ser finito")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples debe contener solo valores finitos")
    if samples.size == 0:
        return db_floor, db_floor
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return _to_db(rms, db_floor), _to_db(peak, db_floor)


def _to_db(value: float, db_floor: float) -> float:
    linear_floor = 10.0 ** (db_floor / 20.0)
    return max(20.0 * np.log10(max(value, linear_floor)), db_floor)


def gain_db(transfer: np.ndarray, db_floor: float = DB_FLOOR) -> np.ndarray:
    """Magnitud de una respuesta compleja expresada en dB."""
    magnitude = np.abs(np.asarray(transfer))
    linear_floor = 10.0 ** (db_floor / 20.0)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude, linear_floor)), db_floor)


def third_octave_bands(
    levels_db: np.ndarray,
    frequencies_hz: np.ndarray,
    db_floor: float = DB_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa un espectro en dB por bandas de 1/3 de octava.

    Usa los centros nominales ISO de 25 Hz a 20 kHz que quedan por debajo de
    Nyquist (la frecuencia del ultimo bin). Cada banda cubre
    ``centro * 2**(+-1/6)`` y su nivel es la media de potencia de los bins que
    caen dentro, con suelo en ``db_floor``. Las bandas sin ningun bin se
    omiten, algo frecuente en graves cuando la FFT es pequena; el bin de DC
    (0 Hz) no cae en ninguna banda.

    Devuelve ``(centros_hz, niveles_db)``.
    """
    levels = np.asarray(levels_db, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, np.zeros(0, dtype=np.float64)

    nyquist = float(frequencies[-1])
    centers = _THIRD_OCTAVE_CENTERS_HZ[_THIRD_OCTAVE_CENTERS_HZ < nyquist]
    lower_edges = centers * 2.0 ** (-1.0 / 6.0)
    upper_edges = centers * 2.0 ** (1.0 / 6.0)
    kept_centers: list[float] = []
    kept_levels: list[float] = []
    for center, lower, upper in zip(centers, lower_edges, upper_edges):
        mask = (frequencies >= lower) & (frequencies < upper)
        if not np.any(mask):
            continue
        power = float(np.mean(np.power(10.0, levels[mask] / 10.0)))
        kept_centers.append(float(center))
        power_floor = 10.0 ** (db_floor / 10.0)
        kept_levels.append(max(10.0 * np.log10(max(power, power_floor)), db_floor))
    return (
        np.asarray(kept_centers, dtype=np.float64),
        np.asarray(kept_levels, dtype=np.float64),
    )
