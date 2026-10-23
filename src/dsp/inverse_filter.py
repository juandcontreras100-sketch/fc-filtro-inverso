"""Inversión regularizada y ecualización FFT por overlap-add.

La coherencia que recibe este módulo es ``gamma_squared`` (no ``gamma``):
sus valores físicos están entre cero y uno.  Las mediciones ligeramente
fuera de ese intervalo se recortan para no volver agresivo al inversor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_finite_1d(name: str, values: np.ndarray, *, complex_ok: bool) -> np.ndarray:
    """Convierte una entrada de espectro y rechaza datos no utilizables."""
    dtype = np.complex128 if complex_ok else np.float64
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} debe ser un vector 1-D no vacío")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} debe contener solo valores finitos")
    return array


def _validate_non_negative_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} debe ser un escalar finito y no negativo")
    return value


def epsilon_profile(
    h_freq: np.ndarray,
    coherence: np.ndarray,
    epsilon_base: float,
    k: float = 5.0,
) -> np.ndarray:
    """Calcula un perfil Tikhonov positivo y dependiente de frecuencia.

    ``epsilon_base`` se expresa respecto al pico de ``|H|^2``. Un suelo de
    precisión de máquina se aplica incluso cuando la medición es nula; así
    ``conj(H) / (|H|^2 + epsilon)`` nunca evalúa una división por cero.

    Args:
        h_freq: Respuesta compleja medida en los bins FFT/RFFT.
        coherence: ``gamma_squared`` para cada bin. Se recorta a ``[0, 1]``.
        epsilon_base: Intensidad base de regularización relativa al pico.
        k: Penalización adicional para bandas de baja coherencia.
    """
    h = _as_finite_1d("h_freq", h_freq, complex_ok=True)
    gamma_squared = _as_finite_1d("coherence", coherence, complex_ok=False)
    if h.shape != gamma_squared.shape:
        raise ValueError("h_freq y coherence deben tener la misma longitud")

    epsilon_base = _validate_non_negative_scalar("epsilon_base", epsilon_base)
    k = _validate_non_negative_scalar("k", k)
    gamma_squared = np.clip(gamma_squared, 0.0, 1.0)

    peak_power = float(np.max(np.abs(h) ** 2))
    requested = epsilon_base * peak_power * (1.0 + k * (1.0 - gamma_squared))

    # La escala 1.0 mantiene un denominador positivo para H == 0, mientras
    # que para respuestas normales el suelo es despreciable frente a epsilon.
    epsilon_floor = np.finfo(np.float64).eps * max(peak_power, 1.0)
    return np.maximum(requested, epsilon_floor)


def estimate_h_inverse(
    h_freq: np.ndarray,
    coherence: np.ndarray,
    epsilon_base: float,
    max_gain_db: float,
) -> np.ndarray:
    """Estima un inverso Wiener/Tikhonov con límite estricto de ganancia.

    La fase de la inversión se conserva. Las muescas profundas se atenúan
    mediante ``epsilon`` y, como segunda barrera de seguridad, su módulo se
    limita a ``max_gain_db``.
    """
    h = _as_finite_1d("h_freq", h_freq, complex_ok=True)
    max_gain_db = _validate_non_negative_scalar("max_gain_db", max_gain_db)
    eps = epsilon_profile(h, coherence, epsilon_base)

    denominator = np.abs(h) ** 2 + eps
    h_inverse = np.conj(h) / denominator

    max_allowed = 10.0 ** (max_gain_db / 20.0)
    magnitude = np.abs(h_inverse)
    gain_scale = np.minimum(1.0, max_allowed / np.maximum(magnitude, 1e-300))
    return h_inverse * gain_scale


def _sqrt_hann(fft_size: int) -> np.ndarray:
    """Ventana sqrt-Hann periódica, COLA para salto N/2."""
    return np.sqrt(np.hanning(fft_size + 1)[:-1])


@dataclass
class OverlapAddFFTEqualizer:
    """Ecualizador WOLA para bloques de audio de tamaño fijo.

    El objeto recibe exactamente ``hop_size`` muestras por llamada a
    :meth:`process_block`. Usa ventanas sqrt-Hann de análisis y síntesis,
    suma las regiones solapadas y las normaliza por la suma de ventanas. La
    primera salida contiene solo el preroll y se descarta internamente; el
    retardo algorítmico es un salto. :meth:`flush` entrega la cola restante.

    ``h_inverse`` debe contener ``fft_size // 2 + 1`` bins, es decir, el
    formato de :func:`numpy.fft.rfft` para una señal real.
    """

    h_inverse: np.ndarray
    fft_size: int
    hop_size: int | None = None
    _window: np.ndarray = field(init=False, repr=False)
    _input: np.ndarray = field(init=False, repr=False)
    _ola: np.ndarray = field(init=False, repr=False)
    _normalizer: np.ndarray = field(init=False, repr=False)
    _discard_preroll: bool = field(init=False, default=True, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fft_size, (int, np.integer)) or self.fft_size < 4:
            raise ValueError("fft_size debe ser un entero de al menos 4")
        if self.fft_size % 2:
            raise ValueError("fft_size debe ser par para usar la ventana COLA N/2")

        self.hop_size = self.fft_size // 2 if self.hop_size is None else self.hop_size
        if self.hop_size != self.fft_size // 2:
            raise ValueError("hop_size debe ser fft_size // 2 para esta ventana COLA")

        self.h_inverse = _as_finite_1d("h_inverse", self.h_inverse, complex_ok=True)
        expected_bins = self.fft_size // 2 + 1
        if self.h_inverse.size != expected_bins:
            raise ValueError(f"h_inverse debe tener {expected_bins} bins para fft_size={self.fft_size}")

        self._window = _sqrt_hann(self.fft_size)
        self._input = np.zeros(self.fft_size - self.hop_size, dtype=np.float64)
        self._ola = np.zeros(self.fft_size, dtype=np.float64)
        self._normalizer = np.zeros(self.fft_size, dtype=np.float64)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(frame * self._window)
        filtered = np.fft.irfft(spectrum * self.h_inverse, n=self.fft_size)
        weighted = filtered * self._window
        window_power = self._window ** 2
        self._ola += weighted
        self._normalizer += window_power

        output = np.divide(
            self._ola[: self.hop_size],
            self._normalizer[: self.hop_size],
            out=np.zeros(self.hop_size, dtype=np.float64),
            where=self._normalizer[: self.hop_size] > np.finfo(np.float64).eps,
        )
        self._ola[:-self.hop_size] = self._ola[self.hop_size :]
        self._ola[-self.hop_size :] = 0.0
        self._normalizer[:-self.hop_size] = self._normalizer[self.hop_size :]
        self._normalizer[-self.hop_size :] = 0.0
        return output

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Procesa un bloque y devuelve audio estable, o vacío en el preroll."""
        samples = _as_finite_1d("block", block, complex_ok=False)
        if samples.size != self.hop_size:
            raise ValueError(f"block debe tener exactamente {self.hop_size} muestras")
        self._input = np.concatenate((self._input, samples))
        output = self._process_frame(self._input)
        self._input = self._input[self.hop_size :]
        if self._discard_preroll:
            self._discard_preroll = False
            return np.empty(0, dtype=np.float64)
        return output

    def flush(self) -> np.ndarray:
        """Procesa un bloque cero y devuelve la cola WOLA; reinicia el estado."""
        tail = self.process_block(np.zeros(self.hop_size, dtype=np.float64))
        self.reset()
        return tail

    def reset(self) -> None:
        """Borra historial y solapamiento antes de una nueva secuencia."""
        self._input.fill(0.0)
        self._ola.fill(0.0)
        self._normalizer.fill(0.0)
        self._discard_preroll = True
