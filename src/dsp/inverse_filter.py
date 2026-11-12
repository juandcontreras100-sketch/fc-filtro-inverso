"""Inversión regularizada y ecualización FFT por overlap-add.

La coherencia que recibe este módulo es ``gamma_squared`` (no ``gamma``):
sus valores físicos están entre cero y uno.  Las mediciones ligeramente
fuera de ese intervalo se recortan para no volver agresivo al inversor.
"""

from __future__ import annotations

import numpy as np

from src.config import MIN_COHERENCE_SQUARED


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
    *,
    peak_percentile: float = 100.0,
    reference_mask: np.ndarray | None = None,
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
        peak_percentile: Percentil de ``|H|^2`` usado como referencia. Con
            ``100`` se usa el máximo (comportamiento histórico). Un valor
            menor, por ejemplo ``99``, evita que un único bin espurio
            (ultrasonido, artefacto de barrido) regule toda la banda.
        reference_mask: Máscara booleana de los bins que pueden fijar el
            pico de referencia. Sirve para excluir artefactos fuera de la
            banda útil. Si no contiene ningún ``True``, se usa toda la banda.
    """
    h = _as_finite_1d("h_freq", h_freq, complex_ok=True)
    gamma_squared = _as_finite_1d("coherence", coherence, complex_ok=False)
    if h.shape != gamma_squared.shape:
        raise ValueError("h_freq y coherence deben tener la misma longitud")

    epsilon_base = _validate_non_negative_scalar("epsilon_base", epsilon_base)
    k = _validate_non_negative_scalar("k", k)
    peak_percentile = float(peak_percentile)
    if not np.isfinite(peak_percentile) or not 0.0 < peak_percentile <= 100.0:
        raise ValueError("peak_percentile debe estar entre 0 y 100")
    gamma_squared = np.clip(gamma_squared, 0.0, 1.0)

    power = np.abs(h) ** 2
    reference_power = power
    if reference_mask is not None:
        mask = np.asarray(reference_mask, dtype=bool)
        if mask.shape != power.shape:
            raise ValueError("reference_mask debe tener la misma longitud que h_freq")
        if np.any(mask):
            reference_power = power[mask]

    peak_power = (
        float(np.max(reference_power))
        if peak_percentile >= 100.0
        else float(np.percentile(reference_power, peak_percentile))
    )
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
    *,
    peak_percentile: float = 100.0,
    reference_mask: np.ndarray | None = None,
    coherence_threshold: float = MIN_COHERENCE_SQUARED,
) -> np.ndarray:
    """Estima un inverso Wiener/Tikhonov con límite estricto de ganancia.

    La fase de la inversión se conserva. Las muescas profundas se atenúan
    mediante ``epsilon`` y, como segunda barrera de seguridad, su módulo se
    limita a ``max_gain_db``. ``peak_percentile`` y ``reference_mask`` se
    pasan a :func:`epsilon_profile`; juntos evitan que un artefacto aislado
    fuera de la banda útil aplaste toda la corrección.
    """
    h = _as_finite_1d("h_freq", h_freq, complex_ok=True)
    max_gain_db = _validate_non_negative_scalar("max_gain_db", max_gain_db)
    coherence_threshold = float(coherence_threshold)
    if (
        not np.isfinite(coherence_threshold)
        or not 0.0 <= coherence_threshold <= 1.0
    ):
        raise ValueError("coherence_threshold debe estar entre 0 y 1")
    gamma_squared = _as_finite_1d("coherence", coherence, complex_ok=False)
    if gamma_squared.shape != h.shape:
        raise ValueError("h_freq y coherence deben tener la misma longitud")
    valid_coherence = np.clip(gamma_squared, 0.0, 1.0) >= coherence_threshold
    eps = epsilon_profile(
        h,
        gamma_squared,
        epsilon_base,
        peak_percentile=peak_percentile,
        reference_mask=reference_mask,
    )

    denominator = np.abs(h) ** 2 + eps
    h_inverse = np.conj(h) / denominator

    max_allowed = 10.0 ** (max_gain_db / 20.0)
    magnitude = np.abs(h_inverse)
    gain_scale = np.minimum(1.0, max_allowed / np.maximum(magnitude, 1e-300))
    h_inverse *= gain_scale
    # Unity means "do not correct": muting a low-coherence band would be a
    # larger and equally unsupported change than retaining the original path.
    h_inverse[~valid_coherence] = 1.0
    return h_inverse


class OverlapAddFFTEqualizer:
    """Ecualizador WOLA causal para bloques de audio de tamaño fijo.

    Cada llamada recibe ``hop_size`` muestras. Una ventana Hann periódica
    cumple COLA al 50 % de solapamiento (la suma de las ventanas desplazadas
    es uno); cada trama se convoluciona linealmente con los ``fft_size``
    coeficientes causales de la inversa y sus colas se suman. El relleno FFT
    evita el aliasado circular que aparece al filtrar tramas sin extensión.

    El primer bloque de salida es el preroll de una trama y se descarta; la
    latencia algorítmica es exactamente un hop. ``h_inverse`` usa el formato
    RFFT de ``fft_size // 2 + 1`` bins.
    """

    def __init__(
        self,
        h_inverse: np.ndarray,
        fft_size: int,
        hop_size: int | None = None,
    ) -> None:
        if not isinstance(fft_size, (int, np.integer)) or fft_size < 4:
            raise ValueError("fft_size debe ser un entero de al menos 4")
        if fft_size % 2:
            raise ValueError("fft_size debe ser par para usar la ventana COLA N/2")

        self.fft_size = int(fft_size)
        self.hop_size = self.fft_size // 2 if hop_size is None else hop_size
        if self.hop_size != self.fft_size // 2:
            raise ValueError("hop_size debe ser fft_size // 2 para esta ventana COLA")

        self._h_inverse = np.empty(0, dtype=np.complex128)
        self._filter_spectrum = np.empty(0, dtype=np.complex128)
        self.h_inverse = h_inverse

        self._window = np.hanning(self.fft_size + 1)[:-1]
        self._input_ring = np.zeros(self.fft_size, dtype=np.float64)
        self._frame = np.zeros(self.fft_size, dtype=np.float64)
        self._ola = np.zeros(2 * self.fft_size, dtype=np.float64)
        self._ring_write = 0
        self._discard_preroll = True

    @property
    def h_inverse(self) -> np.ndarray:
        return self._h_inverse

    @h_inverse.setter
    def h_inverse(self, values: np.ndarray) -> None:
        inverse = _as_finite_1d("h_inverse", values, complex_ok=True)
        expected_bins = self.fft_size // 2 + 1
        if inverse.size != expected_bins:
            raise ValueError(
                f"h_inverse debe tener {expected_bins} bins para fft_size={self.fft_size}"
            )
        impulse = np.fft.irfft(inverse, n=self.fft_size)
        self._h_inverse = inverse
        self._filter_spectrum = np.fft.rfft(impulse, n=2 * self.fft_size)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(
            frame * self._window,
            n=2 * self.fft_size,
        )
        filtered = np.fft.irfft(
            spectrum * self._filter_spectrum,
            n=2 * self.fft_size,
        )
        self._ola[:-1] += filtered[:-1]
        output = self._ola[: self.hop_size].copy()
        self._ola[:-self.hop_size] = self._ola[self.hop_size :]
        self._ola[-self.hop_size :] = 0.0
        return output

    def process_block(self, block: np.ndarray) -> np.ndarray:
        """Procesa un bloque y devuelve audio estable, o vacío en el preroll."""
        samples = _as_finite_1d("block", block, complex_ok=False)
        if samples.size != self.hop_size:
            raise ValueError(f"block debe tener exactamente {self.hop_size} muestras")
        write_end = self._ring_write + self.hop_size
        if write_end <= self.fft_size:
            self._input_ring[self._ring_write : write_end] = samples
        else:
            first_count = self.fft_size - self._ring_write
            self._input_ring[self._ring_write :] = samples[:first_count]
            self._input_ring[: write_end - self.fft_size] = samples[first_count:]
        self._ring_write = write_end % self.fft_size

        first_count = self.fft_size - self._ring_write
        self._frame[:first_count] = self._input_ring[self._ring_write :]
        self._frame[first_count:] = self._input_ring[: self._ring_write]
        output = self._process_frame(self._frame)
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
        self._input_ring.fill(0.0)
        self._frame.fill(0.0)
        self._ring_write = 0
        self._ola.fill(0.0)
        self._discard_preroll = True
