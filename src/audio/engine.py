"""Motor de audio del ecualizador FFT full-duplex.

El callback de ``sounddevice`` es hard-real-time: no usa E/S, no toma locks y
no toca la interfaz. Solo copia el bloque de entrada, aplica el ecualizador y
publica el resultado en una cola que consume el hilo de la GUI.

El intercambio de la respuesta inversa se hace asignando el atributo
``equalizer.h_inverse``. En CPython esa asignacion es atomica (GIL) y el array
nunca se modifica in situ, de modo que el callback siempre ve un filtro
completo, nunca uno a medio construir.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.dsp.inverse_filter import OverlapAddFFTEqualizer, estimate_h_inverse
from src.dsp.limiter import DEFAULT_CEILING, DEFAULT_THRESHOLD, soft_limit

# Techo y umbral del limitador de salida (fraccion de plena escala).
LIMITER_THRESHOLD = DEFAULT_THRESHOLD
OUTPUT_CEILING = DEFAULT_CEILING


class CalibrationDiscardedError(ValueError):
    """El formato cambio y la calibracion cargada dejo de ser valida.

    Se lanza *despues* de aplicar el nuevo formato y reabrir el stream, de modo
    que el motor queda siempre consistente. Quien llama solo debe avisar al
    usuario de que hay que volver a medir.
    """


@dataclass(frozen=True)
class SpectrumFrame:
    """Una captura para la GUI; nunca se modifica desde el callback."""

    input_samples: np.ndarray
    output_samples: np.ndarray


def put_latest(target: "queue.Queue[Any]", item: Any) -> None:
    """Publica sin bloquear; en tiempo real es preferible descartar GUI vieja."""
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        try:
            target.put_nowait(item)
        except queue.Full:
            # Otro consumidor gano la carrera. No se debe bloquear el callback.
            pass


class AudioEngine:
    """Gestiona el stream de audio y el estado del ecualizador.

    La interfaz habla con el motor a traves de metodos simples
    (``load_calibration``, ``recompute_inverse``, ``configure``) y nunca toca
    el callback directamente.
    """

    def __init__(
        self,
        *,
        sample_rate: float = 48_000.0,
        fft_size: int = 2_048,
        channels: int = 1,
        device: str | int | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.channels = int(channels)
        self.device = device

        self.bypass = True
        self.max_gain_db = 12.0
        self.epsilon_base = 0.02
        # Limitador suave antes del DAC; solo se apaga para medir el filtro puro.
        self.limiter_enabled = True

        self.h_response: np.ndarray | None = None
        self.coherence: np.ndarray | None = None
        self.calibration_path: Path | None = None
        # Frecuencia de muestreo con la que se cargo la calibracion: los bins de
        # una medicion solo significan lo mismo a esa frecuencia.
        self._calibration_sample_rate: float | None = None

        self.display_queue: queue.Queue[SpectrumFrame] = queue.Queue(maxsize=2)
        self.status_queue: queue.Queue[str] = queue.Queue(maxsize=8)

        # Se incrementa cada vez que cambia la respuesta inversa, para que la
        # GUI sepa que debe redibujar la curva del filtro.
        self.revision = 0

        self._stream: Any = None
        self.equalizer = self._create_equalizer()

    # ------------------------------------------------------------------ estado
    @property
    def hop_size(self) -> int:
        return self.equalizer.hop_size

    @property
    def bin_count(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def is_calibrated(self) -> bool:
        return self.h_response is not None

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    @property
    def seconds_per_row(self) -> float:
        """Duracion que representa una fila de la cascada."""
        return self.hop_size / self.sample_rate

    def _create_equalizer(self) -> OverlapAddFFTEqualizer:
        return OverlapAddFFTEqualizer(
            h_inverse=np.ones(self.bin_count, dtype=np.complex128),
            fft_size=self.fft_size,
        )

    # ------------------------------------------------------------- calibracion
    def load_calibration(
        self,
        h_path: str | Path,
        coherence_path: str | Path | None = None,
    ) -> None:
        """Carga H(f) y opcionalmente la coherencia; valida las longitudes."""
        h = np.load(h_path)
        if h.ndim != 1 or h.size != self.bin_count:
            raise ValueError(
                f"{h_path} debe contener {self.bin_count} bins para fft_size={self.fft_size}"
            )
        coherence: np.ndarray | None = None
        if coherence_path is not None:
            coherence = np.load(coherence_path)
            if coherence.ndim != 1 or coherence.size != self.bin_count:
                raise ValueError(
                    f"{coherence_path} debe contener {self.bin_count} bins "
                    f"para fft_size={self.fft_size}"
                )
        self.h_response = np.asarray(h, dtype=np.complex128)
        self.coherence = None if coherence is None else np.asarray(coherence, dtype=np.float64)
        self.calibration_path = Path(h_path)
        self._calibration_sample_rate = self.sample_rate
        self.recompute_inverse()

    def clear_calibration(self) -> None:
        """Olvida la medicion y vuelve a un filtro plano."""
        self.h_response = None
        self.coherence = None
        self.calibration_path = None
        self._calibration_sample_rate = None
        self.recompute_inverse()

    def _calibration_is_stale(self) -> bool:
        """True si la medicion cargada no encaja con el formato actual."""
        if self.h_response is None:
            return False
        return (
            self.h_response.size != self.bin_count
            or self._calibration_sample_rate != self.sample_rate
        )

    def recompute_inverse(self) -> None:
        """Recalcula H^-1 y lo publica sin reiniciar el stream de audio."""
        bins = self.bin_count
        if self.bypass or self.h_response is None:
            h_inverse = np.ones(bins, dtype=np.complex128)
        else:
            coherence = (
                np.ones(bins, dtype=np.float64) if self.coherence is None else self.coherence
            )
            h_inverse = estimate_h_inverse(
                self.h_response,
                coherence,
                epsilon_base=self.epsilon_base,
                max_gain_db=self.max_gain_db,
            )
        # Asignacion atomica: el callback nunca ve un array a medio construir.
        self.equalizer.h_inverse = h_inverse
        self.revision += 1

    # ----------------------------------------------------------------- formato
    def configure(
        self,
        *,
        sample_rate: float | None = None,
        fft_size: int | None = None,
        device: Any = ...,
    ) -> None:
        """Cambia el formato del stream y lo reabre.

        ``device`` usa ``...`` como centinela para distinguir "sin cambio" de
        "usar el dispositivo por defecto" (``None``).
        """
        if fft_size is not None and int(fft_size) != self.fft_size:
            new_size = int(fft_size)
            if new_size < 4 or new_size % 2:
                raise ValueError("fft_size debe ser un entero par mayor o igual a 4")
            self.fft_size = new_size
            self.equalizer = self._create_equalizer()

        if sample_rate is not None:
            self.sample_rate = float(sample_rate)
        if device is not ...:
            self.device = device

        # Primero se deja el motor consistente y el stream reabierto; el aviso de
        # calibracion descartada se lanza al final para no dejar el audio mudo.
        discarded = self._calibration_is_stale()
        if discarded:
            self.clear_calibration()
        else:
            self.recompute_inverse()
        self.restart()

        if discarded:
            raise CalibrationDiscardedError(
                "La calibracion se descarto porque no coincide con el nuevo "
                "formato (tamano de FFT o frecuencia de muestreo). "
                "Vuelva a medir el circuito."
            )

    # ------------------------------------------------------------------ stream
    def open(self) -> None:
        """Abre el stream full-duplex de sounddevice."""
        import sounddevice as sd

        self._stream = sd.Stream(
            device=self.device,
            samplerate=self.sample_rate,
            blocksize=self.hop_size,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        """Cierra el stream si esta abierto."""
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def restart(self) -> None:
        """Cierra y vuelve a abrir el stream con el formato vigente."""
        self.close()
        self.open()

    # ---------------------------------------------------------------- callback
    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Callback hard-real-time: no usa Qt, E/S, locks explicitos ni esperas.

        Una excepcion que escapa de aqui hace que PortAudio aborte el stream y
        el audio muere en pleno uso. Por eso cualquier fallo se contiene: la
        salida de ese bloque queda en silencio, se avisa por ``status_queue`` y
        el ecualizador se reinicia para no arrastrar un estado a medias.
        """
        del time_info
        try:
            self._process_callback(indata, outdata, frames, status)
        except Exception as error:  # noqa: BLE001 - el callback nunca debe lanzar
            outdata.fill(0.0)
            put_latest(self.status_queue, f"error en el callback de audio: {error!r}")
            try:
                self.equalizer.reset()
            except Exception:  # noqa: BLE001
                pass

    def _process_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        status: Any,
    ) -> None:
        if status:
            put_latest(self.status_queue, str(status))

        # El flujo usa el primer canal como medicion mono y replica la salida
        # ecualizada en todos los altavoces solicitados.
        input_block = indata[:, 0].copy()
        if not np.all(np.isfinite(input_block)):
            # Un NaN/Inf que entre al ecualizador contaminaria el solapamiento
            # (OLA) y la salida quedaria en NaN para siempre.
            np.nan_to_num(input_block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            put_latest(
                self.status_queue,
                "entrada con valores no finitos: se reemplazaron por cero",
            )
        if frames != self.equalizer.hop_size:
            # No deberia ocurrir con blocksize configurado, pero evita emitir
            # muestras antiguas si el backend incumple el contrato.
            outdata.fill(0.0)
            put_latest(self.status_queue, f"bloque inesperado: {frames} muestras")
            return

        output_block = self.equalizer.process_block(input_block)
        if output_block.size == 0:  # preroll WOLA
            outdata.fill(0.0)
            return

        # Al DAC va la senal limitada; a la GUI va la salida *sin limitar*, para
        # que el medidor CLIP avise cuando el filtro pide mas de la plena escala.
        playback = (
            soft_limit(output_block, LIMITER_THRESHOLD, OUTPUT_CEILING)
            if self.limiter_enabled
            else output_block
        )
        outdata[:] = playback.astype(outdata.dtype, copy=False)[:, np.newaxis]
        put_latest(
            self.display_queue,
            SpectrumFrame(input_samples=input_block, output_samples=output_block.copy()),
        )
