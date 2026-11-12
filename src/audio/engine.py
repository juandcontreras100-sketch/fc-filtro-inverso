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

import os
import queue
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_EPSILON_RATIO,
    DEFAULT_MAX_GAIN_DB,
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_FFT_SIZE_SAMPLES,
)
from src.dsp.inverse_filter import OverlapAddFFTEqualizer, estimate_h_inverse
from src.dsp.limiter import DEFAULT_CEILING, DEFAULT_THRESHOLD, soft_limit
from src.dsp.spectrum import CLIP_THRESHOLD

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
    """Una captura para la GUI; nunca se modifica desde el callback.

    ``input_samples`` y ``output_samples`` son el hop alineado que se usa para
    medir niveles. ``input_window`` y ``output_window`` contienen las ultimas
    ``fft_size`` muestras contiguas (sin huecos) para calcular el espectro.
    """

    input_samples: np.ndarray
    output_samples: np.ndarray
    would_clip: bool = False
    input_window: np.ndarray | None = None
    output_window: np.ndarray | None = None
    input_would_clip: bool = False


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
        sample_rate: float = DEFAULT_SAMPLE_RATE_HZ,
        fft_size: int = DEFAULT_FFT_SIZE_SAMPLES,
        channels: int = 1,
        device: str | int | None = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.fft_size = int(fft_size)
        self.channels = int(channels)
        self.device = device

        self.bypass = True
        self.max_gain_db = DEFAULT_MAX_GAIN_DB
        self.epsilon_base = DEFAULT_EPSILON_RATIO
        # Estado de presentacion de la entrada; el callback nunca lo consulta.
        self.spl_offset_db: float | None = None
        self.weighting: str = "Z"
        # Limitador suave antes del DAC; solo se apaga para medir el filtro puro.
        self.limiter_enabled = True

        self.h_response: np.ndarray | None = None
        self.coherence: np.ndarray | None = None
        self.calibration_path: Path | None = None
        self.calibration_synthetic = False
        self.calibration_frequency_range_hz: tuple[float, float] | None = None
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
        self._reset_display_buffers()

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

    @property
    def algorithmic_latency_samples(self) -> int:
        """Retardo WOLA en muestras, sin incluir buffers del dispositivo."""
        return self.hop_size

    @property
    def algorithmic_latency_seconds(self) -> float:
        """Retardo WOLA en segundos, sin incluir buffers del dispositivo."""
        return self.algorithmic_latency_samples / self.sample_rate

    def _create_equalizer(self) -> OverlapAddFFTEqualizer:
        return OverlapAddFFTEqualizer(
            h_inverse=np.ones(self.bin_count, dtype=np.complex128),
            fft_size=self.fft_size,
        )

    def _reset_display_buffers(self) -> None:
        """Preasigna los buffers circulares que alimentan los espectros."""
        self._pending_display_input = np.zeros(self.hop_size, dtype=np.float64)
        self._input_ring = np.zeros(self.fft_size, dtype=np.float64)
        self._output_ring = np.zeros(self.fft_size, dtype=np.float64)
        self._input_ring_write = 0
        self._output_ring_write = 0
        self._input_window = np.zeros(self.fft_size, dtype=np.float64)
        self._output_window = np.zeros(self.fft_size, dtype=np.float64)

    @staticmethod
    def _write_ring(ring: np.ndarray, write: int, samples: np.ndarray) -> int:
        """Copia ``samples`` al buffer circular sin reservar memoria."""
        end = write + samples.size
        if end <= ring.size:
            ring[write:end] = samples
        else:
            split = ring.size - write
            ring[write:] = samples[:split]
            ring[: end - ring.size] = samples[split:]
        return end % ring.size

    @staticmethod
    def _snapshot_ring(ring: np.ndarray, start: int, target: np.ndarray) -> np.ndarray:
        """Devuelve una copia ordenada (mas antigua primero) del anillo."""
        tail = ring.size - start
        target[:tail] = ring[start:]
        target[tail:] = ring[:start]
        return target.copy()

    # ------------------------------------------------------------- calibracion
    def load_calibration(
        self,
        h_path: str | Path,
        coherence_path: str | Path | None = None,
    ) -> None:
        """Carga H(f) y coherencia solo si los datos y metadatos son compatibles."""
        response_path = Path(h_path)
        metadata_path = response_path.with_suffix(".json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"Faltan metadatos de calibracion ({metadata_path.name}); "
                "no se puede verificar fs, N ni procedencia."
            ) from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"No se pudieron leer los metadatos de calibracion "
                f"{metadata_path}: {error}"
            ) from error

        if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
            raise ValueError("metadata de calibracion debe usar format_version=1")
        metadata_rate = metadata.get("sample_rate_hz")
        metadata_size = metadata.get("fft_size_samples")
        synthetic = metadata.get("synthetic")
        frequency_range = metadata.get("frequency_range_hz")
        if (
            isinstance(metadata_rate, bool)
            or not isinstance(metadata_rate, (int, float))
            or not np.isfinite(metadata_rate)
            or not np.isclose(float(metadata_rate), self.sample_rate, rtol=0.0, atol=1e-6)
        ):
            raise ValueError(
                f"sample_rate_hz incompatible: se espera {self.sample_rate:g} Hz"
            )
        if (
            isinstance(metadata_size, bool)
            or not isinstance(metadata_size, int)
            or metadata_size != self.fft_size
        ):
            raise ValueError(
                f"fft_size_samples incompatible: se espera {self.fft_size}"
            )
        if not isinstance(synthetic, bool):
            raise ValueError("metadata.synthetic debe ser true o false")
        if (
            not isinstance(frequency_range, list)
            or len(frequency_range) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                for value in frequency_range
            )
        ):
            raise ValueError("metadata.frequency_range_hz debe ser [inicio, fin]")
        frequency_min_hz, frequency_max_hz = map(float, frequency_range)
        if not (
            0.0 <= frequency_min_hz < frequency_max_hz <= self.nyquist
        ):
            raise ValueError(
                "metadata.frequency_range_hz debe estar dentro de 0..Nyquist"
            )

        if coherence_path is None:
            coherence_stem = response_path.stem.replace("_H", "_gamma2")
            if coherence_stem == response_path.stem:
                coherence_stem = f"{response_path.stem}_gamma2"
            candidate = response_path.with_name(f"{coherence_stem}.npy")
            if candidate.exists():
                coherence_path = candidate

        h = np.load(response_path, allow_pickle=False)
        if h.dtype.kind != "c":
            raise ValueError(f"{response_path} debe contener H(f) compleja")
        if h.ndim != 1 or h.size != self.bin_count:
            raise ValueError(
                f"{response_path} debe contener {self.bin_count} bins para "
                f"fft_size={self.fft_size}"
            )
        if not np.all(np.isfinite(h)):
            raise ValueError(f"{response_path} debe contener solo valores finitos")

        coherence: np.ndarray | None = None
        if coherence_path is not None:
            coherence = np.load(coherence_path, allow_pickle=False)
            if coherence.dtype.kind != "f":
                raise ValueError(f"{coherence_path} debe contener coherencia real")
            if coherence.ndim != 1 or coherence.size != self.bin_count:
                raise ValueError(
                    f"{coherence_path} debe contener {self.bin_count} bins "
                    f"para fft_size={self.fft_size}"
                )
            if not np.all(np.isfinite(coherence)) or np.any(
                (coherence < 0.0) | (coherence > 1.0)
            ):
                raise ValueError(f"{coherence_path} debe contener gamma² entre 0 y 1")

        validated_h = np.asarray(h, dtype=np.complex128)
        validated_coherence = (
            None if coherence is None else np.asarray(coherence, dtype=np.float64)
        )
        # Build before publishing so any numerical failure leaves the old
        # calibration and filter untouched.
        inverse = self._estimate_inverse(validated_h, validated_coherence)
        self.h_response = validated_h
        self.coherence = validated_coherence
        self.calibration_path = response_path
        self.calibration_synthetic = synthetic
        self.calibration_frequency_range_hz = (
            frequency_min_hz,
            frequency_max_hz,
        )
        self._calibration_sample_rate = self.sample_rate
        self.equalizer.h_inverse = inverse
        self.revision += 1

    def clear_calibration(self) -> None:
        """Olvida la medicion y vuelve a un filtro plano."""
        self.h_response = None
        self.coherence = None
        self.calibration_path = None
        self.calibration_synthetic = False
        self.calibration_frequency_range_hz = None
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
        h_inverse = self._estimate_inverse(self.h_response, self.coherence)
        # Asignacion atomica: el callback nunca ve un array a medio construir.
        self.equalizer.h_inverse = h_inverse
        self.revision += 1

    def _estimate_inverse(
        self,
        h_response: np.ndarray | None,
        coherence: np.ndarray | None,
    ) -> np.ndarray:
        """Calcula el inverso localmente para publicar solo resultados completos."""
        if self.bypass or h_response is None:
            return np.ones(self.bin_count, dtype=np.complex128)

        gamma_squared = (
            np.zeros(self.bin_count, dtype=np.float64)
            if coherence is None
            else coherence
        )
        frequencies = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        upper = min(10_000.0, 0.95 * self.nyquist)
        reference_band = (frequencies >= 100.0) & (frequencies <= upper)
        if not np.any(reference_band):
            reference_band = np.ones(self.bin_count, dtype=bool)

        # Normalize only the reference band so absolute playback and mic gain
        # do not change the A/B level; coherence still gates every bin.
        normalized_h = h_response
        reference = float(np.median(np.abs(normalized_h[reference_band])))
        if np.isfinite(reference) and reference > 0.0:
            normalized_h = normalized_h / reference

        return estimate_h_inverse(
            normalized_h,
            gamma_squared,
            epsilon_base=self.epsilon_base,
            max_gain_db=self.max_gain_db,
            peak_percentile=50.0,
            reference_mask=reference_band,
        )

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
        new_size = self.fft_size if fft_size is None else int(fft_size)
        new_sample_rate = self.sample_rate if sample_rate is None else float(sample_rate)
        if new_size < 4 or new_size % 2:
            raise ValueError("fft_size debe ser un entero par mayor o igual a 4")
        if not np.isfinite(new_sample_rate) or new_sample_rate <= 0.0:
            raise ValueError("sample_rate debe ser finita y positiva")

        calibration_discarded = self.h_response is not None and (
            new_size != self.fft_size or new_sample_rate != self.sample_rate
        )
        if new_size != self.fft_size:
            self.fft_size = new_size
            self.equalizer = self._create_equalizer()
        self.sample_rate = new_sample_rate
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
        if os.environ.get("FC_DEBUG_AUDIO") == "1":
            input_id, output_id = self._stream.device
            print(f"[audio] entrada {input_id}: {sd.query_devices(input_id)['name']}")
            print(f"[audio] salida  {output_id}: {sd.query_devices(output_id)['name']}")

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
        self.equalizer.reset()
        self._reset_display_buffers()
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
            self.equalizer.reset()
            self._reset_display_buffers()
            put_latest(self.status_queue, f"bloque inesperado: {frames} muestras")
            return

        output_block = self.equalizer.process_block(input_block)
        if output_block.size == 0:  # preroll WOLA
            self._pending_display_input = input_block.copy()
            outdata.fill(0.0)
            return
        if not np.all(np.isfinite(output_block)):
            outdata.fill(0.0)
            self.equalizer.reset()
            self._reset_display_buffers()
            put_latest(self.status_queue, "Salida NaN/Inf: bloque silenciado")
            return

        aligned_input = self._pending_display_input
        self._pending_display_input = input_block.copy()
        would_clip = bool(np.max(np.abs(output_block)) >= CLIP_THRESHOLD)
        input_would_clip = bool(np.max(np.abs(aligned_input)) >= CLIP_THRESHOLD)

        # Anillos preasignados: la GUI recibe siempre la ultima ventana
        # completa de fft_size muestras, sin depender de que no se pierda
        # ningun frame en la cola.
        self._input_ring_write = self._write_ring(
            self._input_ring, self._input_ring_write, aligned_input
        )
        self._output_ring_write = self._write_ring(
            self._output_ring, self._output_ring_write, output_block
        )
        input_window = self._snapshot_ring(
            self._input_ring, self._input_ring_write, self._input_window
        )
        output_window = self._snapshot_ring(
            self._output_ring, self._output_ring_write, self._output_window
        )

        playback = (
            soft_limit(output_block, LIMITER_THRESHOLD, OUTPUT_CEILING)
            if self.limiter_enabled
            else output_block
        )
        outdata[:] = playback.astype(outdata.dtype, copy=False)[:, np.newaxis]
        put_latest(
            self.display_queue,
            SpectrumFrame(
                input_samples=aligned_input,
                output_samples=output_block.copy(),
                would_clip=would_clip,
                input_window=input_window,
                output_window=output_window,
                input_would_clip=input_would_clip,
            ),
        )
