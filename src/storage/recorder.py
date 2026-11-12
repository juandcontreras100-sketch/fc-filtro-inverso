"""Grabacion de sesiones del ecualizador en disco.

La grabadora se usa **desde el hilo de la GUI**: recibe los espectros y los
niveles ya calculados y los escribe en disco. Nunca se llama desde el callback
de audio, que debe seguir siendo hard-real-time.

Cada sesion vive en ``sesiones/AAAA-MM-DD_HHMMSS/`` y contiene:

- ``metadata.json``: formato y estado del motor al iniciar la grabacion.
- ``niveles.csv``: una fila por frame de la GUI.
- ``espectros.npz``: espectros de entrada y salida decimados a ~10 Hz.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Encabezado exacto del registro de niveles.
CSV_HEADER = (
    "time_s",
    "input_rms_db",
    "input_peak_db",
    "output_rms_db",
    "output_peak_db",
    "clip",
)
# Intervalo minimo entre espectros guardados (decimacion a ~10 Hz). Se usa
# 0.09 s y no 0.1 s para que un refresco de ~30 FPS (33 ms) guarde una fila
# cada tres frames en vez de saltear el limite por redondeo.
SPECTRUM_INTERVAL_S = 0.09
# Cada cuanto se vacia lo acumulado a disco durante una sesion larga.
FLUSH_INTERVAL_S = 2.0


def _json_device(device: Any) -> Any:
    """Convierte el dispositivo del motor en algo serializable a JSON.

    ``sounddevice`` acepta ``None``, un indice, un nombre o una tupla
    ``(entrada, salida)``; en JSON las tuplas se guardan como listas.
    """
    if device is None or isinstance(device, str):
        return device
    if isinstance(device, (bool, np.bool_)):
        return str(device)
    if isinstance(device, (int, np.integer)):
        return int(device)
    if isinstance(device, (tuple, list)):
        return [_json_device(item) for item in device]
    return str(device)


class SessionRecorder:
    """Graba una sesion de la GUI en una carpeta autocontenida.

    El reloj es inyectable para que las pruebas sean deterministas y para
    compartir la misma nocion de tiempo entre ``start``, los frames y la
    decimacion de espectros.
    """

    def __init__(
        self,
        engine: Any,
        root_dir: str | Path = "sesiones",
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.engine = engine
        self.root_dir = Path(root_dir)
        self._clock = clock

        self._directory: Path | None = None
        self._recording = False
        self._csv_handle: Any = None
        self._csv_writer: Any = None
        self._started_at = 0.0
        self._last_spectrum_at: float | None = None
        self._last_flush_at = 0.0

        # Formato congelado al iniciar: si el motor se reconfigura a mitad de
        # sesion, las filas nuevas no deben mezclar bins de distinto tamano.
        self._bin_count = 0
        self._frequencies = np.empty(0, dtype=np.float64)
        self._input_frames: list[np.ndarray] = []
        self._output_frames: list[np.ndarray] = []

    # ---------------------------------------------------------------- estado
    @property
    def is_recording(self) -> bool:
        """True mientras haya una sesion abierta."""
        return self._recording

    @property
    def directory(self) -> Path | None:
        """Carpeta de la sesion actual o de la ultima grabada."""
        return self._directory

    # ------------------------------------------------------------- grabacion
    def start(self) -> Path:
        """Crea la carpeta y los archivos de una sesion nueva.

        Devuelve el directorio creado. Si ya habia una grabacion en curso
        lanza ``RuntimeError`` para no pisar la sesion anterior.
        """
        if self._recording:
            raise RuntimeError("Ya hay una grabacion de sesion en curso")

        self._reset_state()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        directory = self._new_directory()
        directory.mkdir()

        # Se congelan formato y estado del motor: el archivo describe la
        # sesion tal como empezo, aunque la GUI cambie algo despues.
        self._bin_count = int(self.engine.bin_count)
        self._frequencies = np.fft.rfftfreq(
            int(self.engine.fft_size),
            1.0 / float(self.engine.sample_rate),
        )
        metadata = self._metadata()
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        csv_handle = (directory / "niveles.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        )
        try:
            writer = csv.writer(csv_handle)
            writer.writerow(CSV_HEADER)
            csv_handle.flush()
        except Exception:
            csv_handle.close()
            raise

        self._directory = directory
        self._csv_handle = csv_handle
        self._csv_writer = writer
        now = self._clock()
        self._started_at = now
        self._last_flush_at = now
        self._last_spectrum_at = None
        self._recording = True
        return directory

    def record_frame(
        self,
        *,
        input_db: Any,
        output_db: Any,
        input_rms_db: float,
        input_peak_db: float,
        output_rms_db: float,
        output_peak_db: float,
        clip: Any,
    ) -> None:
        """Registra un frame de la GUI: una fila de niveles y, si toca, un espectro.

        Los espectros se copian para no retener los buffers mutables que la
        interfaz reutiliza. Si su tamano no coincide con los bins del motor, se
        descarta solo el espectro; la fila de niveles se guarda igual.
        """
        if not self._recording:
            return
        now = self._clock()

        self._csv_writer.writerow(
            (
                f"{now - self._started_at:.6f}",
                f"{float(input_rms_db):.6f}",
                f"{float(input_peak_db):.6f}",
                f"{float(output_rms_db):.6f}",
                f"{float(output_peak_db):.6f}",
                int(bool(clip)),
            )
        )
        self._capture_spectra(now, input_db, output_db)

        if now - self._last_flush_at >= FLUSH_INTERVAL_S:
            self._flush()
            self._last_flush_at = now

    def stop(self) -> Path | None:
        """Vacia lo pendiente, cierra el CSV y escribe el NPZ final.

        Devuelve el directorio de la sesion o ``None`` si no habia una
        grabacion activa.
        """
        if not self._recording:
            return None

        directory = self._directory
        try:
            self._flush()
        finally:
            self._close_csv()
            self._recording = False
            self._input_frames = []
            self._output_frames = []
            self._last_spectrum_at = None
        return directory

    # -------------------------------------------------------------- internos
    def _metadata(self) -> dict[str, Any]:
        """Estado del motor que identifica la sesion."""
        calibration = self.engine.calibration_path
        return {
            "sample_rate": float(self.engine.sample_rate),
            "fft_size": int(self.engine.fft_size),
            "device": _json_device(self.engine.device),
            "epsilon_base": float(self.engine.epsilon_base),
            "max_gain_db": float(self.engine.max_gain_db),
            "calibration_file": (
                None if calibration is None else Path(calibration).name
            ),
            "mode": "bypass" if bool(self.engine.bypass) else "ecualizado",
        }

    def _new_directory(self) -> Path:
        """Nombre con fecha/hora y sufijo ``_01``, ``_02``... si ya existe."""
        base = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        candidate = self.root_dir / base
        suffix = 1
        while candidate.exists():
            candidate = self.root_dir / f"{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def _capture_spectra(
        self,
        now: float,
        input_db: Any,
        output_db: Any,
    ) -> None:
        """Guarda el espectro si pasaron 0.1 s desde el ultimo y encaja en formato."""
        if self._last_spectrum_at is not None:
            if now - self._last_spectrum_at < SPECTRUM_INTERVAL_S:
                return

        input_values = np.asarray(input_db, dtype=float).copy()
        output_values = np.asarray(output_db, dtype=float).copy()
        if (
            input_values.ndim != 1
            or output_values.ndim != 1
            or input_values.size != self._bin_count
            or output_values.size != self._bin_count
        ):
            return

        self._input_frames.append(input_values)
        self._output_frames.append(output_values)
        self._last_spectrum_at = now

    def _flush(self) -> None:
        """Vacia el CSV y reescribe el NPZ con lo acumulado hasta ahora."""
        if self._csv_handle is not None:
            self._csv_handle.flush()
        self._write_spectra()

    def _write_spectra(self) -> None:
        """Escribe ``espectros.npz`` completo; con cero frames usa forma ``(0, bins)``."""
        if self._directory is None:
            return
        if self._input_frames:
            input_matrix = np.vstack(self._input_frames)
            output_matrix = np.vstack(self._output_frames)
        else:
            input_matrix = np.empty((0, self._bin_count), dtype=np.float64)
            output_matrix = np.empty((0, self._bin_count), dtype=np.float64)
        np.savez(
            self._directory / "espectros.npz",
            frequencies=self._frequencies,
            input_db=input_matrix,
            output_db=output_matrix,
        )

    def _close_csv(self) -> None:
        if self._csv_handle is None:
            return
        try:
            self._csv_handle.close()
        finally:
            self._csv_handle = None
            self._csv_writer = None

    def _reset_state(self) -> None:
        """Deja la grabadora lista para una sesion nueva."""
        self._close_csv()
        self._directory = None
        self._input_frames = []
        self._output_frames = []
        self._last_spectrum_at = None
