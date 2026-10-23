"""Ventana principal del ecualizador FFT en tiempo real.

La ventana es el unico lugar donde se dibuja. El motor de audio se encarga
del stream y del filtro; aqui solo se leen sus resultados y se le envian los
cambios que el usuario pide desde el panel de control.
"""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from src.audio.engine import AudioEngine, CalibrationDiscardedError
from src.dsp.spectrum import DB_CEILING, DB_FLOOR, gain_db, level_db, spectrum_db
from src.ui import theme
from src.ui.controls import ControlPanel
from src.ui.meters import MeterPanel
from src.ui.readout import CursorReadout
from src.ui.spectrogram import SpectrogramWidget

REFRESH_MS = 33  # ~30 FPS


class MainWindow(QtWidgets.QMainWindow):
    """Interfaz grafica completa del ecualizador."""

    def __init__(self, engine: AudioEngine, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self._filter_revision = -1
        self._filter_db = np.zeros(0)
        self._frequencies = np.zeros(0)
        self._input_db = np.zeros(0)
        self._output_db = np.zeros(0)

        self.setWindowTitle("Ecualizador FFT en tiempo real")
        self.resize(1_320, 880)

        self._build_ui()
        self._connect_signals()
        self._sync_from_engine()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_MS)

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical.addWidget(self._build_top_row())
        vertical.addWidget(self._build_spectrogram())
        vertical.setStretchFactor(0, 3)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([560, 280])
        outer.addWidget(vertical)

        self.statusBar().showMessage("Listo")

    def _build_top_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        self.controls = ControlPanel()
        self.meters = MeterPanel()
        side_layout.addWidget(self.controls)
        side_layout.addWidget(self.meters)
        side_layout.addStretch(1)
        row.addWidget(side)

        graphs = QtWidgets.QWidget()
        graphs_layout = QtWidgets.QVBoxLayout(graphs)
        graphs_layout.setContentsMargins(0, 0, 0, 0)
        graphs_layout.setSpacing(4)

        self.readout_label = QtWidgets.QLabel("Pase el raton sobre las graficas para leer los valores.")
        self.readout_label.setStyleSheet(f"color: {theme.MUTED}; padding: 2px 4px;")
        graphs_layout.addWidget(self.readout_label)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(theme.BACKGROUND)
        graphs_layout.addWidget(self.graphics)

        self.input_plot = self.graphics.addPlot(title="Entrada: senal captada")
        self.input_plot.setLabel("left", "Magnitud", units="dBFS")
        self.input_plot.setLabel("bottom", "Frecuencia", units="Hz")
        self.input_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self.input_plot.showGrid(x=True, y=True, alpha=0.2)
        self.input_curve = self.input_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.5), name="Entrada"
        )
        self.input_plot.addLegend(offset=(12, 12), labelTextColor=theme.MUTED)

        self.graphics.nextRow()

        self.output_plot = self.graphics.addPlot(title="Salida: senal ecualizada")
        self.output_plot.setXLink(self.input_plot)
        self.output_plot.setLabel("left", "Magnitud", units="dBFS")
        self.output_plot.setLabel("bottom", "Frecuencia", units="Hz")
        self.output_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self.output_plot.showGrid(x=True, y=True, alpha=0.2)
        self.output_curve = self.output_plot.plot(
            pen=pg.mkPen(theme.OUTPUT, width=1.5), name="Salida"
        )
        self.filter_curve = self.output_plot.plot(
            pen=pg.mkPen(theme.FILTER, width=1.0, style=QtCore.Qt.DashLine),
            name="Filtro inverso (dB)",
        )
        self.output_plot.addLegend(offset=(12, 12), labelTextColor=theme.MUTED)

        self.readout = CursorReadout([self.input_plot, self.output_plot])

        row.addWidget(graphs)
        row.setStretchFactor(0, 0)
        row.setStretchFactor(1, 1)
        row.setSizes([320, 980])
        return row

    def _build_spectrogram(self) -> SpectrogramWidget:
        self.spectrogram = SpectrogramWidget()
        return self.spectrogram

    # ------------------------------------------------------------- senales
    def _connect_signals(self) -> None:
        self.controls.modeChanged.connect(self._on_mode_changed)
        self.controls.calibrationRequested.connect(self._load_calibration)
        self.controls.clearCalibrationRequested.connect(self._clear_calibration)
        self.controls.gainChanged.connect(self._on_gain_changed)
        self.controls.epsilonChanged.connect(self._on_epsilon_changed)
        self.controls.deviceChanged.connect(self._on_device_changed)
        self.controls.formatChanged.connect(self._on_format_changed)
        self.graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _sync_from_engine(self) -> None:
        """Vuelca el estado inicial del motor en los controles."""
        self.controls.set_format(self.engine.sample_rate, self.engine.fft_size)
        self.controls.set_mode(self.engine.bypass)
        self.controls.set_gain(self.engine.max_gain_db)
        self.controls.set_epsilon(self.engine.epsilon_base)
        self.controls.set_calibration(
            str(self.engine.calibration_path.name) if self.engine.calibration_path else None
        )
        self._rebuild_frequency_axis()
        self._update_mode_status()

    def _rebuild_frequency_axis(self) -> None:
        self._frequencies = np.fft.rfftfreq(
            self.engine.fft_size, d=1.0 / self.engine.sample_rate
        )
        self.spectrogram.reconfigure(
            self.engine.bin_count,
            self.engine.nyquist,
            self.engine.seconds_per_row,
        )
        self.input_plot.setXRange(0.0, self.engine.nyquist, padding=0)

    # ------------------------------------------------------------- acciones
    def _on_mode_changed(self, bypass: bool) -> None:
        self.engine.bypass = bypass
        self.engine.recompute_inverse()
        self._update_mode_status()

    def _on_gain_changed(self, db: float) -> None:
        self.engine.max_gain_db = db
        self.engine.recompute_inverse()
        self._update_mode_status()

    def _on_epsilon_changed(self, value: float) -> None:
        self.engine.epsilon_base = value
        self.engine.recompute_inverse()
        self._update_mode_status()

    def _on_device_changed(self, device: Any) -> None:
        try:
            self.engine.configure(device=device)
        except Exception as error:  # noqa: BLE001 - el panel no debe romperse
            self._set_status(f"No se pudo abrir el dispositivo: {error}")
            self.controls.set_device(self.engine.device)
            return
        self._set_status("Dispositivo actualizado")

    def _on_format_changed(self, sample_rate: float, fft_size: int) -> None:
        try:
            self.engine.configure(sample_rate=sample_rate, fft_size=fft_size)
        except CalibrationDiscardedError as error:
            # El formato nuevo SI se aplico y el audio sigue corriendo: solo se
            # perdio la calibracion. Hay que rehacer los ejes para el nuevo FFT.
            self.controls.set_format(self.engine.sample_rate, self.engine.fft_size)
            self.controls.set_calibration(None)
            self._rebuild_frequency_axis()
            self._update_mode_status()
            self._set_status(str(error))
            return
        except ValueError as error:
            # Formato invalido: el motor no cambio nada.
            self._set_status(str(error))
            self.controls.set_format(self.engine.sample_rate, self.engine.fft_size)
            return
        except Exception as error:  # noqa: BLE001
            # El formato se aplico pero el stream no pudo abrirse.
            self._set_status(f"No se pudo reiniciar el audio: {error}")
            self.controls.set_format(self.engine.sample_rate, self.engine.fft_size)
            self.controls.set_calibration(
                self.engine.calibration_path.name if self.engine.calibration_path else None
            )
            self._rebuild_frequency_axis()
            self._update_mode_status()
            return
        self._rebuild_frequency_axis()
        self._update_mode_status()
        self._set_status("Formato actualizado")

    def _load_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Seleccionar la respuesta medida H(f)",
            "",
            "Archivos NumPy (*.npy)",
        )
        if not path:
            return

        coherence_path: Path | None = None
        candidate = Path(path).with_name("medicion_gamma2.npy")
        if candidate.exists():
            answer = QtWidgets.QMessageBox.question(
                self,
                "Coherencia",
                f"Se encontro {candidate.name} junto a la medicion.\n¿Usarlo como gamma squared?",
            )
            if answer == QtWidgets.QMessageBox.Yes:
                coherence_path = candidate

        try:
            self.engine.load_calibration(path, coherence_path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "Calibracion invalida", str(error))
            return

        self.controls.set_calibration(Path(path).name)
        self._update_mode_status()
        self._set_status("Calibracion cargada")

    def _clear_calibration(self) -> None:
        self.engine.clear_calibration()
        self.controls.set_calibration(None)
        self._update_mode_status()
        self._set_status("Calibracion descartada")

    def _update_mode_status(self) -> None:
        if self.engine.bypass:
            text = "Modo bypass: filtro plano"
        elif self.engine.is_calibrated:
            text = f"Modo ecualizado (ganancia max {self.engine.max_gain_db:+.0f} dB)"
        else:
            text = "Modo ecualizado sin calibracion: filtro plano"
        self.controls.set_status(text)

    # -------------------------------------------------------------- refresco
    def _refresh(self) -> None:
        latest = None
        while True:
            try:
                latest = self.engine.display_queue.get_nowait()
            except queue.Empty:
                break

        while True:
            try:
                self._set_status(self.engine.status_queue.get_nowait())
            except queue.Empty:
                break

        if latest is None:
            return

        self._input_db = spectrum_db(latest.input_samples, self.engine.fft_size)
        self._output_db = spectrum_db(latest.output_samples, self.engine.fft_size)
        self.input_curve.setData(self._frequencies, self._input_db)
        self.output_curve.setData(self._frequencies, self._output_db)
        self._update_filter_curve()
        self.spectrogram.push(self._output_db)

        in_rms, in_peak = level_db(latest.input_samples)
        out_rms, out_peak = level_db(latest.output_samples)
        self.meters.set_input(in_rms, in_peak)
        self.meters.set_output(out_rms, out_peak)

        self.readout.set_spectra(
            self._frequencies,
            {"entrada": self._input_db, "salida": self._output_db},
        )

    def _update_filter_curve(self) -> None:
        if self._filter_revision != self.engine.revision:
            self._filter_revision = self.engine.revision
            self._filter_db = gain_db(self.engine.equalizer.h_inverse)
        if self._filter_db.size == self._frequencies.size:
            self.filter_curve.setData(self._frequencies, self._filter_db)

    def _on_mouse_moved(self, position: QtCore.QPointF) -> None:
        text = self.readout.handle_mouse(position)
        if text:
            self.readout_label.setText(text)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    # -------------------------------------------------------------- cierre
    def closeEvent(self, event: Any) -> None:  # noqa: N802 (Qt)
        self._timer.stop()
        self.engine.close()
        super().closeEvent(event)
