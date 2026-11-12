"""Ventana principal del ecualizador FFT en tiempo real.

La ventana es el unico lugar donde se dibuja. El motor de audio se encarga
del stream y del filtro; aqui solo se leen sus resultados y se le envian los
cambios que el usuario pide desde el panel de control.
"""

from __future__ import annotations

import csv
import queue
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtWidgets

from src.audio.engine import AudioEngine, CalibrationDiscardedError
from src.dsp.spectrum import (
    DB_CEILING,
    DB_FLOOR,
    dbfs_to_spl,
    gain_db,
    level_db,
    spectrum_db,
    total_level_db,
)
from src.storage.recorder import SessionRecorder
from src.ui import theme
from src.ui.calibration import CalibrationDialog
from src.ui.controls import ControlPanel
from src.ui.meters import MeterPanel
from src.ui.readout import CursorReadout
from src.ui.spectrogram import SpectrogramWidget
from src.ui.spl_context import spl_context_text, spl_zone_color
from src.ui.spl_persistence import input_device_name, save_spl_calibration

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
        self._measured_db = np.zeros(0)
        self._inverse_db = np.zeros(0)
        self._product_db = np.zeros(0)
        self._input_total_db = DB_FLOOR
        self._has_input_total = False
        self._recorder: SessionRecorder | None = None
        self._calibration_dialog: CalibrationDialog | None = None
        self._input_history = np.zeros(engine.fft_size)
        self._output_history = np.zeros(engine.fft_size)
        # Peso del espectro anterior en el promedio exponencial de la GUI.
        self.spectrum_averaging = 0.8

        self.setWindowTitle("Ecualizador FFT en tiempo real")
        self.resize(1_360, 960)
        theme.apply_theme(self)

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
        vertical.setStretchFactor(1, 1)
        vertical.setSizes([700, 220])
        outer.addWidget(vertical)

        self.statusBar().showMessage("Listo")

    def _build_top_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(300)
        side.setMaximumWidth(370)
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 4, 0)
        side_layout.setSpacing(6)

        self.meters = MeterPanel()
        self.controls = ControlPanel()

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_scroll.setWidget(self.controls)

        side_layout.addWidget(self.meters)
        side_layout.addWidget(controls_scroll, 1)
        side_layout.addWidget(self.controls.session_box)
        row.addWidget(side)

        graphs = QtWidgets.QWidget()
        graphs_layout = QtWidgets.QVBoxLayout(graphs)
        graphs_layout.setContentsMargins(4, 0, 0, 0)
        graphs_layout.setSpacing(6)

        self.readout_label = QtWidgets.QLabel(
            "Cursor: pase el raton sobre las graficas para inspeccionar frecuencia y nivel."
        )
        self.readout_label.setFont(theme.mono_font(9))
        self.readout_label.setStyleSheet(
            f"background-color: {theme.PANEL}; color: {theme.TEXT}; "
            f"border: 1px solid {theme.PANEL_EDGE}; border-radius: 5px; padding: 5px 10px;"
        )
        graphs_layout.addWidget(self.readout_label)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(theme.BACKGROUND)
        graphs_layout.addWidget(self.graphics)

        self.input_plot = self.graphics.addPlot(title="Entrada: senal captada")
        theme.style_plot(self.input_plot, "Entrada: senal captada")
        self.input_plot.setLogMode(x=True, y=False)
        self.input_plot.setLabel("left", "Magnitud", units="dBFS")
        self.input_plot.setLabel("bottom", "Frecuencia", units="Hz")
        self.input_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self.input_plot.showGrid(x=True, y=True, alpha=0.22)
        self.input_curve = self.input_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.6),
            fillLevel=DB_FLOOR,
            brush=pg.mkBrush(*theme.INPUT_FILL),
            name="Entrada",
        )
        self.input_plot.addLegend(offset=(12, 12), labelTextColor=theme.MUTED)

        self.graphics.nextRow()

        self.output_plot = self.graphics.addPlot(title="Salida: senal ecualizada")
        theme.style_plot(self.output_plot, "Salida: senal ecualizada")
        self.output_plot.setLogMode(x=True, y=False)
        self.output_plot.setLabel("left", "Magnitud", units="dBFS")
        self.output_plot.setLabel("bottom", "Frecuencia", units="Hz")
        self.output_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self.output_plot.showGrid(x=True, y=True, alpha=0.22)
        self.output_plot.showAxis("right")
        self.output_plot.getAxis("right").setLabel(
            "Ganancia del filtro", units="dB", color=theme.FILTER
        )
        self.output_curve = self.output_plot.plot(
            pen=pg.mkPen(theme.OUTPUT, width=1.6),
            fillLevel=DB_FLOOR,
            brush=pg.mkBrush(*theme.OUTPUT_FILL),
            name="Salida",
        )
        self.filter_view = pg.ViewBox(enableMenu=False)
        self.output_plot.scene().addItem(self.filter_view)
        self.output_plot.getAxis("right").linkToView(self.filter_view)
        self.filter_view.setXLink(self.output_plot)
        self.filter_view.setLogMode("x", True)
        self.filter_view.setMouseEnabled(x=False, y=False)
        self.filter_view.setYRange(-60.0, 6.0, padding=0)
        self.output_plot.getViewBox().sigResized.connect(
            self._update_filter_view_geometry
        )
        self.filter_curve = pg.PlotDataItem(
            pen=pg.mkPen(theme.FILTER, width=1.2, style=QtCore.Qt.DashLine)
        )
        self.filter_curve.setLogMode(True, False)
        self.filter_view.addItem(self.filter_curve)
        output_legend = self.output_plot.addLegend(
            offset=(12, 12), labelTextColor=theme.MUTED
        )
        output_legend.addItem(self.filter_curve, "Filtro inverso")
        self._update_filter_view_geometry()

        self.graphics.nextRow()
        self.filter_plot = self.graphics.addPlot(title="Respuesta del filtro")
        theme.style_plot(self.filter_plot, "Respuesta del filtro")
        self.filter_plot.setLabel("left", "Ganancia", units="dB")
        self.filter_plot.setLabel("bottom", "Frecuencia", units="Hz")
        self.filter_plot.setLogMode(x=True, y=False)
        self.filter_plot.setXLink(self.input_plot)
        self.filter_plot.showGrid(x=True, y=True, alpha=0.22)
        self.filter_plot.setYRange(-6.0, 6.0, padding=0)
        self.filter_plot.addItem(
            pg.InfiniteLine(
                pos=0.0,
                angle=0,
                movable=False,
                pen=pg.mkPen(theme.PANEL_EDGE, width=1, style=QtCore.Qt.DotLine),
            )
        )
        self.filter_plot.addLegend(offset=(12, 12), labelTextColor=theme.MUTED)
        self.h_measured_curve = self.filter_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.5), name="|H| medida"
        )
        self.h_inverse_curve = self.filter_plot.plot(
            pen=pg.mkPen(theme.FILTER, width=1.5), name="|H inversa|"
        )
        self.h_product_curve = self.filter_plot.plot(
            pen=pg.mkPen(theme.OUTPUT, width=1.6), name="|H x H inversa|"
        )

        self.readout = CursorReadout(
            [self.input_plot, self.output_plot, self.filter_plot], log_x=True
        )

        row.addWidget(graphs)
        row.setStretchFactor(0, 0)
        row.setStretchFactor(1, 1)
        row.setSizes([330, 970])
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
        self.controls.recordToggled.connect(self._on_record_toggled)
        self.controls.exportRequested.connect(self._on_export_requested)
        self.controls.splDisplayToggled.connect(self._on_spl_display_toggled)
        self.controls.splOffsetChanged.connect(self._on_spl_offset_changed)
        self.controls.splCalibrationRequested.connect(
            self._on_spl_calibration_requested
        )
        self.controls.weightingChanged.connect(self._on_weighting_changed)
        self.graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _sync_from_engine(self) -> None:
        """Vuelca el estado inicial del motor en los controles."""
        self.controls.set_format(self.engine.sample_rate, self.engine.fft_size)
        self.controls.set_mode(self.engine.bypass)
        self.controls.set_gain(self.engine.max_gain_db)
        self.controls.set_epsilon(self.engine.epsilon_base)
        if self.engine.spl_offset_db is not None:
            self.controls.set_spl_offset(self.engine.spl_offset_db)
        self.controls.set_spl_calibration(self.engine.spl_offset_db)
        self.controls.set_weighting(self.engine.weighting)
        self.controls.set_calibration(
            str(self.engine.calibration_path.name) if self.engine.calibration_path else None
        )
        self._rebuild_frequency_axis()
        self._update_input_display()
        self._update_total_level_label()
        self._update_mode_status()

    def _rebuild_frequency_axis(self) -> None:
        self._frequencies = np.fft.rfftfreq(
            self.engine.fft_size, d=1.0 / self.engine.sample_rate
        )
        self._input_history = np.zeros(self.engine.fft_size)
        self._output_history = np.zeros(self.engine.fft_size)
        self.spectrogram.reconfigure(
            self.engine.bin_count,
            self.engine.nyquist,
            self.engine.seconds_per_row,
        )
        upper = min(20_000.0, self.engine.nyquist)
        lower = 20.0 if upper > 20.0 else max(upper * 0.5, 1e-3)
        # En modo log el ViewBox trabaja con log10(Hz), tambien en setLimits.
        x_min, x_max = np.log10(lower), np.log10(upper)
        limits = {
            "xMin": np.log10(20.0),
            "xMax": np.log10(self.engine.nyquist),
            "yMin": DB_FLOOR,
            "yMax": DB_CEILING,
        }
        for plot in (self.input_plot, self.output_plot):
            plot.setMouseEnabled(x=False, y=False)
            plot.setLimits(**limits)
            plot.setXRange(x_min, x_max, padding=0)
        self._update_input_level_axis()
        # Su eje X esta enlazado a la entrada: arrastrarlo tambien la moveria.
        self.filter_plot.setMouseEnabled(x=False, y=False)
        self._update_filter_curve()

    def _update_filter_view_geometry(self) -> None:
        output_view = self.output_plot.getViewBox()
        self.filter_view.setGeometry(output_view.sceneBoundingRect())
        self.filter_view.linkedViewChanged(output_view, self.filter_view.XAxis)

    def _active_spl_offset(self) -> float | None:
        if not self.controls.spl_checkbox.isChecked():
            return None
        return self.engine.spl_offset_db

    def _update_input_level_axis(self) -> None:
        offset = self._active_spl_offset()
        if offset is None:
            lower, upper, unit = DB_FLOOR, DB_CEILING, "dBFS"
        else:
            lower = max(0.0, DB_FLOOR + offset)
            upper = DB_CEILING + offset
            unit = "dB SPL"
        self.input_plot.setLabel("left", "Magnitud", units=unit)
        self.input_plot.setLimits(yMin=lower, yMax=upper)
        self.input_plot.setYRange(lower, upper, padding=0)
        self.input_curve.setFillLevel(lower)

    def _update_input_display(self) -> None:
        input_values = self._input_db
        input_unit = "dBFS"
        offset = self._active_spl_offset()
        if offset is not None:
            input_values = dbfs_to_spl(input_values, offset)
            input_unit = "dB SPL"
        if input_values.size == self._frequencies.size:
            self.input_curve.setData(self._frequencies[1:], input_values[1:])
        self.readout.set_spectra(
            self._frequencies,
            {
                "entrada": input_values,
                "salida": self._output_db,
                "filtro": self._filter_db,
            },
            units={"entrada": input_unit, "salida": "dBFS", "filtro": "dB"},
        )

    def _update_total_level_label(self) -> None:
        if not self._has_input_total:
            self.controls.total_level_label.setText("—")
            self.controls.spl_context_label.setText("")
            self.controls.set_total_level_color()
            return
        value = self._input_total_db
        unit = "dBFS"
        context = ""
        color: str | None = None
        offset = self._active_spl_offset()
        if offset is not None:
            value = dbfs_to_spl(value, offset)
            unit = "dB SPL"
            context = spl_context_text(value)
            if self.engine.weighting == "A":
                color = spl_zone_color(value)
        self.controls.total_level_label.setText(
            f"{value:.1f} {unit} ({self.engine.weighting})"
        )
        self.controls.spl_context_label.setText(context)
        self.controls.set_total_level_color(color)

    # ------------------------------------------------------------- acciones
    def _on_mode_changed(self, bypass: bool) -> None:
        leaving_bypass = self.engine.bypass and not bypass
        self.engine.bypass = bypass
        self.engine.recompute_inverse()
        self._update_mode_status()
        if leaving_bypass:
            self.statusBar().showMessage(
                "Ecualización activa: vigile el nivel y la realimentación acústica.",
                8_000,
            )

    def _on_gain_changed(self, db: float) -> None:
        self.engine.max_gain_db = db
        self.engine.recompute_inverse()
        self._update_mode_status()

    def _on_epsilon_changed(self, value: float) -> None:
        self.engine.epsilon_base = value
        self.engine.recompute_inverse()
        self._update_mode_status()

    def _on_spl_display_toggled(self, active: bool) -> None:
        self.controls.set_spl_calibration(self.engine.spl_offset_db)
        self._update_input_level_axis()
        self._update_input_display()
        self._update_total_level_label()

    def _on_spl_offset_changed(self, value: float) -> None:
        self.engine.spl_offset_db = float(value)
        self.controls.set_spl_calibration(self.engine.spl_offset_db)
        self._update_input_level_axis()
        self._update_input_display()
        self._update_total_level_label()

    def _on_spl_calibration_requested(self) -> None:
        dialog = CalibrationDialog(self.engine.sample_rate, parent=self)
        self._calibration_dialog = dialog
        try:
            result = dialog.exec()
        finally:
            self._calibration_dialog = None
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            return

        offset_db = dialog.offset_db
        if offset_db is None:
            return
        self.engine.spl_offset_db = float(offset_db)
        self.controls.set_spl_offset(offset_db)
        self.controls.set_spl_calibration(offset_db)
        self.controls.set_spl_display(True)
        self._update_input_level_axis()
        self._update_input_display()
        self._update_total_level_label()
        try:
            save_spl_calibration(
                float(offset_db), input_device_name(self.engine.device)
            )
        except Exception as error:  # noqa: BLE001 - la calibracion ya esta aplicada
            self._set_status(
                f"Calibracion aplicada, pero no se pudo guardar: {error}"
            )
        else:
            self._set_status(
                f"Calibracion SPL guardada: offset {offset_db:.1f} dB"
            )

    def _on_weighting_changed(self, weighting: str) -> None:
        self.engine.weighting = weighting
        # El total anterior usaba otra ponderacion; se recalcula en el proximo bloque.
        self._has_input_total = False
        self._update_total_level_label()

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
            text, tip, state = "Bypass", "Filtro plano: la salida copia la entrada.", "bypass"
        elif self.engine.is_calibrated:
            text = f"Ecualizado {self.engine.max_gain_db:+.0f} dB"
            tip = "Filtro inverso activo. Vigile el nivel y la realimentacion."
            state = "active"
        else:
            text, tip, state = (
                "Sin calibracion",
                "Ecualizado pedido, pero no hay H(f): el filtro queda plano.",
                "warning",
            )
        self.controls.set_status(text, state=state)
        self.controls.status_label.setToolTip(tip)

    # -------------------------------------------------------------- refresco
    def _refresh(self) -> None:
        latest = None
        while True:
            try:
                frame = self.engine.display_queue.get_nowait()
            except queue.Empty:
                break
            if self._calibration_dialog is not None:
                self._calibration_dialog.add_block(frame.input_samples)
            latest = frame

        while True:
            try:
                self._set_status(self.engine.status_queue.get_nowait())
            except queue.Empty:
                break

        if latest is None:
            return

        if latest.input_window is not None and latest.output_window is not None:
            self._input_history = latest.input_window
            self._output_history = latest.output_window
        else:
            hop_size = latest.input_samples.size
            self._input_history[:-hop_size] = self._input_history[hop_size:]
            self._input_history[-hop_size:] = latest.input_samples
            self._output_history[:-hop_size] = self._output_history[hop_size:]
            self._output_history[-hop_size:] = latest.output_samples
        self._input_db = self._smooth_spectrum(
            self._input_db,
            spectrum_db(self._input_history, self.engine.fft_size),
        )
        self._output_db = self._smooth_spectrum(
            self._output_db,
            spectrum_db(self._output_history, self.engine.fft_size),
        )
        # El bin 0 (DC) no existe en un eje logaritmico: log10(0) seria NaN.
        self.output_curve.setData(self._frequencies[1:], self._output_db[1:])
        self._update_filter_curve()
        self.spectrogram.push(self._output_db)

        in_rms, in_peak = level_db(latest.input_samples)
        out_rms, out_peak = level_db(latest.output_samples)
        self._input_total_db = total_level_db(
            latest.input_samples,
            weighting=self.engine.weighting,
            sample_rate=self.engine.sample_rate,
        )
        self._has_input_total = True
        self._update_total_level_label()
        self.meters.set_input(in_rms, in_peak)
        self.meters.set_output(out_rms, out_peak, latest.would_clip)

        if self._recorder is not None:
            self._recorder.record_frame(
                input_db=self._input_db,
                output_db=self._output_db,
                input_rms_db=in_rms,
                input_peak_db=in_peak,
                output_rms_db=out_rms,
                output_peak_db=out_peak,
                clip=latest.would_clip,
            )

        self._update_input_display()

    def _smooth_spectrum(
        self, previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Promedio exponencial: ``spectrum_averaging`` pesa el valor previo."""
        if previous.shape != current.shape:
            return current.copy()
        return self.spectrum_averaging * previous + (1.0 - self.spectrum_averaging) * current

    def _update_filter_curve(self) -> None:
        if self._filter_revision != self.engine.revision:
            self._filter_revision = self.engine.revision
            self._update_response_curves()
        if self._filter_db.size == self._frequencies.size:
            self.filter_curve.setData(self._frequencies[1:], self._filter_db[1:])
            lower = min(DB_FLOOR, float(np.min(self._filter_db)) - 3.0)
            upper = max(6.0, float(np.max(self._filter_db)) + 3.0)
            self.filter_view.setYRange(lower, upper, padding=0)

    def _update_response_curves(self) -> None:
        """Dibuja |H| medida, |H^-1| aplicada y su producto en dB."""
        h_inverse = self.engine.equalizer.h_inverse
        self._filter_db = gain_db(h_inverse)
        inverse_db = self._filter_db

        if self.engine.bypass or self.engine.h_response is None:
            measured_db = np.zeros_like(inverse_db)
            inverse_db = np.zeros_like(inverse_db)
            product_db = np.zeros_like(inverse_db)
        else:
            measured_db = gain_db(self.engine.h_response)
            product_db = gain_db(self.engine.h_response * h_inverse)

        self._measured_db = measured_db
        self._inverse_db = inverse_db
        self._product_db = product_db

        frequencies = self._frequencies
        self.h_measured_curve.setData(frequencies[1:], measured_db[1:])
        self.h_inverse_curve.setData(frequencies[1:], inverse_db[1:])
        self.h_product_curve.setData(frequencies[1:], product_db[1:])

        values = np.concatenate((measured_db[1:], inverse_db[1:], product_db[1:]))
        if values.size:
            self.filter_plot.setYRange(
                min(-1.0, float(np.min(values)) - 3.0),
                max(1.0, float(np.max(values)) + 3.0),
                padding=0,
            )

    # --------------------------------------------------------------- sesion
    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            if self._recorder is not None:
                return
            recorder = SessionRecorder(self.engine)
            try:
                directory = recorder.start()
            except Exception as error:  # noqa: BLE001 - el panel no debe romperse
                self._set_status(f"No se pudo iniciar la grabación: {error}")
                self.controls.set_recording(False)
                return
            self._recorder = recorder
            self._set_status(f"Grabando sesión en {directory}")
            return
        self._stop_recording()

    def _stop_recording(self) -> None:
        recorder = self._recorder
        if recorder is None:
            return
        self._recorder = None
        try:
            directory = recorder.stop()
        except Exception as error:  # noqa: BLE001
            self._set_status(f"No se pudo cerrar la sesión: {error}")
            return
        if directory is not None:
            self._set_status(f"Sesión guardada en {directory}")

    def _on_export_requested(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Exportar captura",
            "captura.png",
            "Imagen PNG (*.png)",
        )
        if not path:
            return
        try:
            png_path, csv_path = self._export_capture(path)
        except Exception as error:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Exportación fallida", str(error))
            return
        self._set_status(f"Captura exportada: {png_path.name}, {csv_path.name}")

    def _export_capture(self, path: str | Path) -> tuple[Path, Path]:
        """Guarda un PNG de las gráficas y un CSV del instante actual."""
        png_path = Path(path)
        if png_path.suffix.lower() != ".png":
            png_path = png_path.with_suffix(".png")
        csv_path = png_path.with_suffix(".csv")
        ImageExporter(self.graphics.scene()).export(str(png_path))
        self._write_current_csv(csv_path)
        return png_path, csv_path

    def _write_current_csv(self, path: Path) -> None:
        bins = self._frequencies.size

        def column(values: np.ndarray, fallback: float) -> np.ndarray:
            if values.size == bins:
                return values
            return np.full(bins, fallback, dtype=np.float64)

        columns = {
            "input_db": column(self._input_db, DB_FLOOR),
            "output_db": column(self._output_db, DB_FLOOR),
            "h_measured_db": column(self._measured_db, 0.0),
            "h_inverse_db": column(self._inverse_db, 0.0),
            "h_product_db": column(self._product_db, 0.0),
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frequency_hz", *columns.keys()))
            for index, frequency in enumerate(self._frequencies):
                writer.writerow(
                    (float(frequency), *(float(values[index]) for values in columns.values()))
                )

    def _on_mouse_moved(self, position: QtCore.QPointF) -> None:
        text = self.readout.handle_mouse(position)
        if text:
            self.readout_label.setText(text)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    # -------------------------------------------------------------- cierre
    def closeEvent(self, event: Any) -> None:  # noqa: N802 (Qt)
        self._timer.stop()
        self._stop_recording()
        self.engine.close()
        super().closeEvent(event)
