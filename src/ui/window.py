"""Ventana principal del ecualizador FFT en tiempo real.

La ventana es el unico lugar donde se dibuja. El motor de audio se encarga
del stream y del filtro; aqui solo se leen sus resultados y se le envian los
cambios que el usuario pide desde el panel de control.
"""

from __future__ import annotations

import csv
import queue
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6 import QtCore, QtWidgets

from src.audio.engine import AudioEngine, CalibrationDiscardedError
from src.config import FREQUENCY_MAX_HZ, FREQUENCY_MIN_HZ
from src.dsp.spectrum import (
    DB_CEILING,
    DB_FLOOR,
    DB_FULL_SCALE,
    db_to_spl,
    gain_db,
    level_db,
    spectrum_db,
    third_octave_bands,
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
BAND_BAR_WIDTH = 0.8  # ancho de las barras de 1/3 de octava, en indices de banda
# Ancho fijo de los ejes: alinea los titulos de las tres graficas.
AXIS_WIDTH = 66
# Rango inicial de las curvas de ganancia (ganancia unitaria = DB_FULL_SCALE).
GAIN_VIEW_MIN_DB = DB_FULL_SCALE - 60.0


class MainWindow(QtWidgets.QMainWindow):
    """Interfaz grafica completa del ecualizador."""

    def __init__(
        self,
        engine: AudioEngine,
        parent: QtWidgets.QWidget | None = None,
        *,
        user: Any = None,
        devices: Mapping[str, str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.user = user
        self.devices = dict(devices or {})
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
        # Vista del espectro: "linea" (curvas) o "barras" (1/3 de octava).
        self._spectrum_style = "linea"
        self._third_plot_mode = "respuesta"
        self._band_centers = np.zeros(0)

        title = "Ecualizador FFT en tiempo real"
        if user is not None:
            title = f"{title} - {user.name}"
        self.setWindowTitle(title)
        self.resize(1_360, 960)
        theme.apply_theme(self)

        self._build_ui()
        self._connect_signals()
        self._sync_from_engine()
        QtCore.QTimer.singleShot(0, self._apply_frequency_axis_limits)

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
        graphs_layout.setContentsMargins(2, 0, 0, 0)
        graphs_layout.setSpacing(2)

        self.readout_label = QtWidgets.QLabel(
            "Cursor: pase el raton sobre las graficas para inspeccionar frecuencia y nivel."
        )
        self.readout_label.setFont(theme.mono_font(9))
        self.readout_label.setStyleSheet(
            f"background-color: {theme.PANEL}; color: {theme.TEXT}; "
            f"border: 1px solid {theme.PANEL_EDGE}; border-radius: 5px; padding: 5px 10px;"
        )
        self._readout_hint = self.readout_label.text()
        self.calibration_banner = QtWidgets.QLabel("DATOS SINTÉTICOS")
        self.calibration_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.calibration_banner.setStyleSheet(
            f"background-color: #3a2a12; color: {theme.WARNING}; "
            "font-weight: 700; padding: 3px;"
        )
        self._update_calibration_banner()
        graphs_layout.addWidget(self.calibration_banner)
        self.response_legend = QtWidgets.QLabel()
        self.response_legend.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.response_legend.setTextFormat(QtCore.Qt.RichText)
        self.response_legend.setStyleSheet(
            f"background-color: {theme.PANEL}; padding: 2px 6px;"
        )
        self.response_legend.hide()
        graphs_layout.addWidget(self.response_legend)
        graphs_layout.addWidget(self.readout_label)

        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(theme.BACKGROUND)
        # Sin margen de 9 px ni separacion entre filas: las graficas quedan
        # pegadas al borde y el ViewBox no agrega relleno propio.
        self.graphics.ci.layout.setContentsMargins(2, 2, 2, 2)
        self.graphics.ci.layout.setSpacing(2)
        self.graphics.viewport().installEventFilter(self)
        graphs_layout.addWidget(self.graphics)

        self.input_plot = self.graphics.addPlot(title="Entrada: senal captada")
        theme.style_plot(self.input_plot, "Entrada: senal captada")
        self.input_plot.setLogMode(x=True, y=False)
        self.input_plot.setLabel("left", "Magnitud", units="dB")
        self.input_plot.setLabel("bottom", "Frecuencia", units="kHz")
        self.input_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self.input_plot.showGrid(x=True, y=True, alpha=0.22)
        self.input_curve = self.input_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.6),
            fillLevel=DB_FLOOR,
            brush=pg.mkBrush(*theme.INPUT_FILL),
            name="Entrada",
        )
        self.input_bars = pg.BarGraphItem(
            x=np.zeros(0),
            y1=np.zeros(0),
            width=BAND_BAR_WIDTH,
            brush=pg.mkBrush(theme.INPUT),
        )
        self.input_plot.addItem(self.input_bars)
        self.input_bars.hide()

        self.graphics.nextRow()

        self.output_plot = self.graphics.addPlot(title="Salida: senal ecualizada")
        theme.style_plot(self.output_plot, "Salida: senal ecualizada")
        self.output_plot.setLogMode(x=True, y=False)
        self.output_plot.setLabel("left", "Magnitud", units="dB")
        self.output_plot.setLabel("bottom", "Frecuencia", units="kHz")
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
        self.filter_view.setYRange(GAIN_VIEW_MIN_DB, DB_CEILING, padding=0)
        self.output_plot.getViewBox().sigResized.connect(
            self._update_filter_view_geometry
        )
        self.filter_curve = pg.PlotDataItem(
            pen=pg.mkPen(theme.FILTER, width=1.2, style=QtCore.Qt.DashLine)
        )
        self.filter_curve.setLogMode(True, False)
        self.filter_view.addItem(self.filter_curve)
        self.filter_band_curve = pg.PlotDataItem(
            pen=pg.mkPen(theme.FILTER, width=1.2, style=QtCore.Qt.DashLine)
        )
        self.filter_band_curve.hide()
        self.filter_view.addItem(self.filter_band_curve)
        self._update_filter_view_geometry()
        self.output_bars = pg.BarGraphItem(
            x=np.zeros(0),
            y1=np.zeros(0),
            width=BAND_BAR_WIDTH,
            brush=pg.mkBrush(theme.OUTPUT),
        )
        self.output_plot.addItem(self.output_bars)
        self.output_bars.hide()

        self.graphics.nextRow()
        self.response_plot = self.graphics.addPlot(title="Respuesta del filtro")
        theme.style_plot(self.response_plot, "Respuesta del filtro")
        self.response_plot.setLabel("left", "Magnitud", units="dB")
        self.response_plot.setLabel("bottom", "Frecuencia", units="kHz")
        self.response_plot.setLogMode(x=True, y=False)
        self.response_plot.setXLink(self.input_plot)
        self.response_plot.showGrid(x=True, y=True, alpha=0.22)
        self.response_plot.setYRange(GAIN_VIEW_MIN_DB, DB_CEILING, padding=0)
        self.compare_plot = self.response_plot

        self.response_zero_line = pg.InfiniteLine(
            pos=DB_FULL_SCALE, angle=0, pen=pg.mkPen(theme.MUTED, width=1)
        )
        self.response_plot.addItem(self.response_zero_line)
        self.response_message = pg.TextItem(
            "Sin calibración cargada", color=theme.MUTED, anchor=(0.5, 0.5)
        )
        self.response_plot.addItem(self.response_message)
        self.response_measured_curve = self.response_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.5)
        )
        self.response_inverse_curve = self.response_plot.plot(
            pen=pg.mkPen(theme.FILTER, width=1.5)
        )
        self.response_product_curve = self.response_plot.plot(
            pen=pg.mkPen(theme.OUTPUT, width=1.5)
        )
        self.compare_input_curve = self.response_plot.plot(
            pen=pg.mkPen(theme.INPUT, width=1.5)
        )
        self.compare_output_curve = self.response_plot.plot(
            pen=pg.mkPen(theme.OUTPUT, width=1.5)
        )
        for curve in (
            self.compare_input_curve,
            self.compare_output_curve,
            self.response_measured_curve,
            self.response_inverse_curve,
            self.response_product_curve,
        ):
            curve.hide()

        self.output_plot.setXLink(self.input_plot)
        for plot in (self.input_plot, self.output_plot, self.response_plot):
            plot.getAxis("left").setWidth(AXIS_WIDTH)
            plot.getAxis("right").setWidth(AXIS_WIDTH)
            plot.showAxis("right")
        # Entrada y comparacion no usan la escala derecha, pero reservan el
        # mismo ancho para que los titulos queden alineados con la salida.
        for plot in (self.input_plot, self.response_plot):
            right = plot.getAxis("right")
            right.setStyle(showValues=False)
            right.setTicks([])

        for view in (
            self.input_plot.getViewBox(),
            self.output_plot.getViewBox(),
            self.response_plot.getViewBox(),
            self.filter_view,
        ):
            view.setDefaultPadding(0.0)

        self.readout = CursorReadout(
            [self.input_plot, self.output_plot, self.compare_plot],
            log_x=True,
            frequency_scale=1_000.0,
            curves=[
                [{"name": "entrada", "color": theme.INPUT}],
                [
                    {"name": "salida", "color": theme.OUTPUT},
                    {
                        "name": "filtro",
                        "color": theme.FILTER,
                        "view": self.filter_view,
                    },
                ],
                [
                    {
                        "name": "medida",
                        "color": theme.INPUT,
                        "data": "h_measured_db",
                    },
                    {
                        "name": "inversa",
                        "color": theme.FILTER,
                        "data": "h_inverse_db",
                    },
                    {
                        "name": "producto",
                        "color": theme.OUTPUT,
                        "data": "h_product_db",
                    },
                    {
                        "name": "entrada",
                        "color": theme.INPUT,
                        "data": "entrada_sin_calibrar",
                    },
                    {"name": "salida", "color": theme.OUTPUT},
                ],
            ],
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
        self.controls.spectrumStyleChanged.connect(self._on_spectrum_style_changed)
        self.controls.thirdPlotChanged.connect(self._on_third_plot_changed)
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
        self.controls.set_third_plot_mode(self._third_plot_mode)
        self.controls.set_calibration(
            str(self.engine.calibration_path.name) if self.engine.calibration_path else None,
            synthetic=self.engine.calibration_synthetic,
        )
        self._update_calibration_banner()
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
        self._apply_frequency_axis_limits()
        if self._spectrum_style == "barras":
            self._update_input_display()
            self._apply_band_axis()
        self._update_filter_curve()

    def _apply_frequency_axis_limits(self) -> None:
        """Applies shared frequency bounds after linked plots finish laying out."""
        if self._spectrum_style == "barras":
            self._apply_band_axis()
            return
        lower, upper = self._frequency_axis_bounds()
        # Los datos del eje se expresan en kHz; el ViewBox log trabaja en log10.
        x_min, x_max = np.log10(lower / 1_000.0), np.log10(upper / 1_000.0)
        limits = {
            "xMin": np.log10(lower / 1_000.0),
            "xMax": np.log10(max(float(self.engine.nyquist), lower) / 1_000.0),
            "yMin": DB_FLOOR,
            "yMax": DB_CEILING,
        }
        for plot in (self.input_plot, self.output_plot, self.response_plot):
            plot.setMouseEnabled(x=False, y=False)
            plot.setLimits(**limits)
            plot.setXRange(x_min, x_max, padding=0)
        self._update_input_level_axis()
        self._position_response_message()

    def _frequency_axis_bounds(self) -> tuple[float, float]:
        """Extremos visibles del eje X comunes a todos los espectros."""
        first_bin = (
            float(self._frequencies[1]) if self._frequencies.size > 1 else 20.0
        )
        lower = max(FREQUENCY_MIN_HZ, first_bin)
        upper = min(FREQUENCY_MAX_HZ, float(self.engine.nyquist))
        return lower, max(upper, lower)

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
            lower, upper, unit = DB_FLOOR, DB_CEILING, "dB"
        else:
            lower = max(0.0, float(db_to_spl(DB_FLOOR, offset)))
            upper = float(db_to_spl(DB_CEILING, offset))
            unit = "dB SPL"
        self.input_plot.setLabel("left", "Magnitud", units=unit)
        self.input_plot.setLimits(yMin=lower, yMax=upper)
        self.input_plot.setYRange(lower, upper, padding=0)
        self.input_curve.setFillLevel(lower)

    def _update_input_display(self) -> None:
        if self._spectrum_style == "barras":
            self._update_band_display()
            return
        input_values = self._input_db
        input_unit = "dB"
        offset = self._active_spl_offset()
        if offset is not None:
            input_values = db_to_spl(input_values, offset)
            input_unit = "dB SPL"
        if input_values.size == self._frequencies.size:
            self.input_curve.setData(
                self._frequencies[1:] / 1_000.0, input_values[1:]
            )
        if self._output_db.size == self._frequencies.size:
            self.compare_output_curve.setData(
                self._frequencies[1:] / 1_000.0, self._output_db[1:]
            )
        if self._input_db.size == self._frequencies.size:
            self.compare_input_curve.setData(
                self._frequencies[1:] / 1_000.0, self._input_db[1:]
            )

        spectra = {
            "entrada": input_values,
            "salida": self._output_db,
            "filtro": self._filter_db,
        }
        units = {"entrada": input_unit, "salida": "dB", "filtro": "dB"}
        if offset is not None:
            # La comparacion siempre muestra la entrada sin calibrar.
            spectra["entrada_sin_calibrar"] = self._input_db
            units["entrada_sin_calibrar"] = "dB"
        if self.engine.is_calibrated:
            spectra.update(
                {
                    "h_measured_db": self._measured_db,
                    "h_inverse_db": self._inverse_db,
                    "h_product_db": self._product_db,
                }
            )
            units.update(
                {
                    "h_measured_db": "dB",
                    "h_inverse_db": "dB",
                    "h_product_db": "dB",
                }
            )
            frequency_range = self.engine.calibration_frequency_range_hz
            if frequency_range is not None:
                valid = (
                    (self._frequencies >= frequency_range[0])
                    & (self._frequencies <= frequency_range[1])
                )
                for key in ("h_measured_db", "h_inverse_db", "h_product_db"):
                    values = spectra[key].copy()
                    values[~valid] = np.nan
                    spectra[key] = values
        names = (
            {"entrada_sin_calibrar": "entrada (sin calibrar)"}
            if "entrada_sin_calibrar" in spectra
            else None
        )
        self.readout.set_spectra(
            self._frequencies, spectra, units=units, names=names
        )

    # ------------------------------------------------------ vista del espectro
    def _on_spectrum_style_changed(self, style: str) -> None:
        self._spectrum_style = style
        self._apply_spectrum_style()

    def _on_third_plot_changed(self, mode: str) -> None:
        self._third_plot_mode = mode
        self._apply_third_plot_mode()

    def _apply_spectrum_style(self) -> None:
        """Alterna la presentacion de entrada y salida entre linea y barras."""
        if self._spectrum_style == "barras":
            self.input_curve.setVisible(False)
            self.output_curve.setVisible(False)
            self.filter_curve.setVisible(False)
            self.compare_input_curve.setVisible(False)
            self.compare_output_curve.setVisible(False)
            self.input_bars.setVisible(True)
            self.output_bars.setVisible(True)
            self.filter_band_curve.setVisible(True)
            self.readout.hide()
            self.readout_label.setText(
                "Vista de barras 1/3 oct: el cursor solo lee valores en modo Linea."
            )
            # El eje X de la comparacion no puede seguir indices de banda.
            self.compare_plot.setXLink(None)
            self.filter_view.setLogMode("x", False)
            self.response_plot.setLogMode(x=False, y=False)
            for plot in (self.input_plot, self.output_plot):
                plot.setLogMode(x=False, y=False)
            self._update_filter_curve()
            self._update_input_display()
            self._apply_band_axis()
            return
        self.input_curve.setVisible(True)
        self.output_curve.setVisible(True)
        self.filter_curve.setVisible(True)
        self.compare_input_curve.setVisible(True)
        self.compare_output_curve.setVisible(True)
        self.input_bars.setVisible(False)
        self.output_bars.setVisible(False)
        self.filter_band_curve.setVisible(False)
        self.readout_label.setText(self._readout_hint)
        self.filter_view.setLogMode("x", True)
        self.response_plot.setLogMode(x=True, y=False)
        for plot in (self.input_plot, self.output_plot, self.response_plot):
            plot.setLogMode(x=True, y=False)
            plot.getAxis("bottom").setTicks(None)
        self.compare_plot.setXLink(self.input_plot)
        self._apply_frequency_axis_limits()
        self.output_plot.setXLink(self.input_plot)
        self._position_response_message()
        self._update_filter_curve()
        self._update_input_display()

    def _apply_band_axis(self) -> None:
        """Fija las tres graficas en indices compartidos de banda."""
        band_max = float(max(self._band_centers.size - 1, 1))
        for plot in (self.input_plot, self.output_plot, self.response_plot):
            plot.setMouseEnabled(x=False, y=False)
            plot.setLimits(
                xMin=-1.0, xMax=band_max + 1.0, yMin=DB_FLOOR, yMax=DB_CEILING
            )
            plot.setXRange(-0.5, band_max + 0.5, padding=0)
        self.response_plot.setLabel("bottom", "Banda de 1/3 de octava")
        self._update_input_level_axis()

    def _update_band_display(self) -> None:
        """Agrupa los espectros por bandas de 1/3 de octava y dibuja las barras."""
        if (
            self._frequencies.size == 0
            or self._input_db.size != self._frequencies.size
            or self._output_db.size != self._frequencies.size
            or self._filter_db.size != self._frequencies.size
        ):
            self._clear_band_display()
            return
        input_values = self._input_db
        offset = self._active_spl_offset()
        if offset is not None:
            input_values = db_to_spl(input_values, offset)
        centers, input_levels = third_octave_bands(input_values, self._frequencies)
        _centers, output_levels = third_octave_bands(
            self._output_db, self._frequencies
        )
        _centers, filter_levels = third_octave_bands(
            self._filter_db, self._frequencies
        )
        indices = np.arange(centers.size, dtype=np.float64)
        self.input_bars.setOpts(x=indices, y0=DB_FLOOR, y1=input_levels)
        self.output_bars.setOpts(x=indices, y0=DB_FLOOR, y1=output_levels)
        self.filter_band_curve.setData(indices, filter_levels)
        self.compare_input_curve.setData(indices, input_levels)
        self.compare_output_curve.setData(indices, output_levels)
        if filter_levels.size:
            lower = min(DB_FLOOR, float(np.min(filter_levels)) - 3.0)
            upper = max(DB_CEILING, float(np.max(filter_levels)) + 3.0)
            self.filter_view.setYRange(lower, upper, padding=0)
        if not np.array_equal(centers, self._band_centers):
            self._band_centers = centers
            ticks = [
                (float(index), f"{center:,.10g}".replace(",", "."))
                for index, center in enumerate(centers)
            ]
            for plot in (self.input_plot, self.output_plot):
                plot.getAxis("bottom").setTicks([ticks])
            # Las bandas cambian con Nyquist: reajusta el eje de indices.
            self._apply_band_axis()
        self._update_response_curves()

    def _clear_band_display(self) -> None:
        """Vacia las barras mientras el espectro no tiene el tamano actual."""
        empty = np.zeros(0)
        self._band_centers = empty
        self.input_bars.setOpts(x=empty, y1=empty)
        self.output_bars.setOpts(x=empty, y1=empty)
        self.filter_band_curve.setData(empty, empty)
        for plot in (self.input_plot, self.output_plot):
            plot.getAxis("bottom").setTicks([[]])

    def _update_total_level_label(self) -> None:
        if not self._has_input_total:
            self.controls.total_level_label.setText("—")
            self.controls.spl_context_label.setText("")
            self.controls.set_total_level_color()
            return
        value = self._input_total_db
        unit = "dB"
        context = ""
        color: str | None = None
        offset = self._active_spl_offset()
        if offset is not None:
            value = db_to_spl(value, offset)
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
        self._update_filter_curve()
        self._update_mode_status()
        if leaving_bypass:
            self.statusBar().showMessage(
                "Ecualización activa: vigile el nivel y la realimentación acústica.",
                8_000,
            )

    def _on_gain_changed(self, db: float) -> None:
        self.engine.max_gain_db = db
        self.engine.recompute_inverse()
        self._update_filter_curve()
        self._update_mode_status()

    def _on_epsilon_changed(self, value: float) -> None:
        self.engine.epsilon_base = value
        self.engine.recompute_inverse()
        self._update_filter_curve()
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
        coherence_stem = Path(path).stem.replace("_H", "_gamma2")
        if coherence_stem == Path(path).stem:
            coherence_stem = f"{Path(path).stem}_gamma2"
        candidate = Path(path).with_name(f"{coherence_stem}.npy")
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

        self.controls.set_calibration(
            Path(path).name, synthetic=self.engine.calibration_synthetic
        )
        self._update_calibration_banner()
        self._update_filter_curve()
        self._update_mode_status()
        self._set_status("Calibracion cargada")

    def _clear_calibration(self) -> None:
        self.engine.clear_calibration()
        self.controls.set_calibration(None)
        self.calibration_banner.hide()
        self._update_filter_curve()
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

    def _update_calibration_banner(self) -> None:
        if not self.engine.is_calibrated:
            self.calibration_banner.hide()
            return
        self.calibration_banner.setText(
            "DATOS SINTÉTICOS"
            if self.engine.calibration_synthetic
            else "CALIBRACIÓN NO SINTÉTICA"
        )
        self.calibration_banner.show()

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
        self.output_curve.setData(
            self._frequencies[1:] / 1_000.0, self._output_db[1:]
        )
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
        self.meters.set_input(in_rms, in_peak, latest.input_would_clip)
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
        if self._spectrum_style == "barras":
            # La linea del filtro por bandas se dibuja en _update_band_display.
            return
        if self._filter_db.size == self._frequencies.size:
            self.filter_curve.setData(
                self._frequencies[1:] / 1_000.0, self._filter_db[1:]
            )
            lower = min(DB_FLOOR, float(np.min(self._filter_db)) - 3.0)
            upper = max(DB_CEILING, float(np.max(self._filter_db)) + 3.0)
            self.filter_view.setYRange(lower, upper, padding=0)

    def _update_response_curves(self) -> None:
        """Dibuja curvas de calibracion solo si hay una medicion validada."""
        h_response = self.engine.h_response
        h_inverse = self.engine.equalizer.h_inverse
        self._filter_db = gain_db(h_inverse)
        if h_response is None:
            self._measured_db = np.zeros(0)
            self._inverse_db = np.zeros(0)
            self._product_db = np.zeros(0)
            for curve in (
                self.response_measured_curve,
                self.response_inverse_curve,
                self.response_product_curve,
            ):
                curve.setData([], [])
            self._apply_third_plot_mode()
            return

        self._measured_db = gain_db(h_response)
        self._inverse_db = gain_db(h_inverse)
        self._product_db = gain_db(h_response * h_inverse)

        valid = np.ones(self._frequencies.size, dtype=bool)
        frequency_range = self.engine.calibration_frequency_range_hz
        if frequency_range is not None:
            valid = (
                (self._frequencies >= frequency_range[0])
                & (self._frequencies <= frequency_range[1])
            )
        valid &= self._frequencies > 0.0
        if self._spectrum_style == "linea":
            x_values = self._frequencies[valid] / 1_000.0
            levels = (
                self._measured_db[valid],
                self._inverse_db[valid],
                self._product_db[valid],
            )
            self.response_plot.setLabel("bottom", "Frecuencia", units="kHz")
        else:
            measured_spectrum = self._measured_db.copy()
            inverse_spectrum = self._inverse_db.copy()
            product_spectrum = self._product_db.copy()
            if frequency_range is not None:
                valid = (
                    (self._frequencies >= frequency_range[0])
                    & (self._frequencies <= frequency_range[1])
                )
                measured_spectrum[~valid] = DB_FLOOR
                inverse_spectrum[~valid] = DB_FLOOR
                product_spectrum[~valid] = DB_FLOOR
            centers, measured = third_octave_bands(
                measured_spectrum, self._frequencies
            )
            _centers, inverse = third_octave_bands(
                inverse_spectrum, self._frequencies
            )
            _centers, product = third_octave_bands(
                product_spectrum, self._frequencies
            )
            if frequency_range is not None:
                valid_centers = (
                    (centers >= frequency_range[0])
                    & (centers <= frequency_range[1])
                )
                centers = centers[valid_centers]
                measured = measured[valid_centers]
                inverse = inverse[valid_centers]
                product = product[valid_centers]
            x_values = np.arange(centers.size, dtype=np.float64)
            levels = (measured, inverse, product)
            ticks = [
                (float(index), f"{center:g}")
                for index, center in enumerate(centers)
            ]
            self.response_plot.getAxis("bottom").setTicks([ticks])
            self.response_plot.setLabel("bottom", "Banda de 1/3 de octava")

        self.response_measured_curve.setData(x_values, levels[0])
        self.response_inverse_curve.setData(x_values, levels[1])
        self.response_product_curve.setData(x_values, levels[2])
        nonempty = [values for values in levels if values.size]
        if nonempty:
            all_values = np.concatenate(nonempty)
            self.response_plot.setYRange(
                min(GAIN_VIEW_MIN_DB, float(np.min(all_values)) - 3.0),
                max(DB_CEILING, float(np.max(all_values)) + 3.0),
                padding=0,
            )
        self._apply_third_plot_mode()

    def _position_response_message(self) -> None:
        x_range, y_range = self.response_plot.viewRange()
        self.response_message.setPos(
            (x_range[0] + x_range[1]) / 2.0,
            (y_range[0] + y_range[1]) / 2.0,
        )

    def _apply_third_plot_mode(self) -> None:
        """Shows filter response by default; comparison remains selectable."""
        response_mode = self._third_plot_mode == "respuesta"
        calibrated = self.engine.is_calibrated
        title = (
            "Respuesta del filtro"
            if response_mode
            else "Comparacion entrada/salida"
        )
        if response_mode and calibrated:
            provenance = (
                "DATOS SINTÉTICOS"
                if self.engine.calibration_synthetic
                else "CALIBRACIÓN NO SINTÉTICA"
            )
            title = f"{title} · {provenance}"
        self.response_plot.setTitle(title, color=theme.TEXT, size="10pt")
        self.response_plot.setLabel("left", "Magnitud", units="dB")
        self.response_zero_line.setVisible(response_mode)
        self.response_message.setVisible(response_mode and not calibrated)
        self.response_measured_curve.setVisible(response_mode and calibrated)
        self.response_inverse_curve.setVisible(response_mode and calibrated)
        self.response_product_curve.setVisible(response_mode and calibrated)
        self.compare_input_curve.setVisible(not response_mode)
        self.compare_output_curve.setVisible(not response_mode)
        self.readout.set_enabled(
            self.response_plot, {"entrada", "salida"}, not response_mode
        )
        self.readout.set_enabled(
            self.response_plot,
            {"medida", "inversa", "producto"},
            response_mode and calibrated,
        )

        if response_mode:
            self.response_legend.setVisible(calibrated)
            if calibrated:
                frequency_range = self.engine.calibration_frequency_range_hz
                range_label = ""
                if frequency_range is not None:
                    range_label = (
                        f" · rango {frequency_range[0]:g}–"
                        f"{frequency_range[1]:g} Hz"
                    )
                self.response_legend.setText(
                    f'<span style="color:{theme.INPUT}">|H| medida</span> &nbsp;'
                    f'<span style="color:{theme.FILTER}">|H⁻¹| inversa</span> &nbsp;'
                    f'<span style="color:{theme.OUTPUT}">|H·H⁻¹| producto</span>'
                    f"{range_label}"
                )
        else:
            self.response_legend.setText(
                f'<span style="color:{theme.INPUT}">Entrada</span> &nbsp;'
                f'<span style="color:{theme.OUTPUT}">Salida</span>'
            )
            self.response_legend.show()
            self.response_plot.setYRange(DB_FLOOR, DB_CEILING, padding=0)
        self._position_response_message()

    # --------------------------------------------------------------- sesion
    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            if self._recorder is not None:
                return
            recorder = SessionRecorder(
                self.engine,
                user=self.user,
                devices=self.devices,
            )
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

    def _default_export_path(self) -> Path:
        """Captura por defecto en la carpeta del usuario, si inicio sesion."""
        if self.user is None:
            return Path("captura.png")
        return Path("sesiones") / self.user.folder_name / "captura.png"

    def _on_export_requested(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Exportar captura",
            str(self._default_export_path()),
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
        png_path.parent.mkdir(parents=True, exist_ok=True)
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

        has_calibration = self.engine.is_calibrated
        columns = {
            "input_db": column(self._input_db, DB_FLOOR),
            "output_db": column(self._output_db, DB_FLOOR),
            "h_measured_db": (
                column(self._measured_db, DB_FULL_SCALE) if has_calibration else None
            ),
            "h_inverse_db": (
                column(self._inverse_db, DB_FULL_SCALE) if has_calibration else None
            ),
            "h_product_db": (
                column(self._product_db, DB_FULL_SCALE) if has_calibration else None
            ),
        }
        provenance = (
            "DATOS SINTÉTICOS"
            if self.engine.calibration_synthetic
            else             "MEDICIÓN NO SINTÉTICA"
            if has_calibration
            else "SIN CALIBRACIÓN"
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("frequency_hz", *columns.keys(), "provenance"))
            for index, frequency in enumerate(self._frequencies):
                writer.writerow(
                    (
                        float(frequency),
                        *(
                            "" if values is None else float(values[index])
                            for values in columns.values()
                        ),
                        provenance,
                    )
                )

    def _on_mouse_moved(self, position: QtCore.QPointF) -> None:
        if self._spectrum_style == "barras":
            # La lectura por bandas no esta disponible: ver _apply_spectrum_style.
            return
        text = self.readout.handle_mouse(position)
        if text:
            self.readout_label.setText(text)

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 (Qt)
        """Oculta el cursor cuando el raton sale del area de graficas."""
        graphics = getattr(self, "graphics", None)
        readout = getattr(self, "readout", None)
        if (
            graphics is not None
            and readout is not None
            and watched is graphics.viewport()
            and event.type() == QtCore.QEvent.Type.Leave
        ):
            readout.hide()
        return super().eventFilter(watched, event)

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    # -------------------------------------------------------------- cierre
    def closeEvent(self, event: Any) -> None:  # noqa: N802 (Qt)
        self._timer.stop()
        self._stop_recording()
        self.engine.close()
        super().closeEvent(event)
