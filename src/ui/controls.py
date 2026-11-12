"""Panel de control: modo, calibracion, ganancia, dispositivo y formato."""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore, QtWidgets

from src.ui import theme

SAMPLE_RATES = (44_100.0, 48_000.0, 96_000.0)
FFT_SIZES = (512, 1024, 2048, 4096)
MAX_GAIN_DB = 30.0


class ControlPanel(QtWidgets.QWidget):
    """Agrupa los controles y emite senales; no toca el motor directamente."""

    modeChanged = QtCore.Signal(bool)  # True = bypass
    calibrationRequested = QtCore.Signal()
    clearCalibrationRequested = QtCore.Signal()
    gainChanged = QtCore.Signal(float)
    epsilonChanged = QtCore.Signal(float)
    deviceChanged = QtCore.Signal(object)
    formatChanged = QtCore.Signal(float, int)
    recordToggled = QtCore.Signal(bool)
    exportRequested = QtCore.Signal()
    splDisplayToggled = QtCore.Signal(bool)
    splOffsetChanged = QtCore.Signal(float)
    weightingChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = False
        self.status_state = "bypass"

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 4)
        root.setSpacing(5)

        root.addWidget(self._build_mode_group())
        root.addWidget(self._build_calibration_group())
        root.addWidget(self._build_filter_group())
        root.addWidget(self._build_spl_group())
        root.addWidget(self._build_stream_group())
        root.addWidget(self._build_session_group())
        root.addStretch(1)

        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

    # ------------------------------------------------------------- secciones
    def _group(self, title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setContentsMargins(6, 6, 6, 4)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        return box, form

    def _build_mode_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("1. Operacion")

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Bypass (sin correccion)", True)
        self.mode_combo.addItem("Ecualizado (con filtro inverso)", False)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Operacion", self.mode_combo)

        self.status_label = QtWidgets.QLabel("Sin calibracion cargada")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet(theme.status_badge_style("bypass"))
        form.addRow("Estado", self.status_label)
        return box

    def _build_calibration_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("2. Calibracion H(f)")

        self.load_button = QtWidgets.QPushButton("Cargar (.npy)...")
        self.load_button.setObjectName("primaryButton")
        self.load_button.setToolTip("Cargar respuesta en frecuencia medida H(f)")
        self.load_button.clicked.connect(self.calibrationRequested.emit)

        self.clear_button = QtWidgets.QPushButton("Descartar")
        self.clear_button.setToolTip("Descartar calibracion activa")
        self.clear_button.clicked.connect(self.clearCalibrationRequested.emit)
        self.clear_button.setEnabled(False)

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.setSpacing(6)
        buttons_row.addWidget(self.load_button, 3)
        buttons_row.addWidget(self.clear_button, 2)
        form.addRow(buttons_row)

        self.calibration_label = QtWidgets.QLabel("—")
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setFont(theme.mono_font(9))
        self.calibration_label.setStyleSheet(f"color: {theme.MUTED};")
        form.addRow("Archivo", self.calibration_label)
        return box

    def _build_filter_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("3. Filtro inverso")

        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, int(MAX_GAIN_DB))
        self.gain_slider.setValue(12)
        self.gain_slider.setToolTip("Ganancia maxima permitida en las muescas")
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        self.gain_label = QtWidgets.QLabel("+12.0 dB")
        self.gain_label.setFont(theme.mono_font(9, bold=True))
        self.gain_label.setMinimumWidth(68)
        self.gain_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.gain_label.setStyleSheet(
            f"background-color: {theme.SURFACE}; border: 1px solid {theme.PANEL_EDGE}; "
            f"border-radius: 4px; padding: 2px 6px; color: {theme.FILTER};"
        )

        gain_row = QtWidgets.QHBoxLayout()
        gain_row.setSpacing(6)
        gain_row.addWidget(self.gain_slider, 1)
        gain_row.addWidget(self.gain_label)
        form.addRow("Ganancia max", gain_row)

        self.epsilon_spin = QtWidgets.QDoubleSpinBox()
        self.epsilon_spin.setFont(theme.mono_font(9))
        self.epsilon_spin.setRange(0.0001, 0.5)
        self.epsilon_spin.setDecimals(4)
        self.epsilon_spin.setSingleStep(0.005)
        self.epsilon_spin.setValue(0.02)
        self.epsilon_spin.setToolTip(
            "Regularizacion de Tikhonov: valores altos suavizan el filtro"
        )
        self.epsilon_spin.valueChanged.connect(self._on_epsilon_changed)
        form.addRow("Estabilidad (e)", self.epsilon_spin)
        return box

    def _build_spl_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("4. Entrada SPL")

        self.spl_checkbox = QtWidgets.QCheckBox("Mostrar entrada en dB SPL")
        self.spl_checkbox.setToolTip("La relacion es: dB SPL = dBFS + offset")
        self.spl_checkbox.toggled.connect(self._on_spl_display_toggled)
        form.addRow(self.spl_checkbox)

        self.spl_offset_spin = QtWidgets.QDoubleSpinBox()
        self.spl_offset_spin.setFont(theme.mono_font(9))
        self.spl_offset_spin.setRange(0.0, 200.0)
        self.spl_offset_spin.setDecimals(1)
        self.spl_offset_spin.setSingleStep(0.1)
        self.spl_offset_spin.setValue(120.0)
        self.spl_offset_spin.setSuffix(" dB")
        self.spl_offset_spin.setToolTip(
            "Offset obtenido con un calibrador de 94 dB SPL o un sonometro"
        )
        self.spl_offset_spin.valueChanged.connect(self._on_spl_offset_changed)
        form.addRow("Offset", self.spl_offset_spin)

        self.weighting_combo = QtWidgets.QComboBox()
        self.weighting_combo.addItem("Z (sin ponderacion)", "Z")
        self.weighting_combo.addItem("A (respuesta auditiva)", "A")
        self.weighting_combo.setToolTip("Ponderacion del nivel total de entrada")
        self.weighting_combo.currentIndexChanged.connect(self._on_weighting_changed)
        form.addRow("Ponderacion", self.weighting_combo)

        self.total_level_label = QtWidgets.QLabel("—")
        self.total_level_label.setToolTip("Nivel RMS total del ultimo bloque de entrada")
        self.total_level_label.setFont(theme.mono_font(10, bold=True))
        self.total_level_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.total_level_label.setStyleSheet(
            f"background-color: {theme.SURFACE}; border: 1px solid {theme.PANEL_EDGE}; "
            f"border-radius: 4px; padding: 3px 8px; color: {theme.INPUT};"
        )
        form.addRow("Nivel total", self.total_level_label)
        return box

    def _build_stream_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("5. Dispositivo")

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        form.addRow("Interfaz", self.device_combo)

        self.rate_combo = QtWidgets.QComboBox()
        for rate in SAMPLE_RATES:
            self.rate_combo.addItem(f"{rate / 1000:.0f} kHz", rate)
        self.rate_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Muestreo", self.rate_combo)

        self.fft_combo = QtWidgets.QComboBox()
        for size in FFT_SIZES:
            self.fft_combo.addItem(f"{size}", size)
        self.fft_combo.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Tamano FFT", self.fft_combo)

        self.restart_note = QtWidgets.QLabel(
            "Cambiar el dispositivo o el formato reinicia el audio."
        )
        self.restart_note.setWordWrap(True)
        self.restart_note.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        form.addRow(self.restart_note)
        return box

    def _build_session_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("6. Sesion")

        self.record_button = QtWidgets.QPushButton("Grabar")
        self.record_button.setObjectName("recordButton")
        self.record_button.setCheckable(True)
        self.record_button.setToolTip(
            "Guarda metadata, niveles y espectros de la sesion en sesiones/"
        )
        self.record_button.toggled.connect(self._on_record_toggled)

        self.export_button = QtWidgets.QPushButton("Exportar captura")
        self.export_button.setToolTip(
            "Guarda un PNG de las graficas y un CSV del instante actual"
        )
        self.export_button.clicked.connect(self.exportRequested.emit)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self.record_button, 1)
        actions.addWidget(self.export_button, 1)
        form.addRow(actions)
        return box

    # ------------------------------------------------------------- estados
    def populate_devices(self, devices: list[tuple[Any, str]]) -> None:
        """Llena el combo de dispositivos: ``(valor, etiqueta)``."""
        self._building = True
        self.device_combo.clear()
        for value, label in devices:
            self.device_combo.addItem(label, value)
        self._building = False

    def set_device(self, value: Any) -> None:
        self._building = True
        # Comparacion en Python: findData() de Qt no es fiable con tuplas
        # (entrada, salida). Un valor ajeno a la lista (p. ej. el dado por la
        # linea de comandos) se anade para que el combo no muestre otra cosa.
        index = next(
            (i for i in range(self.device_combo.count())
             if self.device_combo.itemData(i) == value),
            -1,
        )
        if index < 0 and value is not None:
            self.device_combo.addItem(f"Personalizado: {value}", value)
            index = self.device_combo.count() - 1
        if index < 0:
            index = 0
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        self._building = False

    def set_format(self, sample_rate: float, fft_size: int) -> None:
        self._building = True
        rate_index = self.rate_combo.findData(sample_rate)
        if rate_index >= 0:
            self.rate_combo.setCurrentIndex(rate_index)
        fft_index = self.fft_combo.findData(fft_size)
        if fft_index >= 0:
            self.fft_combo.setCurrentIndex(fft_index)
        self._building = False

    def set_mode(self, bypass: bool) -> None:
        self._building = True
        index = self.mode_combo.findData(bool(bypass))
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        self._building = False

    def set_gain(self, db: float) -> None:
        self._building = True
        self.gain_slider.setValue(int(round(db)))
        self.gain_label.setText(f"{db:+.1f} dB")
        self._building = False

    def set_epsilon(self, value: float) -> None:
        self._building = True
        self.epsilon_spin.setValue(value)
        self._building = False

    def set_spl_display(self, active: bool) -> None:
        self._building = True
        try:
            self.spl_checkbox.setChecked(bool(active))
        finally:
            self._building = False

    def set_spl_offset(self, offset_db: float) -> None:
        self._building = True
        try:
            self.spl_offset_spin.setValue(float(offset_db))
        finally:
            self._building = False

    def set_weighting(self, weighting: str) -> None:
        self._building = True
        try:
            index = self.weighting_combo.findData(weighting)
            if index >= 0:
                self.weighting_combo.setCurrentIndex(index)
        finally:
            self._building = False

    def set_calibration(self, path: str | None) -> None:
        self.calibration_label.setText(path or "—")
        self.clear_button.setEnabled(path is not None)

    def set_status(self, text: str, state: str | None = None) -> None:
        if state is None:
            lowered = text.casefold()
            if "bypass" in lowered:
                state = "bypass"
            elif "sin calibracion" in lowered:
                state = "warning"
            elif "ecualizado" in lowered:
                state = "active"
            else:
                state = "bypass"
        self.status_state = state
        self.status_label.setText(text)
        self.status_label.setStyleSheet(theme.status_badge_style(state))

    def set_recording(self, active: bool) -> None:
        """Sincroniza el boton con el estado real sin emitir la senal."""
        self._building = True
        try:
            self.record_button.setChecked(bool(active))
            self.record_button.setText("Detener" if active else "Grabar")
        finally:
            self._building = False

    # -------------------------------------------------------------- handlers
    def _on_mode_changed(self, _index: int) -> None:
        if self._building:
            return
        self.modeChanged.emit(bool(self.mode_combo.currentData()))

    def _on_gain_changed(self, value: int) -> None:
        self.gain_label.setText(f"{value:+.1f} dB")
        if self._building:
            return
        self.gainChanged.emit(float(value))

    def _on_epsilon_changed(self, value: float) -> None:
        if self._building:
            return
        self.epsilonChanged.emit(float(value))

    def _on_device_changed(self, _index: int) -> None:
        if self._building:
            return
        self.deviceChanged.emit(self.device_combo.currentData())

    def _on_format_changed(self, _index: int) -> None:
        if self._building:
            return
        self.formatChanged.emit(
            float(self.rate_combo.currentData()),
            int(self.fft_combo.currentData()),
        )

    def _on_spl_display_toggled(self, active: bool) -> None:
        if self._building:
            return
        self.splDisplayToggled.emit(bool(active))

    def _on_spl_offset_changed(self, value: float) -> None:
        if self._building:
            return
        self.splOffsetChanged.emit(float(value))

    def _on_weighting_changed(self, _index: int) -> None:
        if self._building:
            return
        self.weightingChanged.emit(str(self.weighting_combo.currentData()))

    def _on_record_toggled(self, recording: bool) -> None:
        self.record_button.setText("Detener" if recording else "Grabar")
        if self._building:
            return
        self.recordToggled.emit(bool(recording))
