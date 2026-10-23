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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        root.addWidget(self._build_mode_group())
        root.addWidget(self._build_calibration_group())
        root.addWidget(self._build_filter_group())
        root.addWidget(self._build_stream_group())
        root.addStretch(1)

        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

    # ------------------------------------------------------------- secciones
    def _group(self, title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        box = QtWidgets.QGroupBox(title)
        box.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {theme.PANEL_EDGE}; border-radius: 6px; "
            f"margin-top: 10px; padding-top: 10px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {theme.MUTED}; }}"
        )
        form = QtWidgets.QFormLayout(box)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setSpacing(8)
        return box, form

    def _build_mode_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("Modo")

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Bypass (sin correccion)", True)
        self.mode_combo.addItem("Ecualizado (con filtro inverso)", False)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Operacion", self.mode_combo)

        self.status_label = QtWidgets.QLabel("Sin calibracion cargada")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.MUTED};")
        form.addRow("Estado", self.status_label)
        return box

    def _build_calibration_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("Calibracion")

        self.load_button = QtWidgets.QPushButton("Cargar medicion (.npy)...")
        self.load_button.clicked.connect(self.calibrationRequested.emit)
        form.addRow(self.load_button)

        self.clear_button = QtWidgets.QPushButton("Descartar calibracion")
        self.clear_button.clicked.connect(self.clearCalibrationRequested.emit)
        self.clear_button.setEnabled(False)
        form.addRow(self.clear_button)

        self.calibration_label = QtWidgets.QLabel("—")
        self.calibration_label.setWordWrap(True)
        self.calibration_label.setStyleSheet(f"color: {theme.MUTED};")
        form.addRow("Archivo", self.calibration_label)
        return box

    def _build_filter_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("Filtro inverso")

        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, int(MAX_GAIN_DB))
        self.gain_slider.setValue(12)
        self.gain_slider.setToolTip("Ganancia maxima permitida en las muescas")
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

        self.gain_label = QtWidgets.QLabel("+12.0 dB")
        self.gain_label.setMinimumWidth(58)
        self.gain_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.gain_slider, 1)
        gain_row.addWidget(self.gain_label)
        form.addRow("Ganancia max", gain_row)

        self.epsilon_spin = QtWidgets.QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0001, 0.5)
        self.epsilon_spin.setDecimals(4)
        self.epsilon_spin.setSingleStep(0.005)
        self.epsilon_spin.setValue(0.02)
        self.epsilon_spin.setToolTip("Regularizacion de Tikhonov: valores altos\u00a0suavizan el filtro")
        self.epsilon_spin.valueChanged.connect(self._on_epsilon_changed)
        form.addRow("Estabilidad (e)", self.epsilon_spin)
        return box

    def _build_stream_group(self) -> QtWidgets.QGroupBox:
        box, form = self._group("Dispositivo")

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

        self.restart_note = QtWidgets.QLabel("Cambiar el dispositivo o el formato reinicia el audio.")
        self.restart_note.setWordWrap(True)
        self.restart_note.setStyleSheet(f"color: {theme.MUTED};")
        form.addRow(self.restart_note)
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

    def set_calibration(self, path: str | None) -> None:
        self.calibration_label.setText(path or "—")
        self.clear_button.setEnabled(path is not None)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

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
