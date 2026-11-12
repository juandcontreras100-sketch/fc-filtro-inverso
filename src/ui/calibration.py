"""Asistente de calibracion SPL a partir de bloques de entrada."""

from __future__ import annotations

import numpy as np
from PySide6 import QtWidgets

from src.dsp.spectrum import DB_FULL_SCALE, level_db
from src.ui import theme


class CalibrationDialog(QtWidgets.QDialog):
    """Mide una referencia estable y calcula el SPL de la plena escala digital."""

    def __init__(
        self,
        sample_rate: float,
        reference_spl: float = 94.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sample_rate = float(sample_rate)
        self.offset_db: float | None = None
        self._measuring = False
        self._measurement_valid = False
        self._blocks: list[np.ndarray] = []
        self._block_rms_db: list[float] = []
        self._sample_count = 0

        self.setWindowTitle("Calibracion SPL")
        self.resize(520, 420)
        theme.apply_theme(self)
        self._build_ui(reference_spl)

    def _build_ui(self, reference_spl: float) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        self.stacked_widget = QtWidgets.QStackedWidget()
        outer.addWidget(self.stacked_widget, 1)

        _page, layout, self.preparation_label = self._add_page(
            "Paso 1 de 3: prepara la referencia",
            "Coloca un calibrador de 94 dB SPL a 1 kHz sobre el microfono. "
            "Si usas un sonometro u otra referencia, indica su nivel SPL conocido.",
        )
        reference_row = QtWidgets.QHBoxLayout()
        reference_label = QtWidgets.QLabel("Nivel de referencia (dB SPL):")
        reference_label.setWordWrap(True)
        self.reference_spin = QtWidgets.QDoubleSpinBox()
        self.reference_spin.setRange(0.0, 200.0)
        self.reference_spin.setDecimals(1)
        self.reference_spin.setValue(reference_spl)
        reference_label.setBuddy(self.reference_spin)
        reference_row.addWidget(reference_label, 1)
        reference_row.addWidget(self.reference_spin)
        layout.addLayout(reference_row)
        layout.addStretch(1)
        self.next_button = QtWidgets.QPushButton("Continuar")
        self.next_button.setObjectName("primaryButton")
        self.next_button.clicked.connect(self._show_measurement_step)
        layout.addWidget(self.next_button)

        _page, layout, self.measurement_label = self._add_page(
            "Paso 2 de 3: mide la entrada",
            "Pulsa Iniciar medicion y manten la referencia estable. "
            "La medicion se detendra al recibir 1 segundo de muestras.",
        )
        layout.addStretch(1)
        measurement_buttons = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("Atras")
        self.back_button.clicked.connect(self._show_preparation_step)
        self.start_button = QtWidgets.QPushButton("Iniciar medicion")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_measurement)
        measurement_buttons.addWidget(self.back_button)
        measurement_buttons.addStretch(1)
        measurement_buttons.addWidget(self.start_button)
        layout.addLayout(measurement_buttons)

        _page, layout, result_instructions = self._add_page(
            "Paso 3 de 3: aplica el resultado",
            "Revisa el offset calculado y aplica la calibracion.",
        )
        self.result_label = QtWidgets.QLabel("Aun no hay un offset calculado.")
        self.result_label.setWordWrap(True)
        self.result_label.setFont(theme.mono_font(11, bold=True))
        layout.addWidget(self.result_label)
        layout.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Aplicar calibracion")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.accept)
        layout.addWidget(self.apply_button)

        self.status_label = QtWidgets.QLabel(
            "Prepara la referencia y pulsa Continuar."
        )
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        self.cancel_button = QtWidgets.QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        outer.addWidget(self.cancel_button)
        self.next_button.setFocus()

    def _add_page(
        self, title_text: str, instructions_text: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout, QtWidgets.QLabel]:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        title = QtWidgets.QLabel(title_text)
        title.setWordWrap(True)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        instructions = QtWidgets.QLabel(instructions_text)
        instructions.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(instructions)
        self.stacked_widget.addWidget(page)
        return page, layout, instructions

    def _show_preparation_step(self) -> None:
        self._measuring = False
        self.offset_db = None
        self._measurement_valid = False
        self.apply_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.stacked_widget.setCurrentIndex(0)
        self.status_label.setText("Prepara la referencia y pulsa Continuar.")
        self.next_button.setFocus()

    def _show_measurement_step(self) -> None:
        self._measuring = False
        self.stacked_widget.setCurrentIndex(1)
        self.status_label.setText(
            "Pulsa Iniciar medicion cuando la referencia este lista."
        )
        self.start_button.setFocus()

    def _start_measurement(self) -> None:
        self._blocks = []
        self._block_rms_db = []
        self._sample_count = 0
        self.offset_db = None
        self._measurement_valid = False
        self._measuring = True
        self.apply_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stacked_widget.setCurrentIndex(1)
        self.status_label.setText(
            "Midiendo: manten estable la referencia durante 1 segundo."
        )

    def add_block(self, samples: np.ndarray) -> None:
        """Acumula entrada solo durante una medicion iniciada desde la GUI."""
        if not self._measuring:
            return

        block = np.asarray(samples, dtype=np.float64).reshape(-1)
        if block.size == 0:
            return
        remaining = max(0, int(np.ceil(self.sample_rate)) - self._sample_count)
        block = block[:remaining].copy()
        if block.size == 0:
            return

        rms_db, _peak_db = level_db(block)
        self._blocks.append(block)
        self._block_rms_db.append(rms_db)
        self._sample_count += block.size
        self.status_label.setText(
            f"Midiendo: {self._sample_count / self.sample_rate:.2f} s recibidos."
        )
        if self._sample_count >= self.sample_rate:
            self._finish_measurement()

    def _finish_measurement(self) -> None:
        self._measuring = False
        self.start_button.setEnabled(True)
        samples = np.concatenate(self._blocks)
        total_rms_db, peak_db = level_db(samples)
        variation_db = max(self._block_rms_db) - min(self._block_rms_db)

        if total_rms_db <= DB_FULL_SCALE - 70.0:
            self._reject_measurement(
                "El nivel es demasiado bajo. Sube la ganancia o acerca la "
                "referencia y vuelve a medir."
            )
            return
        if peak_db >= DB_FULL_SCALE - 1.0:
            self._reject_measurement(
                "La senal tiene saturacion. Baja la ganancia para evitar la "
                "saturacion y vuelve a medir."
            )
            return
        if variation_db >= 1.0:
            self._reject_measurement(
                "El nivel vario demasiado. Manten estable la referencia "
                "durante toda la medicion y vuelve a medir."
            )
            return

        self.offset_db = (
            float(self.reference_spin.value()) - total_rms_db + DB_FULL_SCALE
        )
        self._measurement_valid = True
        self.result_label.setText(
            f"Offset resultante: {self.offset_db:.2f} dB\n"
            f"Nivel RMS medido: {total_rms_db:.2f} dB\n"
            f"Referencia: {self.reference_spin.value():.1f} dB SPL"
        )
        self.status_label.setText(
            "Medicion valida. Revisa el offset y aplica la calibracion."
        )
        self.apply_button.setEnabled(True)
        self.stacked_widget.setCurrentIndex(2)
        self.apply_button.setFocus()

    def _reject_measurement(self, message: str) -> None:
        self.offset_db = None
        self._measurement_valid = False
        self.apply_button.setEnabled(False)
        self.status_label.setText(message)
        self.start_button.setFocus()

    def accept(self) -> None:
        """Impide aplicar un offset hasta completar una medicion valida."""
        if not self._measurement_valid or self.offset_db is None:
            self.status_label.setText(
                "No hay una medicion valida. Mide 1 segundo de entrada antes "
                "de aplicar la calibracion."
            )
            return
        super().accept()
