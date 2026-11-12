"""Dialogo de inicio de sesion y seleccion de la interfaz de audio.

Entrada y salida se abren en un unico ``sd.Stream``, asi que el altavoz solo
ofrece dispositivos de la misma API de audio que el microfono. El dispositivo
resultante para el motor es ``(indice_entrada, indice_salida)``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PySide6 import QtWidgets

from src.audio.devices import (
    build_input_options,
    build_output_options,
    device_name,
    device_pair,
    find_device_by_name,
)
from src.storage.profiles import ProfileStore, UserProfile
from src.ui import theme


class LoginDialog(QtWidgets.QDialog):
    """Pide usuario, documento, microfono y altavoz antes de abrir el motor."""

    def __init__(
        self,
        devices: Sequence[Mapping[str, Any]],
        hostapis: Sequence[Mapping[str, Any]],
        store: ProfileStore | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.devices = list(devices)
        self.hostapis = list(hostapis)
        self.store = store if store is not None else ProfileStore()
        self.profile: UserProfile | None = None
        self.device: tuple[int | None, int | None] | None = None
        self._building = False

        self.setWindowTitle("Iniciar sesion - Ecualizador FFT")
        self.setModal(True)
        theme.apply_theme(self)
        self._build_ui()
        self._populate_inputs()
        self._apply_initial_profile(self.store.load())

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QtWidgets.QLabel("Iniciar sesion")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)

        description = QtWidgets.QLabel(
            "Identifique la sesion y elija la interfaz que se abrira como "
            "entrada y salida de audio."
        )
        description.setWordWrap(True)
        description.setStyleSheet(f"color: {theme.MUTED};")
        root.addWidget(description)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 4, 0, 4)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Nombre y apellido")
        self.name_edit.setMaxLength(80)
        form.addRow("Nombre", self.name_edit)

        self.document_edit = QtWidgets.QLineEdit()
        self.document_edit.setPlaceholderText("Solo letras, numeros, punto y guion")
        self.document_edit.setMaxLength(40)
        form.addRow("Documento", self.document_edit)

        self.mic_combo = QtWidgets.QComboBox()
        self.mic_combo.currentIndexChanged.connect(self._on_microphone_changed)
        form.addRow("Microfono", self.mic_combo)

        self.speaker_combo = QtWidgets.QComboBox()
        form.addRow("Altavoz", self.speaker_combo)

        note = QtWidgets.QLabel(
            "El altavoz se limita a la API del microfono elegido para poder "
            "abrir un unico stream full-duplex."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {theme.MUTED}; font-size: 8pt;")
        form.addRow(note)
        root.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Entrar")
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setObjectName("primaryButton")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("Cancelar")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.resize(440, 260)

    def _populate_inputs(self) -> None:
        self._building = True
        try:
            self.mic_combo.clear()
            for value, label in build_input_options(self.devices, self.hostapis):
                self.mic_combo.addItem(label, value)
        finally:
            self._building = False
        self._populate_outputs()

    def _populate_outputs(self, preserve_name: str | None = None) -> None:
        input_index = self.mic_combo.currentData()
        if preserve_name is None:
            preserve_name = device_name(self.devices, self.speaker_combo.currentData())
        options = build_output_options(self.devices, self.hostapis, input_index)

        self._building = True
        try:
            self.speaker_combo.clear()
            for value, label in options:
                self.speaker_combo.addItem(label, value)
            self._select_by_name(self.speaker_combo, preserve_name)
        finally:
            self._building = False

    def _apply_initial_profile(self, profile: UserProfile | None) -> None:
        if profile is None:
            self._select_by_name(self.speaker_combo, None)
            return

        self.name_edit.setText(profile.name)
        self.document_edit.setText(profile.document)
        input_index = find_device_by_name(self.devices, profile.input_device)
        self._select_value(self.mic_combo, input_index)
        self._populate_outputs(preserve_name=profile.output_device)
        output_index = find_device_by_name(self.devices, profile.output_device)
        if output_index is not None:
            self._select_value(self.speaker_combo, output_index)

    # ------------------------------------------------------------- seleccion
    @staticmethod
    def _select_value(combo: QtWidgets.QComboBox, value: Any) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def _select_by_name(self, combo: QtWidgets.QComboBox, name: str | None) -> None:
        value = find_device_by_name(self.devices, name)
        self._select_value(combo, value)

    def _on_microphone_changed(self, _index: int) -> None:
        if self._building:
            return
        self._populate_outputs()

    # ---------------------------------------------------------------- aceptar
    def accept(self) -> None:  # noqa: N802 (Qt)
        try:
            base = UserProfile(
                self.name_edit.text(),
                self.document_edit.text(),
            )
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Datos incompletos", str(error))
            return

        input_index = self.mic_combo.currentData()
        output_index = self.speaker_combo.currentData()
        profile = UserProfile(
            base.name,
            base.document,
            input_device=device_name(self.devices, input_index),
            output_device=device_name(self.devices, output_index),
        )
        try:
            self.store.save(profile)
        except OSError as error:
            QtWidgets.QMessageBox.warning(
                self,
                "No se pudo guardar el perfil",
                f"La sesion continuara, pero no se recordaran los dispositivos:\n{error}",
            )

        self.profile = profile
        self.device = device_pair(
            input_index, output_index, self.devices, self.hostapis
        )
        super().accept()
