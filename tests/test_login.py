"""Pruebas del dialogo de inicio de sesion y de la seleccion de dispositivos."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyqtgraph as pg
from PySide6 import QtWidgets

from main import session_device_names
from src.storage.profiles import ProfileStore, UserProfile
from src.ui.login import LoginDialog

HOSTAPIS = [
    {"name": "MME", "default_input_device": 0, "default_output_device": 1},
    {
        "name": "Windows WASAPI",
        "default_input_device": 2,
        "default_output_device": 3,
    },
    {"name": "Windows WDM-KS", "default_input_device": -1, "default_output_device": -1},
]


def _dev(name, hostapi, n_in, n_out):
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": n_in,
        "max_output_channels": n_out,
    }


DEVICES = [
    _dev("Microphone (MME)", 0, 1, 0),
    _dev("Speakers (MME)", 0, 0, 1),
    _dev("Microphone (WASAPI)", 1, 1, 0),
    _dev("Speakers (WASAPI)", 1, 0, 1),
]


def _values(combo: QtWidgets.QComboBox) -> list[object]:
    return [combo.itemData(index) for index in range(combo.count())]


class LoginDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-login")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = ProfileStore(Path(self._tmp.name) / "perfiles.json")

    def _dialog(self, devices=None) -> LoginDialog:
        return LoginDialog(devices or DEVICES, HOSTAPIS, self.store)

    def test_the_default_options_are_first_in_both_combos(self):
        dialog = self._dialog()

        self.assertIsNone(dialog.mic_combo.itemData(0))
        self.assertIn("por defecto del sistema", dialog.mic_combo.itemText(0))
        self.assertIsNone(dialog.speaker_combo.itemData(0))
        self.assertIn("por defecto del sistema", dialog.speaker_combo.itemText(0))

    def test_the_last_user_and_devices_are_preselected_by_name(self):
        self.store.save(
            UserProfile(
                "Juan Perez",
                "1234567",
                input_device="Microphone (WASAPI)",
                output_device="Speakers (WASAPI)",
            )
        )

        dialog = self._dialog()

        self.assertEqual(dialog.name_edit.text(), "Juan Perez")
        self.assertEqual(dialog.document_edit.text(), "1234567")
        self.assertEqual(dialog.mic_combo.currentData(), 2)
        self.assertEqual(dialog.speaker_combo.currentData(), 3)

    def test_reordered_indices_are_resolved_again_by_name(self):
        self.store.save(
            UserProfile(
                "Ana",
                "7654321",
                input_device="Microphone (WASAPI)",
                output_device="Speakers (WASAPI)",
            )
        )
        reordered = [DEVICES[1], DEVICES[0], DEVICES[3], DEVICES[2]]

        dialog = self._dialog(reordered)

        self.assertEqual(dialog.mic_combo.currentData(), 3)
        self.assertEqual(dialog.speaker_combo.currentData(), 2)

    def test_changing_the_microphone_recalculates_the_speaker_api(self):
        dialog = self._dialog()
        dialog.mic_combo.setCurrentIndex(dialog.mic_combo.findData(0))

        self.assertEqual(_values(dialog.speaker_combo), [None, 1])

        dialog.mic_combo.setCurrentIndex(dialog.mic_combo.findData(2))

        self.assertEqual(_values(dialog.speaker_combo), [None, 3])

    def test_accepting_a_valid_user_saves_the_profile_and_builds_a_pair(self):
        dialog = self._dialog()
        dialog.name_edit.setText("Juan Perez")
        dialog.document_edit.setText("1234567")
        dialog.mic_combo.setCurrentIndex(dialog.mic_combo.findData(0))
        dialog.speaker_combo.setCurrentIndex(dialog.speaker_combo.findData(1))

        dialog.accept()

        self.assertEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        self.assertIsNotNone(dialog.profile)
        self.assertEqual(dialog.profile.input_device, "Microphone (MME)")
        self.assertEqual(dialog.profile.output_device, "Speakers (MME)")
        self.assertEqual(dialog.device, (0, 1))
        self.assertEqual(self.store.load(), dialog.profile)

    def test_both_defaults_produce_a_none_device(self):
        dialog = self._dialog()
        dialog.name_edit.setText("Ana")
        dialog.document_edit.setText("123")

        dialog.accept()

        self.assertIsNone(dialog.device)

    def test_an_invalid_document_does_not_accept_or_save(self):
        dialog = self._dialog()
        dialog.name_edit.setText("Juan")
        dialog.document_edit.setText("12 345")

        with mock.patch.object(QtWidgets.QMessageBox, "warning") as warning:
            dialog.accept()

        self.assertNotEqual(dialog.result(), QtWidgets.QDialog.DialogCode.Accepted)
        self.assertIsNone(dialog.profile)
        self.assertIsNone(self.store.load())
        warning.assert_called_once()

    def test_a_missing_name_does_not_accept_or_save(self):
        dialog = self._dialog()
        dialog.document_edit.setText("1234567")

        with mock.patch.object(QtWidgets.QMessageBox, "warning"):
            dialog.accept()

        self.assertIsNone(dialog.profile)
        self.assertIsNone(self.store.load())


class SessionDeviceNamesTests(unittest.TestCase):
    """Nombres que se guardan en la metadata de la grabacion."""

    def test_a_full_duplex_device_names_both_sides(self):
        names = session_device_names(3, DEVICES, HOSTAPIS)

        self.assertEqual(
            names,
            {"entrada": "Speakers (WASAPI)", "salida": "Speakers (WASAPI)"},
        )

    def test_a_pair_uses_each_device_name(self):
        names = session_device_names((2, 3), DEVICES, HOSTAPIS)

        self.assertEqual(
            names,
            {"entrada": "Microphone (WASAPI)", "salida": "Speakers (WASAPI)"},
        )

    def test_a_pair_with_a_default_side_resolves_inside_the_host_api(self):
        names = session_device_names((2, None), DEVICES, HOSTAPIS)

        self.assertEqual(
            names,
            {"entrada": "Microphone (WASAPI)", "salida": "Speakers (WASAPI)"},
        )

    def test_the_system_default_has_no_names(self):
        self.assertEqual(
            session_device_names(None, DEVICES, HOSTAPIS),
            {"entrada": None, "salida": None},
        )


if __name__ == "__main__":
    unittest.main()
