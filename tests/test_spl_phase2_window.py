import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from src.audio.engine import AudioEngine, SpectrumFrame
from src.ui import theme
from src.ui import window as window_module
from src.ui.window import MainWindow


class SplPhase2WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-ventana-spl-fase-2")

    def setUp(self):
        self.engine = AudioEngine(
            sample_rate=48_000.0,
            fft_size=512,
            device="Dispositivo de prueba",
        )
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window._calibration_dialog = None
        self.window.close()

    def _show_total(self, total_spl: float, offset_db: float = 94.0) -> None:
        self.engine.spl_offset_db = offset_db
        self.window.controls.set_spl_offset(offset_db)
        self.window.controls.set_spl_display(True)
        self.window._input_total_db = total_spl - offset_db
        self.window._has_input_total = True
        self.window._update_input_level_axis()
        self.window._update_input_display()
        self.window._update_total_level_label()

    def _frame(self, value: float) -> SpectrumFrame:
        hop_size = self.engine.hop_size
        fft_size = self.engine.fft_size
        return SpectrumFrame(
            input_samples=np.full(hop_size, value),
            output_samples=np.zeros(hop_size),
            input_window=np.full(fft_size, value),
            output_window=np.zeros(fft_size),
        )

    def test_calibration_button_emits_request_and_status_synchronizes_silently(self):
        requests: list[bool] = []
        offset_events: list[float] = []
        display_events: list[bool] = []
        self.window.controls.splCalibrationRequested.connect(lambda: requests.append(True))
        self.window.controls.splOffsetChanged.connect(offset_events.append)
        self.window.controls.splDisplayToggled.connect(display_events.append)

        # La ventana tambien abre el asistente modal al recibir esta senal;
        # se aisla porque esta prueba solo verifica el panel de control.
        with mock.patch.object(
            window_module, "CalibrationDialog", create=True
        ):
            self.window.controls.spl_calibration_button.click()
        self.window.controls.set_spl_calibration(94.0)

        self.assertEqual(requests, [True])
        self.assertEqual(self.window.controls.spl_calibration_button.text(), "Calibrar SPL...")
        self.assertTrue(self.window.controls.spl_status_label.isHidden())
        self.assertEqual(offset_events, [])
        self.assertEqual(display_events, [])

        self.window.controls.set_spl_calibration(None)
        self.assertFalse(self.window.controls.spl_status_label.isHidden())
        self.assertEqual(
            self.window.controls.spl_status_label.text(),
            "SIN CALIBRAR: valores en dBFS",
        )
        self.assertTrue(self.window.controls.spl_status_label.wordWrap())

    def test_uncalibrated_spl_toggle_never_adopts_120_or_presents_spl(self):
        self.window._input_total_db = -20.0
        self.window._has_input_total = True

        self.window.controls.spl_checkbox.setChecked(True)

        self.assertIsNone(self.engine.spl_offset_db)
        self.assertEqual(self.window.controls.spl_offset_spin.value(), 120.0)
        self.assertFalse(self.window.controls.spl_status_label.isHidden())
        self.assertEqual(
            self.window.controls.spl_status_label.text(),
            "SIN CALIBRAR: valores en dBFS",
        )
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dBFS")
        self.assertEqual(self.window.readout._units["entrada"], "dBFS")
        self.assertEqual(self.window.controls.total_level_label.text(), "-20.0 dBFS (Z)")
        self.assertEqual(self.window.controls.spl_context_label.text(), "")

    def test_real_offset_change_clears_uncalibrated_status_and_updates_spl(self):
        self.window._input_total_db = -20.0
        self.window._has_input_total = True
        self.window.controls.spl_checkbox.setChecked(True)

        self.window.controls.spl_offset_spin.setValue(121.5)

        self.assertEqual(self.engine.spl_offset_db, 121.5)
        self.assertTrue(self.window.controls.spl_status_label.isHidden())
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dB SPL")
        self.assertEqual(self.window.readout._units["entrada"], "dB SPL")
        self.assertEqual(self.window.controls.total_level_label.text(), "101.5 dB SPL (Z)")

    def test_a_weighted_spl_total_shows_context_and_safe_warning_danger_colors(self):
        self.window.controls.set_weighting("A")
        self.engine.weighting = "A"
        expected_colors = {
            60.0: theme.SPL_SAFE,
            75.0: theme.SPL_WARNING,
            90.0: theme.SPL_DANGER,
        }

        for level_spl, expected_color in expected_colors.items():
            with self.subTest(level_spl=level_spl):
                self._show_total(level_spl)

                label = self.window.controls.total_level_label
                self.assertEqual(label.text(), f"{level_spl:.1f} dB SPL (A)")
                self.assertEqual(
                    self.window.controls.spl_context_label.text(),
                    window_module.spl_context_text(level_spl),
                )
                self.assertEqual(window_module.spl_zone_color(level_spl), expected_color)
                self.assertIn(expected_color, label.styleSheet())

    def test_z_restores_normal_color_and_disabling_spl_clears_context(self):
        self.window.controls.set_weighting("A")
        self.engine.weighting = "A"
        self._show_total(90.0)

        self.window.controls.set_weighting("Z")
        self.engine.weighting = "Z"
        self.window._update_total_level_label()

        self.assertIn(theme.INPUT, self.window.controls.total_level_label.styleSheet())
        self.assertEqual(
            self.window.controls.spl_context_label.text(),
            window_module.spl_context_text(90.0),
        )

        self.window.controls.set_spl_display(False)
        self.window._update_total_level_label()

        self.assertIn(theme.INPUT, self.window.controls.total_level_label.styleSheet())
        self.assertEqual(self.window.controls.spl_context_label.text(), "")
        self.assertEqual(self.window.controls.total_level_label.text(), "-4.0 dBFS (Z)")

    def test_refresh_feeds_every_queued_frame_to_the_open_dialog(self):
        dialog = mock.MagicMock()
        self.window._calibration_dialog = dialog
        frames = [self._frame(0.1), self._frame(0.2)]
        for frame in frames:
            self.engine.display_queue.put(frame)

        self.window._refresh()

        self.assertEqual(dialog.add_block.call_count, 2)
        for call, frame in zip(dialog.add_block.call_args_list, frames):
            self.assertIs(call.args[0], frame.input_samples)

    def test_accepted_calibration_applies_offset_and_saves_the_input_device(self):
        dialog = mock.MagicMock()
        dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        dialog.offset_db = 117.5
        offset_events: list[float] = []
        display_events: list[bool] = []
        self.window.controls.splOffsetChanged.connect(offset_events.append)
        self.window.controls.splDisplayToggled.connect(display_events.append)

        with (
            mock.patch.object(
                window_module,
                "CalibrationDialog",
                return_value=dialog,
                create=True,
            ) as dialog_type,
            mock.patch.object(
                window_module,
                "input_device_name",
                return_value="Entrada simulada",
                create=True,
            ) as device_name,
            mock.patch.object(
                window_module,
                "save_spl_calibration",
                create=True,
            ) as save_calibration,
        ):
            self.window.controls.spl_calibration_button.click()

        dialog_type.assert_called_once_with(48_000.0, parent=self.window)
        dialog.exec.assert_called_once_with()
        device_name.assert_called_once_with("Dispositivo de prueba")
        save_calibration.assert_called_once_with(117.5, "Entrada simulada")
        self.assertEqual(self.engine.spl_offset_db, 117.5)
        self.assertEqual(self.window.controls.spl_offset_spin.value(), 117.5)
        self.assertTrue(self.window.controls.spl_checkbox.isChecked())
        self.assertTrue(self.window.controls.spl_status_label.isHidden())
        self.assertEqual(self.window.input_plot.getAxis("left").labelUnits, "dB SPL")
        self.assertEqual(offset_events, [])
        self.assertEqual(display_events, [])
        self.assertIsNone(self.window._calibration_dialog)

    def test_save_failure_keeps_the_accepted_calibration_applied(self):
        dialog = mock.MagicMock()
        dialog.exec.return_value = QtWidgets.QDialog.DialogCode.Accepted
        dialog.offset_db = 116.0

        with (
            mock.patch.object(
                window_module,
                "CalibrationDialog",
                return_value=dialog,
                create=True,
            ),
            mock.patch.object(
                window_module,
                "input_device_name",
                return_value="Entrada simulada",
                create=True,
            ),
            mock.patch.object(
                window_module,
                "save_spl_calibration",
                side_effect=OSError("disco lleno"),
                create=True,
            ),
        ):
            self.window._on_spl_calibration_requested()

        self.assertEqual(self.engine.spl_offset_db, 116.0)
        self.assertTrue(self.window.controls.spl_checkbox.isChecked())
        self.assertIn("no se pudo guardar", self.window.statusBar().currentMessage())
        self.assertIn("disco lleno", self.window.statusBar().currentMessage())


if __name__ == "__main__":
    unittest.main()
