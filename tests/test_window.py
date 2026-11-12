import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from src.audio.engine import AudioEngine
from src.storage.profiles import UserProfile
from src.ui import theme
from src.ui.window import MainWindow


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-ventana")

    def setUp(self):
        self.window = MainWindow(AudioEngine(sample_rate=48_000.0, fft_size=512))

    def tearDown(self):
        self.window.close()

    def test_the_logged_in_user_and_devices_are_remembered_by_the_window(self):
        profile = UserProfile("Juan Perez", "1234567")
        window = MainWindow(
            AudioEngine(sample_rate=48_000.0, fft_size=512),
            user=profile,
            devices={"entrada": "Microfono", "salida": "Altavoz"},
        )
        self.addCleanup(window.close)

        self.assertIs(window.user, profile)
        self.assertEqual(
            window.devices, {"entrada": "Microfono", "salida": "Altavoz"}
        )
        self.assertIn("Juan Perez", window.windowTitle())

    def test_without_a_user_the_title_has_no_name(self):
        self.assertEqual(
            self.window.windowTitle(), "Ecualizador FFT en tiempo real"
        )

    def test_filter_curve_uses_a_right_axis_in_db_gain(self):
        right_axis = self.window.output_plot.getAxis("right")

        self.assertTrue(right_axis.isVisible())
        self.assertIn("Ganancia", right_axis.labelText)
        self.assertIn("dB", right_axis.labelUnits)
        self.assertIsNot(self.window.filter_view, self.window.output_plot.getViewBox())

    def test_filter_curve_can_show_gain_above_the_signal_dbfs_range(self):
        self.window.engine.equalizer.h_inverse = np.full(
            self.window.engine.bin_count, 10.0, dtype=np.complex128
        )
        self.window.engine.revision += 1

        self.window._update_filter_curve()
        _frequencies, values = self.window.filter_curve.getData()
        minimum, maximum = self.window.filter_view.viewRange()[1]

        self.assertGreater(float(np.max(values)), 6.0)
        self.assertLessEqual(minimum, float(np.min(values)))
        self.assertGreaterEqual(maximum, float(np.max(values)))

    def test_leaving_bypass_shows_a_feedback_and_level_warning(self):
        self.window.engine.bypass = True

        self.window._on_mode_changed(False)

        message = self.window.statusBar().currentMessage().lower()
        self.assertIn("realimentación", message)
        self.assertIn("nivel", message)

    def test_theme_provides_dark_stylesheet_and_monospace_font(self):
        qss = theme.stylesheet()
        font = theme.mono_font()

        self.assertIn(theme.BACKGROUND, qss)
        self.assertIn(theme.PANEL, qss)
        self.assertTrue(font.fixedPitch())
        self.assertTrue(self.window.styleSheet())

    def test_the_third_plot_compares_input_and_output_in_dbfs(self):
        compare = self.window.compare_plot

        self.assertTrue(compare.ctrl.logXCheck.isChecked())
        self.assertIs(
            compare.getViewBox().linkedView(compare.getViewBox().XAxis),
            self.window.input_plot.getViewBox(),
        )
        self.assertEqual(compare.getAxis("left").labelUnits, "dBFS")
        self.assertIsNotNone(compare.legend)
        self.assertEqual(
            [label.text for _sample, label in compare.legend.items],
            ["Entrada", "Salida"],
        )

    def test_readout_covers_all_three_frequency_plots_with_monospace_font(self):
        self.assertEqual(
            self.window.readout._plots,
            [
                self.window.input_plot,
                self.window.output_plot,
                self.window.compare_plot,
            ],
        )
        self.assertTrue(self.window.readout_label.font().fixedPitch())
        self.assertTrue(self.window.controls.gain_label.font().fixedPitch())
        self.assertTrue(self.window.controls.total_level_label.font().fixedPitch())

    def test_status_badge_reflects_bypass_uncalibrated_and_active_states(self):
        self.window.engine.bypass = True
        self.window._update_mode_status()
        self.assertEqual(self.window.controls.status_state, "bypass")
        self.assertEqual(self.window.controls.status_label.text(), "Bypass")

        self.window.engine.bypass = False
        self.window.engine.h_response = None
        self.window._update_mode_status()
        self.assertEqual(self.window.controls.status_state, "warning")
        self.assertEqual(self.window.controls.status_label.text(), "Sin calibracion")

        self.window.engine.h_response = np.ones(
            self.window.engine.bin_count, dtype=np.complex128
        )
        self.window._update_mode_status()
        self.assertEqual(self.window.controls.status_state, "active")
        self.assertEqual(self.window.controls.status_label.text(), "Ecualizado +12 dB")
        self.assertFalse(self.window.controls.status_label.wordWrap())

    def test_session_actions_stay_visible_outside_the_scroll(self):
        self.assertIsNot(
            self.window.controls.session_box.parent(),
            self.window.controls,
        )