import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from src.audio.engine import AudioEngine
from src.dsp.spectrum import DB_CEILING
from src.storage.profiles import UserProfile
from src.ui import theme, window as window_module
from src.ui.window import AXIS_WIDTH, MainWindow


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

    def test_the_three_plots_share_axis_widths_to_align_their_titles(self):
        self.window.show()
        self.window.resize(1_200, 800)
        self.app.processEvents()

        plots = (
            self.window.input_plot,
            self.window.output_plot,
            self.window.compare_plot,
        )
        for plot in plots:
            with self.subTest(plot=plot.titleLabel.text):
                self.assertEqual(plot.getAxis("left").width(), AXIS_WIDTH)
                self.assertEqual(plot.getAxis("right").width(), AXIS_WIDTH)
                self.assertTrue(plot.getAxis("right").isVisible())

        boxes = [plot.getViewBox().sceneBoundingRect() for plot in plots]
        self.assertEqual(len({round(box.width(), 1) for box in boxes}), 1)
        self.assertEqual(len({round(box.left(), 1) for box in boxes}), 1)

    def test_unused_right_axes_are_visible_but_empty(self):
        for plot in (self.window.input_plot, self.window.compare_plot):
            right = plot.getAxis("right")
            self.assertTrue(right.isVisible())
            self.assertFalse(right.style["showValues"])
            self.assertEqual(right.labelText, "")

    def test_group_boxes_leave_room_for_their_titles(self):
        qss = theme.stylesheet()

        self.assertIn("margin-top: 16px;", qss)
        self.assertIn("padding: 14px 6px 6px 6px;", qss)

    def test_filter_curve_uses_a_right_axis_in_db_gain(self):
        right_axis = self.window.output_plot.getAxis("right")

        self.assertTrue(right_axis.isVisible())
        self.assertIn("Ganancia", right_axis.labelText)
        self.assertIn("dB", right_axis.labelUnits)
        self.assertIsNot(self.window.filter_view, self.window.output_plot.getViewBox())

    def test_filter_curve_can_show_gain_above_the_signal_db_range(self):
        self.window.engine.equalizer.h_inverse = np.full(
            self.window.engine.bin_count, 10.0, dtype=np.complex128
        )
        self.window.engine.revision += 1

        self.window._update_filter_curve()
        _frequencies, values = self.window.filter_curve.getData()
        minimum, maximum = self.window.filter_view.viewRange()[1]

        self.assertGreater(float(np.max(values)), DB_CEILING)
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

    def test_the_third_plot_defaults_to_filter_response(self):
        response = self.window.response_plot

        self.assertTrue(response.ctrl.logXCheck.isChecked())
        self.assertIs(
            response.getViewBox().linkedView(response.getViewBox().XAxis),
            self.window.input_plot.getViewBox(),
        )
        self.assertEqual(response.getAxis("left").labelUnits, "dB")
        self.assertTrue(self.window.response_message.isVisible())
        self.assertFalse(self.window.compare_input_curve.isVisible())

    def test_comparison_mode_shows_input_and_output_in_db(self):
        combo = self.window.controls.third_plot_combo
        combo.setCurrentIndex(combo.findData("comparacion"))

        self.assertEqual(self.window.response_plot.getAxis("left").labelUnits, "dB")
        self.assertTrue(self.window.compare_input_curve.isVisible())
        self.assertTrue(self.window.compare_output_curve.isVisible())
        self.assertFalse(self.window.response_message.isVisible())
        self.assertEqual(
            self.window.response_plot.titleLabel.text,
            "Comparacion entrada/salida",
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