import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from src.audio.engine import AudioEngine
from src.ui.window import MainWindow


class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-ventana")

    def setUp(self):
        self.window = MainWindow(AudioEngine(sample_rate=48_000.0, fft_size=512))

    def tearDown(self):
        self.window.close()

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