import os

# Las pruebas de interfaz no deben abrir una ventana real.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest

import numpy as np
import pyqtgraph as pg

from src.audio.engine import AudioEngine, SpectrumFrame
from src.dsp.spectrum import DB_CEILING, DB_FLOOR
from src.ui.window import MainWindow


class MainWindowPlotLimitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = pg.mkQApp("test-limites-graficas")

    def setUp(self):
        # Nyquist = 20 kHz, el mismo limite superior que muestran las graficas.
        self.engine = AudioEngine(sample_rate=40_000.0, fft_size=512)
        self.window = MainWindow(self.engine)

    def tearDown(self):
        self.window.close()

    def test_plots_do_not_pan_or_zoom_with_the_mouse(self):
        for plot in (self.window.input_plot, self.window.output_plot):
            with self.subTest(plot=plot.titleLabel.text):
                self.assertEqual(
                    plot.getViewBox().state["mouseEnabled"], [False, False]
                )

    def test_plots_show_the_20_hz_to_nyquist_range(self):
        for plot in (self.window.input_plot, self.window.output_plot):
            with self.subTest(plot=plot.titleLabel.text):
                x_range, y_range = plot.viewRange()
                # El eje X es logaritmico: viewRange devuelve log10(Hz).
                self.assertAlmostEqual(10 ** x_range[0], 20.0, places=6)
                self.assertAlmostEqual(
                    10 ** x_range[1], self.engine.nyquist, places=6
                )
                self.assertAlmostEqual(y_range[0], DB_FLOOR, places=6)
                self.assertAlmostEqual(y_range[1], DB_CEILING, places=6)

    def test_spectrum_curves_skip_the_dc_bin(self):
        hop_size = self.engine.hop_size
        fft_size = self.engine.fft_size
        self.engine.display_queue.put(
            SpectrumFrame(
                input_samples=np.full(hop_size, 0.1),
                output_samples=np.full(hop_size, 0.05),
                would_clip=False,
                input_window=np.full(fft_size, 0.1),
                output_window=np.full(fft_size, 0.05),
            )
        )

        self.window._refresh()

        expected_frequencies = self.window._frequencies[1:]
        for curve in (self.window.input_curve, self.window.output_curve):
            frequencies, _values = curve.getOriginalDataset()
            self.assertGreater(float(frequencies[0]), 0.0)
            np.testing.assert_allclose(frequencies, expected_frequencies)


if __name__ == "__main__":
    unittest.main()
