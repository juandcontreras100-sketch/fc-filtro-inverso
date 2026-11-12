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

    def test_plots_start_at_the_first_bin_when_it_is_above_20_hz(self):
        first_bin = self.engine.sample_rate / self.engine.fft_size
        self.assertGreater(first_bin, 20.0)

        for plot in (self.window.input_plot, self.window.output_plot):
            with self.subTest(plot=plot.titleLabel.text):
                x_range, y_range = plot.viewRange()
                # El eje X es logaritmico: viewRange devuelve log10(Hz).
                self.assertAlmostEqual(10 ** x_range[0], first_bin, places=6)
                self.assertAlmostEqual(
                    10 ** x_range[1], self.engine.nyquist, places=6
                )
                self.assertAlmostEqual(y_range[0], DB_FLOOR, places=6)
                self.assertAlmostEqual(y_range[1], DB_CEILING, places=6)

    def test_plots_keep_the_20_hz_minimum_when_the_first_bin_is_lower(self):
        engine = AudioEngine(sample_rate=192_000.0, fft_size=16_384)
        window = MainWindow(engine)
        self.addCleanup(window.close)
        self.assertLess(engine.sample_rate / engine.fft_size, 20.0)

        for plot in (window.input_plot, window.output_plot):
            with self.subTest(plot=plot.titleLabel.text):
                x_min = plot.viewRange()[0][0]
                self.assertAlmostEqual(10 ** x_min, 20.0, places=6)

    def test_graphs_have_no_padding_and_tight_margins(self):
        layout = self.window.graphics.ci.layout
        margins = layout.getContentsMargins()
        for value in margins:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 2.0)
        self.assertLessEqual(layout.verticalSpacing(), 2.0)

        for view in (
            self.window.input_plot.getViewBox(),
            self.window.output_plot.getViewBox(),
            self.window.filter_plot.getViewBox(),
            self.window.filter_view,
            self.window.spectrogram.plot.getViewBox(),
        ):
            self.assertEqual(float(view.state["defaultPadding"]), 0.0)

    def test_spectrogram_starts_at_the_first_bin(self):
        x_min, x_max = self.window.spectrogram.plot.viewRange()[0]

        self.assertAlmostEqual(x_min, self.engine.sample_rate / self.engine.fft_size)
        self.assertAlmostEqual(x_max, self.engine.nyquist)

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
