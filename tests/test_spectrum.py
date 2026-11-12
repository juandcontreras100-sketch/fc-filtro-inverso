import unittest

import numpy as np

from src.audio.engine import AudioEngine
from src.dsp.spectrum import (
    DB_FLOOR,
    gain_db,
    level_db,
    spectrum_db,
    third_octave_bands,
)


class SpectrumTests(unittest.TestCase):
    def test_a_full_scale_sine_on_an_exact_bin_reads_zero_dbfs(self):
        fft_size = 1024
        sample_rate = 48_000.0
        tone_hz = 3_000.0  # 64 * 48000 / 1024: un bin exacto
        t = np.arange(fft_size) / sample_rate
        samples = np.sin(2.0 * np.pi * tone_hz * t)

        values = spectrum_db(samples, fft_size)

        peak = int(np.argmax(values))
        self.assertAlmostEqual(float(values[peak]), 0.0, delta=0.1)

    def test_dc_and_nyquist_are_not_doubled(self):
        fft_size = 1024
        dc = np.ones(fft_size)
        nyquist = np.empty(fft_size)
        nyquist[0::2] = 1.0
        nyquist[1::2] = -1.0

        dc_values = spectrum_db(dc, fft_size)
        nyquist_values = spectrum_db(nyquist, fft_size)

        self.assertAlmostEqual(float(dc_values[0]), 0.0, delta=0.1)
        self.assertAlmostEqual(float(nyquist_values[-1]), 0.0, delta=0.1)

    def test_spectrum_db_of_a_pure_tone_peaks_at_its_frequency(self):
        fft_size = 1024
        sample_rate = 48_000.0
        tone_hz = 3_000.0
        t = np.arange(fft_size) / sample_rate
        samples = np.sin(2.0 * np.pi * tone_hz * t)

        values = spectrum_db(samples, fft_size)
        frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        peak = int(np.argmax(values))
        self.assertAlmostEqual(frequencies[peak], tone_hz, delta=sample_rate / fft_size)
        self.assertGreater(values[peak], -10.0)

    def test_spectrum_db_never_goes_below_the_floor(self):
        values = spectrum_db(np.zeros(256), 512)
        self.assertEqual(values.size, 257)
        self.assertTrue(np.all(values >= DB_FLOOR))

    def test_spectrum_db_of_empty_block_is_the_floor(self):
        values = spectrum_db(np.zeros(0), 256)
        self.assertEqual(values.size, 129)
        self.assertTrue(np.all(values == DB_FLOOR))

    def test_level_db_reports_peak_and_rms(self):
        samples = np.array([0.5, -0.5, 0.5, -0.5])
        rms, peak = level_db(samples)

        self.assertAlmostEqual(peak, 20.0 * np.log10(0.5), places=6)
        self.assertAlmostEqual(rms, 20.0 * np.log10(0.5), places=6)

    def test_level_db_rms_is_below_peak_for_a_sparse_signal(self):
        samples = np.array([1.0, 0.0, 0.0, 0.0])
        rms, peak = level_db(samples)

        self.assertAlmostEqual(peak, 0.0, places=6)
        self.assertLess(rms, peak)

    def test_gain_db_of_unity_is_zero(self):
        self.assertAlmostEqual(float(gain_db(np.ones(4))[0]), 0.0, places=6)

    def test_gain_db_reflects_the_magnitude(self):
        values = gain_db(np.array([1.0, 2.0]))
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        self.assertAlmostEqual(float(values[1]), 20.0 * np.log10(2.0), places=6)


class ThirdOctaveBandTests(unittest.TestCase):
    """Agrupado por bandas de 1/3 de octava para la vista de barras."""

    @classmethod
    def setUpClass(cls):
        cls.fft_size = 4096
        cls.sample_rate = 48_000.0
        cls.frequencies = np.fft.rfftfreq(
            cls.fft_size, d=1.0 / cls.sample_rate
        )

    def test_a_pure_tone_lands_in_its_band(self):
        tone_hz = 1_000.0
        t = np.arange(self.fft_size) / self.sample_rate
        samples = 0.5 * np.sin(2.0 * np.pi * tone_hz * t)
        levels = spectrum_db(samples, self.fft_size)

        centers, band_levels = third_octave_bands(levels, self.frequencies)

        self.assertEqual(centers.size, band_levels.size)
        peak = int(np.argmax(band_levels))
        self.assertEqual(centers[peak], tone_hz)
        # Las bandas vecinas quedan muy por debajo del tono.
        for offset in (-1, 1):
            self.assertLess(
                band_levels[peak + offset], band_levels[peak] - 20.0
            )

    def test_band_levels_never_go_below_the_floor(self):
        values = spectrum_db(np.zeros(self.fft_size), self.fft_size)

        centers, band_levels = third_octave_bands(values, self.frequencies)

        self.assertGreater(centers.size, 0)
        np.testing.assert_allclose(band_levels, DB_FLOOR)
        self.assertTrue(np.all(band_levels >= DB_FLOOR))

    def test_bands_without_bins_are_omitted(self):
        # Resolucion de 93.75 Hz: las bandas de graves quedan sin bins.
        fft_size = 512
        frequencies = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        values = spectrum_db(np.zeros(fft_size), fft_size)

        centers, band_levels = third_octave_bands(values, frequencies)

        self.assertEqual(centers.size, band_levels.size)
        self.assertGreater(centers.size, 0)
        self.assertEqual(centers[0], 100.0)
        self.assertTrue(np.all(centers >= 100.0))
        self.assertTrue(np.all(centers < self.sample_rate / 2.0))


class SpectrumWindowTests(unittest.TestCase):
    def test_a_frame_carries_the_last_fft_size_contiguous_samples(self):
        fft_size = 16
        hop_size = fft_size // 2
        engine = AudioEngine(sample_rate=48_000.0, fft_size=fft_size)

        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        frame = None
        for index in range(12):
            block = np.full(hop_size, float(index + 1), dtype=np.float32)
            outdata = np.zeros((hop_size, 1), dtype=np.float32)
            engine._callback(block[:, np.newaxis], outdata, hop_size, None, None)
            while not engine.display_queue.empty():
                frame = engine.display_queue.get_nowait()
                inputs.extend(frame.input_samples)
                outputs.extend(frame.output_samples)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.input_window.size, fft_size)
        self.assertEqual(frame.output_window.size, fft_size)
        np.testing.assert_allclose(frame.input_window, np.asarray(inputs[-fft_size:]))
        np.testing.assert_allclose(frame.output_window, np.asarray(outputs[-fft_size:]))
