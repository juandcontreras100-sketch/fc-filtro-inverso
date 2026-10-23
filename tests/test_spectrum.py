import unittest

import numpy as np

from src.dsp.spectrum import DB_FLOOR, gain_db, level_db, spectrum_db


class SpectrumTests(unittest.TestCase):
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
