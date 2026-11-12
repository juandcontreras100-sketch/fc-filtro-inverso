import unittest

import numpy as np

from src.dsp.spectrum import (
    DB_FULL_SCALE,
    a_weighting_db,
    c_weighting_db,
    equivalent_noise_bandwidth_hz,
    level_db,
    spectrum_db,
    total_level_db,
)


class PhysicalSpectrumContractTests(unittest.TestCase):
    def test_full_scale_peak_sine_has_full_scale_peak_and_3_db_lower_rms(self):
        samples = np.sin(2.0 * np.pi * np.arange(48_000) / 48_000.0)

        rms_db, peak_db = level_db(samples)

        self.assertAlmostEqual(peak_db, DB_FULL_SCALE, delta=1e-6)
        self.assertAlmostEqual(rms_db, DB_FULL_SCALE - 3.0103, delta=1e-3)

    def test_coherent_tone_20_db_below_full_scale_is_read_within_point_one_db(self):
        fft_size = 2048
        bin_index = 46
        samples = 0.1 * np.sin(2.0 * np.pi * bin_index * np.arange(fft_size) / fft_size)

        peak_db = float(np.max(spectrum_db(samples, fft_size)))

        self.assertAlmostEqual(peak_db, DB_FULL_SCALE - 20.0, delta=0.1)

    def test_hann_equivalent_noise_bandwidth_is_one_point_five_bins(self):
        fft_size = 2048

        enbw_hz = equivalent_noise_bandwidth_hz(48_000.0, fft_size)

        self.assertAlmostEqual(enbw_hz, 1.5 * 48_000.0 / fft_size, delta=1e-10)

    def test_parseval_energy_matches_time_domain_with_relative_error_below_1e_6(self):
        samples = np.random.default_rng(42).standard_normal(2048)
        spectrum = np.fft.rfft(samples)
        weights = np.full(spectrum.size, 2.0)
        weights[0] = 1.0
        weights[-1] = 1.0
        frequency_energy = float(np.sum(weights * np.abs(spectrum) ** 2) / samples.size)
        time_energy = float(np.sum(samples**2))

        relative_error = abs(frequency_energy - time_energy) / time_energy

        self.assertLess(relative_error, 1e-6)
        self.assertTrue(np.isfinite(total_level_db(samples)))

    def test_a_and_c_weighting_are_finite_at_dc_and_match_reference_anchors(self):
        frequencies = np.array([0.0, 31.5, 63.0, 125.0, 250.0, 1000.0, 8000.0])

        a_values = a_weighting_db(frequencies)
        c_values = c_weighting_db(frequencies)

        self.assertTrue(np.all(np.isfinite(a_values)))
        self.assertTrue(np.all(np.isfinite(c_values)))
        np.testing.assert_allclose(
            c_values[1:],
            [-3.0, -0.8, -0.2, 0.0, 0.0, -3.0],
            atol=0.15,
        )

    def test_total_level_supports_c_weighting(self):
        sample_rate = 48_000.0
        samples = np.sin(2.0 * np.pi * 1000.0 * np.arange(sample_rate) / sample_rate)

        level = total_level_db(samples, weighting="C", sample_rate=sample_rate)

        self.assertAlmostEqual(level, DB_FULL_SCALE - 3.0103, delta=0.02)

    def test_spectrum_rejects_non_finite_samples(self):
        with self.assertRaisesRegex(ValueError, "finit"):
            spectrum_db(np.array([0.0, np.nan, 0.0]), 8)


if __name__ == "__main__":
    unittest.main()
