import unittest

import numpy as np

from analisis_numerico import analyze_measurement, compare_transfer_linearity
from generar_medicion_demo import generate_measurement


class NumericalAnalysisTests(unittest.TestCase):
    def test_analysis_reports_all_requested_numerical_metrics(self):
        fft_size = 2048
        sample_rate = 48_000.0
        h_response, coherence = generate_measurement(fft_size, sample_rate)

        result = analyze_measurement(
            h_response,
            coherence,
            sample_rate=sample_rate,
            epsilon_base=0.02,
            max_gain_db=12.0,
            iterations=2_000,
        )

        self.assertLess(result["gradient_max_error"], 1e-3)
        self.assertGreater(result["condition_max"], result["condition_min"])
        self.assertTrue(np.isfinite(result["group_delay_rms_ms"]))
        self.assertGreater(result["filter_energy"], 0.0)
        self.assertGreaterEqual(result["cutoff_frequency_hz"], 0.0)

    def test_transfer_linearity_comparison_is_zero_for_matching_responses(self):
        h_response, _ = generate_measurement(1024, 48_000.0)

        result = compare_transfer_linearity(h_response, h_response.copy())

        self.assertAlmostEqual(result["relative_error_max"], 0.0)
        self.assertAlmostEqual(result["magnitude_difference_db_max"], 0.0)

    def test_transfer_linearity_comparison_detects_amplitude_dependence(self):
        h_low, _ = generate_measurement(1024, 48_000.0)
        h_high = h_low.copy()
        h_high[100:300] *= 1.1

        result = compare_transfer_linearity(h_low, h_high)

        self.assertGreater(result["relative_error_max"], 0.05)
        self.assertGreater(result["magnitude_difference_db_max"], 0.5)