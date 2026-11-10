import unittest

import numpy as np

from generar_medicion_demo import generate_measurement


class DemoMeasurementTests(unittest.TestCase):
    def test_measurement_uses_requested_fft_size_and_sample_rate(self):
        fft_size = 4096
        h_response, coherence = generate_measurement(fft_size, 96_000.0)

        self.assertEqual(h_response.shape, (fft_size // 2 + 1,))
        self.assertEqual(coherence.shape, h_response.shape)
        self.assertEqual(h_response.dtype, np.complex128)
        self.assertEqual(coherence.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(h_response)))
        self.assertTrue(np.all((coherence >= 0.0) & (coherence <= 1.0)))

    def test_notches_are_generated_at_the_expected_frequencies(self):
        fft_size = 4096
        sample_rate = 48_000.0
        h_response, _ = generate_measurement(fft_size, sample_rate)
        frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

        self.assertLess(abs(h_response[np.argmin(abs(frequencies - 1_000.0))]), 0.2)
        self.assertLess(abs(h_response[np.argmin(abs(frequencies - 4_000.0))]), 0.4)

    def test_rejects_invalid_fft_or_sample_rate(self):
        with self.assertRaisesRegex(ValueError, "fft_size"):
            generate_measurement(1023, 48_000.0)
        with self.assertRaisesRegex(ValueError, "sample_rate"):
            generate_measurement(1024, 0.0)