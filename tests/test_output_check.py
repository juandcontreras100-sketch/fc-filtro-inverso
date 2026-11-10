import unittest

import numpy as np

from probar_salida import build_tone


class OutputCheckTests(unittest.TestCase):
    def test_build_tone_has_expected_duration_and_frequency(self):
        tone = build_tone(48_000, 0.5, 1_000.0, 0.1)

        self.assertEqual(tone.shape, (24_000,))
        self.assertLessEqual(float(np.max(np.abs(tone))), 0.1 + 1e-6)
        spectrum = np.abs(np.fft.rfft(tone.astype(np.float64)))
        frequencies = np.fft.rfftfreq(tone.size, 1.0 / 48_000)
        self.assertAlmostEqual(
            float(frequencies[int(np.argmax(spectrum))]), 1_000.0, delta=2.0
        )

    def test_build_tone_starts_and_ends_near_zero(self):
        tone = build_tone(48_000, 0.2, 1_000.0, 0.1, fade_seconds=0.02)

        self.assertLess(abs(float(tone[0])), 1e-3)
        self.assertLess(abs(float(tone[-1])), 1e-3)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            build_tone(0, 1.0, 1_000.0, 0.1)
        with self.assertRaises(ValueError):
            build_tone(48_000, 0.0, 1_000.0, 0.1)
        with self.assertRaises(ValueError):
            build_tone(48_000, 1.0, 0.0, 0.1)
        with self.assertRaises(ValueError):
            build_tone(48_000, 1.0, 1_000.0, 1.1)
