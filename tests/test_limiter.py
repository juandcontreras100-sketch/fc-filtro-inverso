import unittest

import numpy as np

from src.dsp.limiter import soft_limit


class SoftLimitTests(unittest.TestCase):
    def test_signals_below_the_threshold_pass_unchanged(self):
        samples = np.linspace(-0.79, 0.79, 101)
        self.assertTrue(np.array_equal(soft_limit(samples, threshold=0.8, ceiling=0.98), samples))

    def test_output_never_exceeds_the_ceiling(self):
        samples = np.array([0.9, 1.0, 2.0, 10.0, 1e6, np.inf, -0.9, -2.0, -1e6, -np.inf])
        limited = soft_limit(samples, threshold=0.8, ceiling=0.98)
        self.assertLessEqual(float(np.max(np.abs(limited))), 0.98 + 1e-12)
        self.assertTrue(np.all(np.isfinite(limited)))

    def test_it_is_odd_symmetric(self):
        samples = np.linspace(0.0, 5.0, 501)
        self.assertTrue(
            np.allclose(soft_limit(-samples, 0.8, 0.98), -soft_limit(samples, 0.8, 0.98))
        )

    def test_it_is_monotonic_so_it_never_inverts_a_waveform(self):
        ramp = np.linspace(-5.0, 5.0, 2001)
        limited = soft_limit(ramp, 0.8, 0.98)
        self.assertTrue(np.all(np.diff(limited) >= 0.0))

    def test_it_is_continuous_and_smooth_at_the_threshold(self):
        step = 1e-6
        below = soft_limit(np.array([0.8 - step]), 0.8, 0.98)[0]
        above = soft_limit(np.array([0.8 + step]), 0.8, 0.98)[0]
        self.assertAlmostEqual(above - below, 2 * step, delta=1e-9)  # pendiente 1

    def test_the_knee_compresses_but_stays_below_the_ceiling(self):
        value = soft_limit(np.array([1.0]), 0.8, 0.98)[0]
        self.assertGreater(value, 0.8)
        self.assertLess(value, 0.98)
        self.assertLess(value, 1.0)  # comprime: no deja pasar el 1.0 original

    def test_an_empty_block_returns_an_empty_block(self):
        self.assertEqual(soft_limit(np.zeros(0)).size, 0)

    def test_invalid_parameters_are_rejected(self):
        samples = np.zeros(4)
        with self.assertRaises(ValueError):
            soft_limit(samples, threshold=0.0, ceiling=0.98)
        with self.assertRaises(ValueError):
            soft_limit(samples, threshold=0.9, ceiling=0.8)
        with self.assertRaises(ValueError):
            soft_limit(samples, threshold=0.5, ceiling=1.2)


if __name__ == "__main__":
    unittest.main()
