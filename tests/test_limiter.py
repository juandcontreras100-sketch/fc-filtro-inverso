import unittest

import numpy as np

from src.dsp.limiter import soft_limit


class SoftLimiterTests(unittest.TestCase):
    def test_samples_below_threshold_pass_unchanged(self):
        samples = np.linspace(-0.79, 0.79, 101)
        self.assertTrue(np.array_equal(soft_limit(samples), samples))

    def test_extreme_finite_samples_stay_within_the_ceiling(self):
        samples = np.array([-1e6, -2.0, -0.9, 0.9, 2.0, 1e6])
        limited = soft_limit(samples)

        self.assertTrue(np.all(np.isfinite(limited)))
        self.assertLessEqual(float(np.max(np.abs(limited))), 0.98)
        self.assertTrue(np.all(np.sign(limited) == np.sign(samples)))

    def test_non_finite_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            soft_limit(np.array([0.0, np.nan]))

    def test_it_is_odd_symmetric_and_monotonic(self):
        samples = np.linspace(0.0, 5.0, 501)
        limited = soft_limit(samples)
        self.assertTrue(np.allclose(soft_limit(-samples), -limited))
        self.assertTrue(np.all(np.diff(soft_limit(np.linspace(-5, 5, 2001))) >= 0.0))

    def test_the_transition_is_smooth_and_compresses_peaks(self):
        threshold, step = 0.8, 1e-6
        below = soft_limit(np.array([threshold - step]))[0]
        above = soft_limit(np.array([threshold + step]))[0]
        self.assertAlmostEqual(above - below, 2 * step, delta=1e-9)

        peak = soft_limit(np.array([1.0]))[0]
        self.assertGreater(peak, threshold)
        self.assertLess(peak, 1.0)

    def test_knee_compatibility_sets_the_transition_from_the_ceiling(self):
        samples = np.array([-0.8, -0.2, 0.2, 0.8])
        self.assertTrue(
            np.array_equal(soft_limit(samples, ceiling=0.98, knee=0.1), samples)
        )

    def test_invalid_parameters_and_empty_input(self):
        self.assertEqual(soft_limit(np.zeros(0)).size, 0)
        for kwargs in (
            {"threshold": 0.0},
            {"threshold": 0.9, "ceiling": 0.8},
            {"ceiling": 1.2},
            {"knee": 0.0},
            {"threshold": 0.7, "knee": 0.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                soft_limit(np.zeros(4), **kwargs)


if __name__ == "__main__":
    unittest.main()
