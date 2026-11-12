import unittest

import numpy as np

from src.dsp.inverse_filter import estimate_h_inverse


class InverseFilterCoherenceTests(unittest.TestCase):
    def test_bands_below_coherence_threshold_are_not_corrected(self):
        h = np.array([1.0, 0.1, 0.1], dtype=np.complex128)
        coherence = np.array([1.0, 0.899, 0.9])

        inverse = estimate_h_inverse(
            h,
            coherence,
            epsilon_base=0.02,
            max_gain_db=12.0,
        )

        self.assertAlmostEqual(abs(inverse[0]), 1.0 / 1.02, places=6)
        self.assertEqual(inverse[1], 1.0)
        self.assertNotEqual(inverse[2], 0.0)

    def test_invalid_coherence_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coherence_threshold"):
            estimate_h_inverse(
                np.ones(3),
                np.ones(3),
                epsilon_base=0.02,
                max_gain_db=12.0,
                coherence_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()
