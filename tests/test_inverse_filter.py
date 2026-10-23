import unittest

import numpy as np

from src.dsp.inverse_filter import (
    OverlapAddFFTEqualizer,
    epsilon_profile,
    estimate_h_inverse,
)


class InverseFilterTests(unittest.TestCase):
    def test_epsilon_profile_uniform_coherence_is_constant_fraction_of_peak(self):
        h = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=complex)
        eps = epsilon_profile(h, np.ones(h.size), epsilon_base=0.02)
        self.assertTrue(np.allclose(eps, 0.02 * np.max(np.abs(h) ** 2)))

    def test_epsilon_profile_increases_where_coherence_is_low(self):
        h = np.ones(5, dtype=complex)
        eps = epsilon_profile(h, np.array([1.0, 1.0, 0.1, 1.0, 1.0]), epsilon_base=0.02)
        self.assertGreater(eps[2], eps[0])

    def test_estimate_h_inverse_of_flat_response_is_approximately_unity(self):
        inverse = estimate_h_inverse(np.ones(9), np.ones(9), 0.02, 12.0)
        self.assertTrue(np.allclose(inverse, 1.0, atol=0.05))

    def test_estimate_h_inverse_clips_gain_at_deep_notch(self):
        inverse = estimate_h_inverse(
            np.array([1.0, 1.0, 0.01, 1.0, 1.0]), np.ones(5), 1e-4, 12.0
        )
        self.assertLessEqual(np.max(np.abs(inverse)), 10 ** (12.0 / 20.0) * 1.001)

    def test_zero_response_has_positive_epsilon_and_finite_zero_inverse(self):
        h = np.zeros(9, dtype=complex)
        eps = epsilon_profile(h, np.ones(h.size), epsilon_base=0.0)
        inverse = estimate_h_inverse(h, np.ones(h.size), epsilon_base=0.0, max_gain_db=12.0)

        self.assertTrue(np.all(eps > 0.0))
        self.assertTrue(np.all(np.isfinite(inverse)))
        self.assertTrue(np.array_equal(inverse, np.zeros_like(inverse)))

    def test_nyquist_bin_is_stable_when_it_is_a_notch(self):
        # RFFT de N=16: el último bin es exactamente Nyquist (fs/2).
        fft_size = 16
        h = np.ones(fft_size // 2 + 1, dtype=complex)
        h[-1] = 0.0
        inverse = estimate_h_inverse(h, np.ones(h.size), epsilon_base=1e-4, max_gain_db=6.0)

        self.assertTrue(np.isfinite(inverse[-1]))
        self.assertEqual(inverse[-1], 0.0)
        self.assertLessEqual(np.max(np.abs(inverse)), 10 ** (6.0 / 20.0))

    def test_overlap_add_identity_reconstructs_audio_without_block_discontinuities(self):
        fft_size = 64
        equalizer = OverlapAddFFTEqualizer(np.ones(fft_size // 2 + 1), fft_size)
        audio = np.sin(2 * np.pi * 0.07 * np.arange(8 * equalizer.hop_size))

        output = [equalizer.process_block(block) for block in audio.reshape(-1, equalizer.hop_size)]
        output.append(equalizer.flush())
        rendered = np.concatenate([block for block in output if block.size])[: audio.size]

        self.assertEqual(rendered.shape, audio.shape)
        self.assertTrue(np.allclose(rendered, audio, atol=1e-12))

    def test_overlap_add_rejects_wrong_block_size_and_spectrum_shape(self):
        with self.assertRaisesRegex(ValueError, "bins"):
            OverlapAddFFTEqualizer(np.ones(4), fft_size=16)

        equalizer = OverlapAddFFTEqualizer(np.ones(9), fft_size=16)
        with self.assertRaisesRegex(ValueError, "exactamente"):
            equalizer.process_block(np.zeros(7))
