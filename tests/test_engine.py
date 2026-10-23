import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio.engine import AudioEngine


class AudioEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.bins = self.engine.bin_count

    def _write_measurement(self, bins: int, name: str = "medicion_H.npy") -> Path:
        folder = Path(tempfile.mkdtemp())
        path = folder / name
        np.save(path, np.ones(bins, dtype=np.complex128))
        return path

    def test_engine_starts_flat_and_in_bypass(self):
        self.assertTrue(self.engine.bypass)
        self.assertFalse(self.engine.is_calibrated)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_hop_size_is_half_the_fft_size(self):
        self.assertEqual(self.engine.hop_size, 256)
        self.assertEqual(self.engine.bin_count, 257)

    def test_loading_a_measurement_switches_to_the_ecualizado_filter(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.bypass = False
        self.engine.recompute_inverse()

        self.assertTrue(self.engine.is_calibrated)
        self.assertAlmostEqual(self.engine.calibration_path, path)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0, atol=0.05))

    def test_loading_a_measurement_with_the_wrong_length_is_rejected(self):
        path = self._write_measurement(self.bins + 10)
        with self.assertRaisesRegex(ValueError, "bins"):
            self.engine.load_calibration(path)
        self.assertFalse(self.engine.is_calibrated)

    def test_bypass_keeps_the_filter_flat_even_with_a_measurement(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.bypass = True
        self.engine.recompute_inverse()

        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_the_inverse_compensates_a_notch(self):
        folder = Path(tempfile.mkdtemp())
        path = folder / "medicion_H.npy"
        h = np.ones(self.bins, dtype=np.complex128)
        h[64] = 0.1  # notch profundo
        np.save(path, h)

        self.engine.load_calibration(path)
        self.engine.bypass = False
        self.engine.max_gain_db = 18.0
        self.engine.recompute_inverse()

        gain = abs(self.engine.equalizer.h_inverse[64])
        self.assertGreater(gain, 3.0)
        self.assertLessEqual(gain, 10.0 ** (18.0 / 20.0) + 1e-9)

    def test_clear_calibration_returns_to_a_flat_filter(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)
        self.engine.clear_calibration()

        self.assertFalse(self.engine.is_calibrated)
        self.assertIsNone(self.engine.calibration_path)
        self.assertTrue(np.allclose(self.engine.equalizer.h_inverse, 1.0))

    def test_changing_the_fft_size_discards_an_incompatible_measurement(self):
        path = self._write_measurement(self.bins)
        self.engine.load_calibration(path)

        with self.assertRaisesRegex(ValueError, "calibracion"):
            self.engine.configure(fft_size=1024)

        self.assertFalse(self.engine.is_calibrated)
        self.assertEqual(self.engine.fft_size, 1024)
        self.assertEqual(self.engine.hop_size, 512)

    def test_recompute_inverse_bumps_the_revision(self):
        before = self.engine.revision
        self.engine.recompute_inverse()
        self.assertGreater(self.engine.revision, before)
