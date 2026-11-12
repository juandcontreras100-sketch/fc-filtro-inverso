import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.audio.engine import AudioEngine


class CalibrationContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=512)
        self.directory = Path(tempfile.mkdtemp())
        self.h_path = self.directory / "measured_H.npy"
        self.coherence_path = self.directory / "measured_gamma2.npy"
        self.metadata_path = self.h_path.with_suffix(".json")

    def _write_calibration(
        self,
        *,
        sample_rate_hz=48_000.0,
        fft_size_samples=512,
        synthetic=False,
        response=None,
        coherence=None,
        metadata=True,
    ):
        bins = fft_size_samples // 2 + 1
        h_values = np.ones(bins, dtype=np.complex128) if response is None else response
        coherence_values = (
            np.ones(bins, dtype=np.float64) if coherence is None else coherence
        )
        np.save(self.h_path, h_values)
        np.save(self.coherence_path, coherence_values)
        if metadata:
            self.metadata_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "sample_rate_hz": sample_rate_hz,
                        "fft_size_samples": fft_size_samples,
                        "frequency_range_hz": [20.0, min(20_000.0, sample_rate_hz / 2)],
                        "synthetic": synthetic,
                    }
                ),
                encoding="utf-8",
            )

    def test_loading_requires_metadata_to_verify_sample_rate_and_fft_size(self):
        self._write_calibration(metadata=False)

        with self.assertRaisesRegex(ValueError, "metadatos"):
            self.engine.load_calibration(self.h_path, self.coherence_path)

        self.assertFalse(self.engine.is_calibrated)

    def test_loading_rejects_incompatible_format_without_changing_engine(self):
        self._write_calibration(sample_rate_hz=44_100.0)
        old_filter = self.engine.equalizer.h_inverse.copy()

        with self.assertRaisesRegex(ValueError, "sample_rate_hz"):
            self.engine.load_calibration(self.h_path, self.coherence_path)

        self.assertFalse(self.engine.is_calibrated)
        np.testing.assert_array_equal(self.engine.equalizer.h_inverse, old_filter)

    def test_loading_rejects_non_complex_or_non_finite_measurements(self):
        self._write_calibration(response=np.ones(self.engine.bin_count))
        with self.assertRaisesRegex(ValueError, "compleja"):
            self.engine.load_calibration(self.h_path, self.coherence_path)

        invalid = np.ones(self.engine.bin_count, dtype=np.complex128)
        invalid[4] = np.nan + 1j
        self._write_calibration(response=invalid)
        with self.assertRaisesRegex(ValueError, "finit"):
            self.engine.load_calibration(self.h_path, self.coherence_path)

    def test_loading_rejects_coherence_outside_zero_to_one(self):
        coherence = np.ones(self.engine.bin_count)
        coherence[10] = 1.1
        self._write_calibration(coherence=coherence)

        with self.assertRaisesRegex(ValueError, "gamma"):
            self.engine.load_calibration(self.h_path, self.coherence_path)

        self.assertFalse(self.engine.is_calibrated)

    def test_synthetic_provenance_is_retained_and_missing_coherence_is_not_assumed(self):
        self._write_calibration(synthetic=True, coherence=np.zeros(self.engine.bin_count))

        self.engine.load_calibration(self.h_path, self.coherence_path)

        self.assertTrue(self.engine.calibration_synthetic)
        self.engine.bypass = False
        self.engine.recompute_inverse()
        np.testing.assert_array_equal(
            self.engine.equalizer.h_inverse,
            np.ones(self.engine.bin_count),
        )


if __name__ == "__main__":
    unittest.main()
