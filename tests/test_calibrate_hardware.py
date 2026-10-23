import unittest

import numpy as np

from calibrate_hardware import estimate_transfer, log_sine_sweep


class CalibrationTests(unittest.TestCase):
    def test_transfer_recovers_gain_after_capture_delay(self):
        sample_rate = 48_000.0
        sweep = log_sine_sweep(sample_rate, 0.1, 500.0, 10_000.0, 0.1)
        delay = 321
        gain = 0.6
        recorded = np.concatenate((np.zeros(delay), gain * sweep, np.zeros(500)))

        transfer, start = estimate_transfer(
            sweep,
            recorded,
            sample_rate=sample_rate,
            fft_size=256,
            start_freq=500.0,
            stop_freq=10_000.0,
        )
        frequencies = np.fft.rfftfreq(256, d=1.0 / sample_rate)
        # Se excluye el borde de 10 kHz: allí termina el fundido del sweep y
        # el calibrador lo deja correctamente en bypass si no hay energía.
        in_band = (frequencies >= 500.0) & (frequencies <= 9_000.0)

        self.assertEqual(start, delay)
        self.assertTrue(np.allclose(transfer[in_band], gain, rtol=1e-3, atol=1e-6))
        self.assertEqual(transfer[0], 1.0)  # fuera de la banda medida: bypass
