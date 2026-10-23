"""Pruebas del callback de audio sin hardware: se invoca como lo haria PortAudio."""

import unittest

import numpy as np

from src.audio import engine as engine_module
from src.audio.engine import AudioEngine

FFT_SIZE = 512
HOP = FFT_SIZE // 2


def _drain(q) -> list[str]:
    items: list[str] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


class CallbackRobustnessTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=FFT_SIZE)
        self.tone = (0.3 * np.sin(2 * np.pi * 0.05 * np.arange(HOP * 16))).astype(np.float32)

    def _run_block(self, block: np.ndarray, frames: int | None = None) -> np.ndarray:
        indata = block.astype(np.float32)[:, np.newaxis]
        outdata = np.ones((indata.shape[0], 1), dtype=np.float32)  # basura previa
        self.engine._callback(indata, outdata, frames or block.size, None, None)
        return outdata

    def _clean_block(self, index: int) -> np.ndarray:
        return self.tone[index * HOP : (index + 1) * HOP]

    def test_non_finite_input_does_not_raise_and_output_stays_finite(self):
        self._run_block(self._clean_block(0))  # preroll

        bad = self._clean_block(1).copy()
        bad[10] = np.nan
        bad[20] = np.inf
        bad[30] = -np.inf
        out = self._run_block(bad)

        self.assertTrue(np.all(np.isfinite(out)))

    def test_non_finite_input_does_not_poison_the_following_blocks(self):
        self._run_block(self._clean_block(0))
        bad = self._clean_block(1).copy()
        bad[:] = np.nan
        self._run_block(bad)

        for index in range(2, 8):
            out = self._run_block(self._clean_block(index))
            self.assertTrue(np.all(np.isfinite(out)), f"bloque {index} no finito")

    def test_non_finite_input_is_reported_on_the_status_queue(self):
        self._run_block(self._clean_block(0))
        bad = self._clean_block(1).copy()
        bad[0] = np.nan
        self._run_block(bad)

        messages = _drain(self.engine.status_queue)
        self.assertTrue(any("no finit" in message for message in messages), messages)

    def test_an_internal_failure_is_contained_silenced_and_reported(self):
        def boom(_block):
            raise RuntimeError("fallo simulado")

        self.engine.equalizer.process_block = boom  # type: ignore[method-assign]
        out = self._run_block(self._clean_block(0))

        self.assertTrue(np.all(out == 0.0))
        messages = _drain(self.engine.status_queue)
        self.assertTrue(any("fallo simulado" in message for message in messages), messages)

    def test_the_engine_keeps_working_after_an_internal_failure(self):
        original = self.engine.equalizer.process_block
        calls = {"count": 0}

        def flaky(block):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("fallo unico")
            return original(block)

        self.engine.equalizer.process_block = flaky  # type: ignore[method-assign]

        outputs = [self._run_block(self._clean_block(i)) for i in range(8)]

        self.assertTrue(all(np.all(np.isfinite(out)) for out in outputs))
        # Tras recuperarse, el ecualizador vuelve a entregar audio real.
        self.assertGreater(float(np.max(np.abs(outputs[-1]))), 0.05)

    def test_an_unexpected_block_size_is_reported_with_silence(self):
        out = self._run_block(np.zeros(100, dtype=np.float32))

        self.assertTrue(np.all(out == 0.0))
        messages = _drain(self.engine.status_queue)
        self.assertTrue(any("inesperado" in message for message in messages), messages)


class CallbackLimiterTests(unittest.TestCase):
    def setUp(self):
        self.engine = AudioEngine(sample_rate=48_000.0, fft_size=FFT_SIZE)
        self.tone = (0.3 * np.sin(2 * np.pi * 0.05 * np.arange(HOP * 16))).astype(np.float32)

    def _run(self, blocks: int = 12) -> list[np.ndarray]:
        outputs = []
        for index in range(blocks):
            block = self.tone[(index % 16) * HOP : (index % 16 + 1) * HOP]
            indata = block[:, np.newaxis]
            outdata = np.zeros_like(indata)
            self.engine._callback(indata, outdata, HOP, None, None)
            outputs.append(outdata)
        return outputs

    def _boost(self, gain: float) -> None:
        # Asignacion directa: salta el tope de max_gain_db para simular un filtro agresivo.
        self.engine.equalizer.h_inverse = np.full(self.engine.bin_count, gain, dtype=np.complex128)

    def test_loud_output_is_kept_under_the_ceiling(self):
        self._boost(10.0)  # +20 dB: el tono de 0,3 llegaria a ~3,0
        peak = max(float(np.max(np.abs(out))) for out in self._run())

        self.assertLessEqual(peak, engine_module.OUTPUT_CEILING + 1e-6)
        self.assertGreater(peak, 0.8)  # si esta limitando, no esta mudo

    def test_normal_levels_are_not_touched(self):
        peak = max(float(np.max(np.abs(out))) for out in self._run())

        self.assertAlmostEqual(peak, 0.3, delta=0.02)

    def test_the_display_receives_the_raw_output_so_clip_can_light_up(self):
        self._boost(10.0)
        self._run()

        frame = self.engine.display_queue.get_nowait()
        while not self.engine.display_queue.empty():
            frame = self.engine.display_queue.get_nowait()
        self.assertGreater(float(np.max(np.abs(frame.output_samples))), 1.0)

    def test_the_limiter_can_be_disabled(self):
        self._boost(10.0)
        self.engine.limiter_enabled = False
        peak = max(float(np.max(np.abs(out))) for out in self._run())

        self.assertGreater(peak, 1.0)


if __name__ == "__main__":
    unittest.main()
