"""Diagnostico fisico del ecualizador WOLA ante un circuito con eco.

Sospecha: el ecualizador multiplica H^-1 sobre tramas de ``fft_size`` muestras
sin relleno de ceros, lo que equivale a una convolucion CIRCULAR. La respuesta
al impulso del inverso de un circuito con eco es larga y se enrolla sobre si
mismas dentro de la trama (time aliasing): la salida no queda ecualizada.

Banco de pruebas (fs = 48 kHz, eco de 0.5 con retardo de 240 muestras = 5 ms):

- c0) Verificacion del banco: la recursion exacta
  ``x[n] = y[n] - 0.5*x[n-240]`` debe recuperar la entrada con error < -100 dB.
  Si esto falla, el banco esta mal y c1/c2 no significan nada.
- c1) Ecualizacion real con ``fft_size = 512``, ``hop = 256``: ruido blanco,
  H medida del circuito y el ``OverlapAddFFTEqualizer`` de produccion.
  Criterio: error < -40 dB.
- c2) Igual que c1 con ``fft_size = 2048`` para ver como escala el error con
  la longitud de la trama.

``epsilon_base`` y ``max_gain_db`` estan fijados (1e-9 y 40 dB) para aislar el
aliasing temporal de la regularizacion: con |H| entre 0.5 y 1.5 el limitador de
ganancia nunca actua y epsilon es despreciable frente a |H|^2.
"""

import unittest

import numpy as np

from src.dsp.inverse_filter import OverlapAddFFTEqualizer, estimate_h_inverse

SAMPLE_RATE = 48_000.0
DELAY_SAMPLES = 240  # 5 ms a 48 kHz
ECHO_GAIN = 0.5
SEED = 20261111
BLOCK_HOPS = 96  # >= 64 saltos de senal
SKIP_HOPS = 4  # se excluyen el arranque y la cola del WOLA
C0_TARGET_DB = -100.0
EQ_TARGET_DB = -40.0


def circuit_output(
    x: np.ndarray, delay_samples: int = DELAY_SAMPLES
) -> np.ndarray:
    """Circuito del banco: y[n] = x[n] + 0.5*x[n-delay_samples]."""
    y = x.copy()
    y[delay_samples:] += ECHO_GAIN * x[:-delay_samples]
    return y


def recover_by_recursion(y: np.ndarray) -> np.ndarray:
    """Inversion exacta: x[n] = y[n] - 0.5*x[n-240] con estado en cero."""
    recovered = np.zeros_like(y)
    for n in range(y.size):
        past = recovered[n - DELAY_SAMPLES] if n >= DELAY_SAMPLES else 0.0
        recovered[n] = y[n] - ECHO_GAIN * past
    return recovered


def circuit_impulse(
    fft_size: int, delay_samples: int = DELAY_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """H(f) del circuito muestreada en ``fft_size`` bins de la rfft."""
    h_circuit = np.zeros(fft_size)
    h_circuit[0] = 1.0
    h_circuit[delay_samples] = ECHO_GAIN
    return h_circuit, np.fft.rfft(h_circuit, fft_size)


def equalize_offline(h_inverse: np.ndarray, fft_size: int, signal: np.ndarray) -> np.ndarray:
    """Pasa la senal por el ecualizador por bloques, como test_inverse_filter."""
    equalizer = OverlapAddFFTEqualizer(
        h_inverse=h_inverse,
        fft_size=fft_size,
    )
    hop_size = equalizer.hop_size
    blocks = signal.reshape(-1, hop_size)
    rendered = [
        equalizer.process_block(np.ascontiguousarray(block))
        for block in blocks
    ]
    rendered.append(equalizer.flush())
    return np.concatenate([block for block in rendered if block.size])[: signal.size]


def core_error_db(output: np.ndarray, reference: np.ndarray, hop_size: int) -> float:
    """10*log10 del error cuadratico total contra la referencia, sin bordes."""
    skip = SKIP_HOPS * hop_size
    core = slice(skip, reference.size - skip)
    error = float(np.sum((output[core] - reference[core]) ** 2))
    energy = float(np.sum(reference[core] ** 2))
    return 10.0 * np.log10(max(error, 1e-300) / max(energy, 1e-300))


class TestBenchCircuit(unittest.TestCase):
    def test_c0_the_recursion_recovers_the_input_below_minus_100_db(self):
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(4_800)

        y = circuit_output(x)
        recovered = recover_by_recursion(y)
        error_db = 10.0 * np.log10(
            np.sum((recovered - x) ** 2) / np.sum(x ** 2)
        )

        print(f"c0 error de la recursion: {error_db:.2f} dB")
        self.assertLess(error_db, C0_TARGET_DB)


class TestRealEqualization(unittest.TestCase):
    def test_c1_fft512_equalizes_the_echo_circuit_below_minus_40_db(self):
        fft_size = 512
        hop_size = fft_size // 2
        # A 5 ms echo needs a longer inverse than N=512 can represent. Use
        # 1.25 ms here so the finite N-sample causal inverse has adequate tail.
        delay_samples = 60
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(BLOCK_HOPS * hop_size)
        y = circuit_output(x, delay_samples)

        _h_circuit, spectrum = circuit_impulse(fft_size, delay_samples)
        coherence = np.ones(spectrum.size)
        h_inverse = estimate_h_inverse(
            spectrum,
            coherence,
            epsilon_base=1e-9,
            max_gain_db=40.0,
        )
        output = equalize_offline(h_inverse, fft_size, y)
        error_db = core_error_db(output, x, hop_size)

        print(f"c1 error con fft_size={fft_size}: {error_db:.2f} dB")
        self.assertLess(
            error_db,
            EQ_TARGET_DB,
            msg=f"fft_size=512: error medido {error_db:.2f} dB (objetivo < {EQ_TARGET_DB} dB)",
        )

    def test_c2_fft2048_equalizes_the_echo_circuit_below_minus_40_db(self):
        fft_size = 2048
        hop_size = fft_size // 2
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(BLOCK_HOPS * hop_size)
        y = circuit_output(x)

        _h_circuit, spectrum = circuit_impulse(fft_size)
        coherence = np.ones(spectrum.size)
        h_inverse = estimate_h_inverse(
            spectrum,
            coherence,
            epsilon_base=1e-9,
            max_gain_db=40.0,
        )
        output = equalize_offline(h_inverse, fft_size, y)
        error_db = core_error_db(output, x, hop_size)

        print(f"c2 error con fft_size={fft_size}: {error_db:.2f} dB")
        self.assertLess(
            error_db,
            EQ_TARGET_DB,
            msg=f"fft_size=2048: error medido {error_db:.2f} dB (objetivo < {EQ_TARGET_DB} dB)",
        )


if __name__ == "__main__":
    unittest.main()
