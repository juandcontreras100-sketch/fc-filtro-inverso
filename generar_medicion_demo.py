"""Genera una medicion sintetica con una muesca para probar la carga de calibracion."""

import sys

import numpy as np

fft_size = 2048
sample_rate = 48_000.0
bins = fft_size // 2 + 1
frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

# Respuesta base plana con una muesca de -18 dB en 1 kHz y otra de -9 dB en 4 kHz.
magnitude = np.ones(bins)
for center, depth_db, width_hz in ((1_000.0, -18.0, 250.0), (4_000.0, -9.0, 500.0)):
    magnitude *= 10.0 ** (depth_db / 20.0 * np.exp(-((frequencies - center) / width_hz) ** 2))

phase = np.zeros(bins)
h = magnitude * np.exp(1j * phase)
h[0] = abs(h[0])          # fase cero en DC
h[-1] = abs(h[-1])        # Nyquist real

np.save("medicion_H.npy", h.astype(np.complex128))
np.save("medicion_gamma2.npy", np.ones(bins, dtype=np.float64))

print(f"medicion_H.npy: {bins} bins, muesca de -18 dB en 1 kHz y -9 dB en 4 kHz")
