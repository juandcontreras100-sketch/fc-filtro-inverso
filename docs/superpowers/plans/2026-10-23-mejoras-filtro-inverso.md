# Hoja de ruta de mejoras — Filtro inverso en tiempo real

**Fecha de exposición:** martes 10 de noviembre de 2026 (18 días desde el 23-oct).
**Criterio de prioridad:** (riesgo de que falle o sea peligroso en la demo) × (qué tan barato es arreglarlo), después valor académico y por último pulido. Todo cambio de código va con prueba primero (TDD).

## Prioridades

| ID | Mejora | Origen | Estado |
|---|---|---|---|
| **P0-1** | `configure()` deja el stream consistente al descartar la calibración; excepción propia `CalibrationDiscardedError`; un cambio de frecuencia de muestreo también invalida la calibración | Análisis propio | Hecho |
| **P0-2** | El callback de audio nunca lanza: contiene fallos, sanea NaN/Inf y avisa por `status_queue` | Cursor #3 (parcial) | Hecho |
| **P0-3** | Limitador suave (`src/dsp/limiter.py`) antes del DAC; el medidor CLIP sigue viendo la salida sin limitar | Cursor #4 | Hecho |
| **P0-4** | Selección de dispositivo en Windows: aceptar el par entrada/salida y emparejar el ESP32 | Análisis propio | Pendiente: necesita la lista real de dispositivos de tu PC |
| **P1-1** | Mover el código académico de la raíz a `src/dsp/numerics.py` (con sus pruebas en `tests/`), sin duplicar `epsilon_profile` ni `estimate_h_inverse` | Cursor #1 | Pendiente |
| **P1-2** | Coherencia real en la calibración: promediar varios barridos, calcular γ² y guardar `medicion_gamma2.npy`; corregir el README | Cursor #2 | Pendiente |
| **P1-3** | `generar_medicion_demo.py` con `--fft-size` y `--sample-rate` | Cursor #10 | Pendiente |
| **P2-1** | Curva del filtro en un eje secundario propio (ganancia en dB, no dBFS) | Cursor #5 | Pendiente |
| **P2-2** | `analisis_numerico.py`: gradiente vs analítica, número de condición, retardo de grupo, Simpson, bisección y comparación de dos amplitudes para medir linealidad | Análisis propio | Pendiente |
| **P2-3** | Accesos directos que prefieran `.venv`; fijar versiones en `requirements.txt` | Cursor #9 y #7 (parcial) | Pendiente |
| **P3** | Buffer circular preasignado, espectro alineado con la trama WOLA, deconvolución de Farina, `pyproject.toml`, aviso al salir de bypass | Cursor #3, #6, #8, #7, #11 | Diferido |

## Por qué se difiere lo de P3

- **Buffer circular (Cursor #3):** con bloques de 256 a 1024 muestras, el `np.concatenate` por bloque cuesta microsegundos. Se mide antes de optimizar; lo que sí rompía el stream (NaN y excepciones) ya está resuelto en P0-2.
- **Espectro vs trama WOLA (Cursor #6):** es una diferencia de resolución visual, no un error; no afecta el audio.
- **Farina (Cursor #8):** el método actual basta para estimar H lineal. Para saber cuánto se aparta el circuito de ser lineal es más útil y barato comparar dos calibraciones a distinta amplitud (P2-2).
- **Aviso al salir de bypass (Cursor #11):** el limitador (P0-3) ya acota el peor caso de realimentación; queda como extra.

## Cronograma

| Fechas | Trabajo |
|---|---|
| Vie 23 – dom 25 oct | P0-1, P0-2 y P0-3 (hechos). Tú: ejecutar `py -c "import sounddevice as sd; print(sd.query_devices())"` con el ESP32 conectado para cerrar P0-4 |
| Lun 26 – vie 30 oct | P0-4, P1-1, P1-2, P1-3. Tú, en el banco: niveles hacia el PCM1808, tierra común y calibración real |
| Sáb 31 oct – jue 5 nov | P2-1, P2-2, P2-3. Firmware del ESP32-S3 revisado o plan B (tarjeta de sonido del PC) ensayado |
| Vie 6 nov | **Congelar el código**: desde aquí solo se corrigen bugs |
| Sáb 7 – lun 9 nov | Ensayo completo de principio a fin, con calibración real guardada y respaldo con la medición sintética |
| Mar 10 nov | Exposición |

## Notas de verificación

- Las pruebas se escriben con `unittest`, igual que el resto de `tests/`. Se ejecutan con `py -m pytest -q` o con `py -m unittest discover -s tests -t .`.
- `tests/test_controls.py` necesita PySide6 y no se pudo ejecutar en el entorno donde se hicieron los cambios de P0; ejecútalo en tu PC.
- El `test_inverse_filter.py` de la raíz usa pytest y sigue fuera de `testpaths` hasta completar P1-1.
