# Filtro inverso con interfaz en tiempo real

Ecualizador FFT full-duplex en tiempo real: mide la respuesta de un circuito,
calcula su filtro inverso regularizado y lo aplica mientras se ven los
espectros de entrada y salida en vivo.

## Inicio rápido

Desde esta carpeta, cree y active un entorno virtual e instale las dependencias:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Como alternativa, puede instalar el proyecto editable con sus herramientas de
prueba: `py -m pip install -e ".[test]"`.

Abra la interfaz de espectros y ecualización en tiempo real:

```powershell
py main.py
```

Sin argumentos se abre en modo *bypass*: sirve para comprobar el micrófono, los altavoces y las gráficas sin usar una corrección medida. Use auriculares o volumen bajo para evitar realimentación acústica.

### Accesos directos

`crear_accesos.ps1` genera tres accesos en el escritorio:

| Acceso | Qué abre |
|---|---|
| `Filtro Inverso.lnk` | La interfaz gráfica, sin ventana de consola. |
| `Filtro Inverso (Consola).lnk` | La misma interfaz conservando la consola para ver errores. |
| `Calibrar Filtro Inverso.lnk` | La calibración del circuito. |

```powershell
powershell -ExecutionPolicy Bypass -File .\crear_accesos.ps1
```

## La interfaz

- **Panel de control** (izquierda): modo *bypass* / *ecualizado*, carga de la
  medición `.npy`, ganancia máxima, estabilidad (ε), interfaz de audio,
  frecuencia de muestreo y tamaño de FFT.
- **Medidores de nivel**: pico y RMS de entrada y salida. Un limitador suave
  protege la salida; `CLIP` indica que la señal previa al limitador habría
  alcanzado el umbral de saturación.
- **Espectros**: entrada (naranja) y salida (verde), más la curva del filtro
  inverso aplicado (azul discontinuo). Pase el ratón por encima para leer la
  frecuencia y los dB exactos del cursor. La salida usa dBFS y el filtro tiene
  un eje independiente de ganancia en dB.
- **Cascada**: historial del espectro de salida; el tiempo avanza hacia abajo.

Cambiar la interfaz, el muestreo o el tamaño de FFT reinicia el stream de audio.
La calibración se descarta si deja de ser compatible con el nuevo tamaño de FFT
o con la nueva frecuencia de muestreo.

### Elegir la interfaz de audio (Windows)

En Windows la entrada y la salida de una misma interfaz (por ejemplo el ESP32-S3
como tarjeta de sonido USB) aparecen como dos dispositivos separados. La lista
«Interfaz» los ofrece ya emparejados por nombre, primero WASAPI y luego MME.
Para ver qué detecta el programa en su equipo:

```powershell
py -m src.audio.devices
```

Desde la línea de comandos también se pueden indicar a mano:

```powershell
py main.py --input-device 9 --output-device 8
```

La salida está limitada suavemente a 0,98 de la plena escala para no recortar
con ganancias altas del filtro; el medidor CLIP sigue avisando cuando el filtro
pide más.

## Calibración

Para generar una respuesta medida, conecte el circuito, mantenga un volumen seguro y ejecute:

```powershell
py calibrate_hardware.py
```

El calibrador reproduce cuatro barridos por defecto, promedia la respuesta y
guarda `medicion_H.npy` y la coherencia cuadrática `medicion_gamma2.npy`. Ajuste
el número de capturas con `--sweeps`; se requieren al menos dos. Después inicie
la interfaz con los archivos producidos:

```powershell
py main.py --h-response medicion_H.npy --coherence medicion_gamma2.npy
```

Para guardar también una respuesta impulsional diagnóstica mediante
deconvolución Farina, añada `--farina-output medicion_ir.npy`. Este archivo no
se usa para construir el filtro inverso.

También puede cargar la medición desde el botón *Cargar medicion (.npy)...* del
panel de control, sin reiniciar el programa.

Para probar la interfaz sin circuito, `generar_medicion_demo.py` sintetiza una
medición con muescas en 1 kHz y 4 kHz. Acepte el formato predeterminado o
especifique el mismo tamaño FFT y frecuencia de muestreo que usará en la interfaz:

```powershell
py generar_medicion_demo.py --fft-size 4096 --sample-rate 96000
py main.py --fft-size 4096 --sample-rate 96000 --h-response medicion_H.npy --coherence medicion_gamma2.npy
```

## Pruebas automatizadas

```powershell
py -m pytest -q
```

## Análisis numérico

Ejecute el análisis de la medición demo para comparar la solución analítica con
el descenso de gradiente y obtener condición, retardo de grupo, energía y corte:

```powershell
py analisis_numerico.py
```

Para evaluar linealidad, mida el circuito a dos amplitudes distintas y guarde
las respuestas en carpetas separadas:

```powershell
py calibrate_hardware.py --amplitude 0.05 --output .\medicion_baja\medicion_H.npy
py calibrate_hardware.py --amplitude 0.15 --output .\medicion_alta\medicion_H.npy
py analisis_numerico.py --linearity-low .\medicion_baja\medicion_H.npy --linearity-high .\medicion_alta\medicion_H.npy
```

Para analizar una medición concreta, use `--h-response`, `--coherence`,
`--sample-rate` y `--fft-size` con el mismo formato de captura.

## Estructura del proyecto

```
main.py                       Punto de entrada: argumentos de línea de comandos.
calibrate_hardware.py         Medición por barrido logarítmico (log sine sweep).
generar_medicion_demo.py      Medición sintética para probar la interfaz.
crear_accesos.ps1             Genera los accesos directos del escritorio.
src/audio/engine.py           Stream full-duplex, ecualizador y calibración.
src/dsp/inverse_filter.py     Filtro inverso Wiener/Tikhonov y procesador WOLA.
src/dsp/numerics.py           Métodos numéricos y métricas académicas.
analisis_numerico.py          Comparación analítica/gradiente y linealidad.
src/dsp/spectrum.py           Espectro, niveles y ganancia en dB.
src/ui/window.py              Ventana principal: gráfica, panel y refresco.
src/ui/controls.py            Panel de control.
src/ui/meters.py              Medidores de nivel e indicador de saturación.
src/ui/spectrogram.py         Cascada temporal.
src/ui/readout.py             Lectura de valores bajo el cursor.
src/ui/theme.py               Paleta de la interfaz.
```
