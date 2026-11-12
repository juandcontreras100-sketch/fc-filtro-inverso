# Prompt para un agente de IA

Copiá y pegá este prompt en otro agente con acceso a esta máquina:

---

Sos un agente de IA con acceso a Windows y al proyecto `C:\Users\juan_\Desktop\Projects\fc-filtro-inverso`. Tu objetivo es dejar lista y verificada una demo del filtro inverso en la que el micrófono y la amplificación TL072 del circuito sean la entrada de audio, y se escuchen los dos escenarios: bypass (sin filtro) y ecualizado (con filtro inverso).

## Contexto

- La app usa `sounddevice`/PortAudio en full-duplex. La interfaz de audio es el **USB PnP Sound Device (PCM2902)**, entrada MME 1 y salida MME 7, a 48 kHz.
- El **ESP32-D0WD-V3** está en COM5. Su CP2102 es solo serial: no es un micrófono USB. El ESP32 se usa únicamente como monitor del TL072 (ADC GPIO34 con divisor), no transporta audio.
- La app puede estar abierta con `py iniciar_demo.py`, en bypass, 48 kHz, FFT 2048. Ese lanzador fuerza el par PCM2902.
- La entrada del PCM2902 está en silencio (≈ 43.3 dB). El circuito micrófono → TL072 → PCM2902 no está entregando señal o no está conectado.
- `medicion_H.npy` y `medicion_gamma2.npy` actuales son la medición **sintética** de `generar_medicion_demo.py` (muescas en 1 kHz y 4 kHz, coherencia 1.0). No usarlas como medición real.
- Ya existen `iniciar_demo.py` y `calibrar_demo.py`, que fuerzan el par PCM2902. También existe `LEEME_DEMO.md` con el procedimiento.

## Tareas

1. **Verificar la entrada.** Ejecutar una captura de 2 s con `sounddevice` en el dispositivo de entrada 1 y calcular RMS/dB (amplitud 1.0 = 140 dB). Si sigue en silencio, guiar al usuario a revisar: jack rosa, cable, capacitor de acople, divisor, bias del electret, nivel/mute de Windows y salida del TL072. No continuar hasta obtener señal.
2. **Verificar la polarización del TL072.** Medir pin 8 vs pin 4 y el DC en pin 1/pin 7. En single-supply, la salida debe estar cerca de V+/2; en dual, cerca de 0 V. Si está saturada, no conectar audio.
3. **Calibrar eléctricamente.** PCM2902 salida → 1 µF + 2.2 kΩ → entrada del TL072; salida del TL072 → PCM2902 entrada con 10 µF + divisor. Ejecutar `py calibrar_demo.py --amplitude 0.1`. Verificar que no sature y que la coherencia mediana entre 100 Hz y 10 kHz sea > 0.8. Guardar `medicion_H.npy` y `medicion_gamma2.npy` reales.
4. **Verificar el filtro inverso.** Calcular `estimate_h_inverse`, medir planitud antes/después en 100 Hz–10 kHz, ganancia máxima ≤ 12 dB y retardo de grupo. Si la mejora es pobre, ajustar ε/ganancia o revisar la medición.
5. **Probar en vivo.** Reconectar el micrófono y abrir `py iniciar_demo.py --h-response medicion_H.npy --coherence medicion_gamma2.npy`. Comparar bypass vs ecualizado con auriculares, sin `CLIP` ni realimentación.
6. **Automatizar la verificación.** Crear `verificar_circuito.py` con un reporte PASS/FAIL, `verificacion/reporte.json` y gráficos. Agregar pruebas unitarias con datos sintéticos.
7. **Documentar.** Actualizar `LEEME_DEMO.md` con los comandos exactos, el cableado final y los resultados obtenidos.

## Restricciones

- Windows; usar `py`, no `python`.
- No usar Bluetooth para la demo: agrega latencia y deriva de reloj.
- No conectar la salida del TL072 directa al PCM2902 ni al ADC del ESP32 sin acople, atenuación y protección.
- La app usa el canal 0 de entrada: conectar la señal al canal izquierdo.
- Si el TL072 es dual-supply (±12 V), no conectar su salida al ADC sin clamp/level shifter.
- No tocar el backup `C:\Users\juan_\esp32-diag\backup\esp32_original_4MB.bin`.

## Entregables

- Entrada del PCM2902 con señal verificada.
- Calibración real del TL072 (`medicion_H.npy` + `medicion_gamma2.npy`).
- A/B funcionando: bypass vs filtro inverso.
- Reporte de verificación PASS/FAIL.
- Guía breve y reproducible para la exposición.
