# Tareas: integración de la protoboard

## Tarea 1: Trazar y validar la protoboard

**Descripción:** Identificar CI, orientación, GND, alimentación, polarización
del micrófono y ruta de señal; medir voltajes DC antes de conectar audio USB.

**Criterios de aceptación:**
- [ ] Referencia y orientación del CI confirmadas.
- [ ] GND y rieles medidos respecto al GND USB.
- [ ] Salida del amplificador con DC seguro y ruta de señal documentada.

**Verificación:** registro de continuidad/voltajes y actualización del mapeo.

**Dependencias:** Ninguna. **Alcance:** M.

## Tarea 2: Validar la interfaz PCM2902 de forma aislada

**Descripción:** Con la protoboard desconectada, verificar el recorrido verde →
rosa con un tono de 1 kHz.

**Criterios de aceptación:**
- [ ] `py probar_loopback.py` informa `PASS`.
- [ ] Tono de 1 kHz y RMS superan −60 dBFS sin saturación.

**Verificación:** reporte de la prueba con dispositivo y niveles anotados.

**Dependencias:** Ninguna. **Alcance:** S.

## Tarea 3: Conectar la salida del amplificador de forma segura

**Descripción:** Implementar y verificar acople, atenuación/protección y tierra
común entre la salida identificada del amplificador y el canal izquierdo rosa.

**Criterios de aceptación:**
- [ ] Sin DC peligroso en la entrada de audio.
- [ ] `py verificar_entrada.py --duration 8` detecta señal > −60 dBFS.
- [ ] No hay clipping y el circuito no se calienta anormalmente.

**Verificación:** reporte JSON/SVG de entrada y revisión visual del montaje.

**Dependencias:** Tareas 1 y 2. **Alcance:** M.

## Tarea 4: Hacer reproducible el emparejamiento del programa

**Descripción:** Confirmar que los lanzadores y la app seleccionan por nombre el
PCM2902 correcto, entrada/salida a 48 kHz, sin depender de índices cambiantes.

**Criterios de aceptación:**
- [ ] Diagnóstico muestra el par USB PnP esperado.
- [ ] La app captura el canal izquierdo y reproduce en el PCM2902.

**Verificación:** `py iniciar_demo.py --check` (o diagnóstico equivalente).

**Dependencias:** Tarea 2. **Alcance:** S.

## Punto de control: adquisición física

- [ ] Tareas 1–4 aprobadas y la señal real llega al programa.

## Tarea 5: Calibrar la cadena eléctrica real

**Descripción:** Sustituir temporalmente el micrófono por el sweep controlado y
medir el circuito, sin sobrescribir los datos sintéticos hasta aprobar calidad.

**Criterios de aceptación:**
- [ ] `py calibrar_demo.py --amplitude 0.1` (o nivel seguro) termina correctamente.
- [ ] RMS > −60 dBFS, sin saturación y coherencia mediana > 0.8 entre 100 Hz y 10 kHz.
- [ ] `medicion_H.npy` y `medicion_gamma2.npy` se marcan como medición real con fecha/configuración.

**Verificación:** salida del calibrador y análisis de los archivos generados.

**Dependencias:** Tareas 3 y 4. **Alcance:** M.

## Tarea 6: Validar el filtro con la medición real

**Descripción:** Evaluar planitud, ganancia máxima, estabilidad y retardo con la
calibración de la Tarea 5 antes de escuchar audio en vivo.

**Criterios de aceptación:**
- [ ] Ganancia máxima limitada a un valor seguro (inicialmente ≤ 12 dB).
- [ ] La corrección no amplifica regiones de baja coherencia ni produce `CLIP`.
- [ ] Pruebas automatizadas relevantes pasan.

**Verificación:** análisis numérico y `py -m pytest -q`.

**Dependencias:** Tarea 5. **Alcance:** M.

## Tarea 7: Validar demo A/B en vivo

**Descripción:** Reconectar el micrófono, comparar bypass y ecualizado con
auriculares y documentar niveles/ajustes finales.

**Criterios de aceptación:**
- [ ] Diferencia A/B audible y estable.
- [ ] Sin acople ni indicador `CLIP` durante la demostración.

**Verificación:** ejecución guiada de `py iniciar_demo.py` con calibración real.

**Dependencias:** Tarea 6. **Alcance:** S.

## Tarea 8: Consolidar evidencia y guía de exposición

**Descripción:** Incorporar un reporte PASS/FAIL reproducible y actualizar la
guía con el cableado confirmado, comandos y resultados reales.

**Criterios de aceptación:**
- [ ] Un reporte conserva dispositivo, niveles, calibración y estado final.
- [ ] La guía permite repetir el montaje y demo sin asumir índices de audio.

**Verificación:** revisión de documentación y ejecución en seco de los comandos.

**Dependencias:** Tareas 5–7. **Alcance:** M.
