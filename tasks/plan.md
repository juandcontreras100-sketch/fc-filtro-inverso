# Plan de trabajo: integración de la protoboard con el filtro inverso

## Objetivo

Dejar una demostración reproducible en la que el micrófono y amplificador de la
protoboard alimenten la entrada del PCM2902, el programa ejecute en tiempo real
en bypass o con filtro inverso y la calibración usada sea una medición eléctrica
real, no la medición sintética actual.

## Estado conocido

- Windows reconoce el PCM2902 como `USB PnP Sound Device` (entrada y salida).
- La entrada reciente fue silencio: aproximadamente 44 dB.
- El loopback PCM2902 (salida verde a entrada rosa) tampoco detectó el tono de
  1 kHz; por tanto, todavía no se puede atribuir el fallo a la protoboard.
- La topología de la protoboard no está confirmada: modelo/orientación del CI,
  tierra, alimentación y cables de audio siguen sin medir.
- El programa ya dispone de captura, diagnóstico, calibración con barrido,
  controles de calidad y modos bypass/ecualizado. No se debe cambiar el DSP
  hasta disponer de una señal física válida.

## Decisiones de trabajo

- Usar el PCM2902 a 48 kHz y su canal de entrada 0; no Bluetooth.
- Tratar el ESP32/CP2102, si está presente, solo como telemetría ADC y no como
  interfaz de audio.
- No conectar directamente una salida de TL072 al PCM2902 ni al ADC: exigir
  acople, atenuación y tierra común verificada.
- Toda calibración nueva queda separada de los archivos sintéticos hasta que
  pase los criterios de calidad.

## Dependencias

```text
Seguridad y trazado eléctrico
        ↓
Loopback PCM2902 aprobado ──→ señal de protoboard aprobada
        ↓                             ↓
Configuración reproducible de audio ←─┘
        ↓
Calibración eléctrica real
        ↓
Validación del filtro
        ↓
Prueba A/B y documentación
```

## Fases

### Fase 1 — Identificación y seguridad

- [ ] Tarea 1: Confirmar visual y eléctricamente la protoboard.
- [ ] Tarea 2: Aislar y validar el PCM2902 con loopback.

### Punto de control 1

- [ ] Modelo/orientación del CI, alimentación y GND confirmados con medición.
- [ ] Loopback de 1 kHz aprobado o causa del PCM2902/cable corregida.

### Fase 2 — Emparejar hardware y adquisición

- [ ] Tarea 3: Integrar de forma segura la salida del amplificador con la entrada del PCM2902.
- [ ] Tarea 4: Fijar y comprobar el par de dispositivos que emplea el programa.

### Punto de control 2

- [ ] Micrófono de la protoboard produce señal útil (> 80 dB) sin saturación.
- [ ] Captura de entrada y salida está documentada y repetible a 48 kHz.

### Fase 3 — Emparejar la medición y el código

- [ ] Tarea 5: Generar y aceptar una calibración eléctrica real.
- [ ] Tarea 6: Validar el comportamiento del filtro inverso con dicha calibración.

### Punto de control 3

- [ ] Coherencia mediana 100 Hz–10 kHz > 0.8, sin clipping y archivos reales identificados.
- [ ] Ganancia y estabilidad del inverso dentro de límites seguros.

### Fase 4 — Demo reproducible

- [ ] Tarea 7: Ejecutar y documentar la comparación en vivo bypass/ecualizado.
- [ ] Tarea 8: Consolidar diagnóstico automático y guía de exposición.

### Punto de control final

- [ ] Demo A/B funcional con auriculares, sin realimentación ni `CLIP`.
- [ ] Instrucciones y evidencia de verificación actualizadas.

## Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| CI, GND o alimentación mal identificados | Alto | Multímetro y trazado antes de conectar la interfaz de audio. |
| Señal o DC excesivo hacia PCM2902 | Alto | Capacitor de acople, divisor/protección y comprobación de voltaje DC. |
| PCM2902/cable o Windows no enruta audio | Alto | Loopback aislado antes de involucrar el circuito. |
| Calibración de ruido/saturada | Medio | Rechazo automático por RMS, clipping y coherencia; no sobrescribir la referencia sintética. |
| Acople acústico durante A/B | Medio | Auriculares cerrados, volumen bajo y limitador activo. |

## Preguntas abiertas antes de la Fase 1

- ¿Disponemos ahora de un multímetro y de un cable AUX macho-macho funcional?
- ¿Cuál es la referencia impresa exacta del circuito integrado de ocho patas?
- ¿El PCM2902 sigue exponiendo el mismo dispositivo de entrada/salida después de reconectarlo?
