# Demo: circuito (PCM2902 + TL072) y respaldo con la laptop

## Estado actual

- **Camino elegido: demo con el micrófono y los parlantes de la laptop.** No usa el circuito ni el PCM2902.
- La demo quedó **calibrada y verificada**: vos confirmaste que *Ecualizado* suena mejor que *Bypass*, sin CLIP.
- Calibración real guardada en `verificacion/laptop/`: coherencia mediana **0.929** entre 100 Hz y 10 kHz y RMS mínimo **−32,88 dBFS**.
- Los índices de `sounddevice` **cambian** cuando Bluetooth se conecta o desconecta. Los lanzadores fijan los dispositivos **por nombre**, no por índice.
- Para que el micrófono capte los parlantes hay que desactivar las **mejoras de audio** de Windows (por dispositivo y globales). Sin eso, la cancelación de eco borra el barrido y la calibración falla.
- Niveles usados en la demo: micrófono **90 %**, parlantes **50 %**, ganancia máxima del filtro **6 dB**. Con el micrófono al 50 % la salida quedaba inaudible.
- El PCM2902 sigue **sin verificar**: su salida no se escuchó por el jack verde, el loopback verde→rosa falló y la entrada no captó ninguna fuente conocida.
- La entrada del PCM2902 dio **−94,90 dBFS** con el circuito desconectado; el loopback, **−96,32 dBFS** con la componente de 1 kHz en **−127,03 dBFS**.
- La laptop reprodujo el tono por los **parlantes**, no por el cable: su jack no detecta auriculares de cable. Bluetooth se reconecta solo y se roba la salida.
- La protoboard **no está trazada**: no se sabe si el chip es TL072, cuál es el GND real ni qué lado del capacitor/divisor es la señal. Ver `MAPEO_PROTOBOARD.md`.
- **`medicion_H.npy` / `medicion_gamma2.npy` de la raíz siguen siendo sintéticas.** La calibración real de la laptop está en `verificacion/laptop/`.
- Si reaparece olor a quemado o calentamiento anormal, desconectá y detené las pruebas.

## Demo de respaldo con la laptop

Usa el micrófono y los parlantes de la laptop. Es la demo que funciona hoy,
pero **no pasa por el micrófono ni por el TL072 del circuito**.

### Preparación

1. Apagá los auriculares Bluetooth y guardalos en el estuche.
2. Desactivá las mejoras de audio:
   - Configuración → Sistema → Sonido → Todos los dispositivos de sonido → **Microphone Array (Qualcomm)** → **Mejoras de audio: Desactivado**.
   - Configuración → Sistema → Sonido → **Avanzado** → **Permitir que las aplicaciones usen mejoras de audio: Desactivado**.
3. Dejá la salida predeterminada de Windows en **Speakers (Qualcomm)** y el volumen en ~50 %. Micrófono en ~90 %.
4. Verificá el par de audio:

```powershell
cd C:\Users\juan_\Desktop\Projects\fc-filtro-inverso
py iniciar_demo_laptop.py --check
```

### Abrir la demo calibrada

```powershell
py iniciar_demo_laptop.py --h-response verificacion\laptop\medicion_H.npy --coherence verificacion\laptop\medicion_gamma2.npy --max-gain-db 6
```

- En la app, dejá **Interfaz: “Sistema”**, **Muestreo: 48 kHz**, **FFT: 2048**.
- Arranca en **Ecualizado**. Hablá a 30–50 cm del micrófono y alterná **Bypass (sin corrección)** para comparar.
- Si aparece **CLIP**, bajá el micrófono a 75 % o la ganancia máxima a 3–6 dB.
- Si aparece acople, bajá el volumen de los parlantes y alejate del micrófono.

### Calibrar de nuevo

```powershell
py calibrar_demo_laptop.py --amplitude 0.05 --sweeps 4
```

Guarda `verificacion/laptop/medicion_H.npy` y `verificacion/laptop/medicion_gamma2.npy`. Solo los acepta si no hay saturación y la coherencia mediana entre 100 Hz y 10 kHz supera 0.8. **No limites el barrido a 10 kHz**: en esta laptop la coherencia cae a 0.43 y la calibración se rechaza. El barrido completo (20 Hz–20 kHz) dio 0.929.

> La calibración mide la respuesta acústica de la laptop (parlantes + ambiente + micrófono), no la del TL072. No sirve como calibración del circuito.

### Si no se escucha nada

1. Confirmá que los parlantes suenan con un tono ajeno a la app:

```powershell
py probar_salida.py --device Qualcomm --duration 3 --amplitude 0.1 --no-prompt
```

2. Abrí la app con diagnóstico para ver el par de audio y los niveles reales:

```powershell
$env:FC_DEBUG_AUDIO='1'
py iniciar_demo_laptop.py --h-response verificacion\laptop\medicion_H.npy --coherence verificacion\laptop\medicion_gamma2.npy --max-gain-db 6
```

El diagnóstico imprime `[audio] entrada ...`, `[audio] salida ...`, y una línea por segundo con `in`/`out` en dBFS. Si `out` está por debajo de −50 dBFS, subí el micrófono o el volumen; si `out` es razonable pero no se oye, revisá la salida predeterminada de Windows y el mute.

## Demo con el circuito (bloqueada)

Las secciones siguientes describen el camino original con PCM2902 + TL072.
**No están verificadas** y no deben ejecutarse hasta resolver el mapeo de la
protoboard y la ruta de audio del PCM2902.

## Cadena

```
Micrófono → TL072 (amplificación/filtro) → PCM2902 entrada (rosa) → USB → PC
                                                                          ↓
Auriculares ← PCM2902 salida (verde) ← app (bypass / filtro inverso) ←────┘
```

- Entrada: jack **rosa** del PCM2902.
- Salida: jack **verde**, con auriculares cerrados.
- No uses Bluetooth para la demo: agrega latencia y deriva de reloj.
- La app usa el **canal 0** (izquierdo); conectá la señal a ese canal.

## 1. Verificar que el micrófono del circuito llega a la PC

Con el circuito alimentado y el micrófono conectado al TL072:

```powershell
cd C:\Users\juan_\Desktop\Projects\fc-filtro-inverso
py -c "import sounddevice as sd, numpy as np; x=sd.rec(int(2*48000), samplerate=48000, channels=1, device=1, dtype='float32', blocking=True); r=float(np.sqrt(np.mean(x*x))); print('RMS', r, 'dBFS', 20*np.log10(max(r,1e-12)))"
```

Hablá o soplá al micrófono. Deberías ver un RMS bastante mayor que −60 dBFS. Si la app está abierta, mirá directamente su medidor de entrada; si no se mueve, cerrá la app y ejecutá el one-liner (el PCM2902 no siempre permite dos capturas simultáneas). Si sigue en −96 dBFS, revisá:

- Jack rosa, no el verde.
- Capacitor de acople y divisor: que no corten la señal.
- Nivel de grabación de Windows y mute.
- Bias del electret: sin alimentación, el micrófono no funciona.
- Salida del TL072: medila con multímetro o con el ADC del ESP32.

Antes de probar el circuito, aislá el PCM2902 con loopback:

1. Cerrá la app.
2. Conectá un cable AUX estéreo de 3,5 mm macho-macho entre la salida verde y la entrada rosa del **mismo PCM2902**. Desconectá la protoboard/TL072 de esos jacks.
3. Dejá el volumen de salida bajo (aprox. 50 %) y ejecutá `py probar_loopback.py`.
4. El resultado válido es **PASS**, RMS mayor que −60 dBFS y tono dominante cercano a 1 kHz. Un pico espectral aislado no demuestra que el tono llegó.

Si falla, no calibres. Revisá en Windows `mmsys.cpl` → **Grabación** → **Micrófono (USB PnP Sound Device)** → **Propiedades** → **Niveles**: entrada sin mute y nivel de grabación al 100 %. En **Opciones avanzadas**, elegí 48 kHz si está disponible. Verificá que el cable esté completamente insertado en los jacks del adaptador, no en los del portátil. No aumentes el volumen de salida para compensar una ruta de captura que no está verificada.

Para verificar la señal física del micrófono/TL072 (no el loopback), desconectá el cable verde→rosa, conecta la salida del circuito al canal izquierdo de la entrada rosa **con el acople, atenuación y protección ya validados**, y ejecutá:

```powershell
py verificar_entrada.py --duration 8
```

Al aparecer el mensaje de captura, habla al micrófono durante los 8 segundos. El programa solo graba desde la entrada; no reproduce ni guarda audio. Genera `verificacion/entrada.json` y `verificacion/entrada.svg`. PASS requiere RMS superior a −60 dBFS y ausencia de saturación. FAIL significa que la cadena todavía no está verificada: no calibres hasta resolverlo.

## 2. Calibrar la amplificación del TL072

La calibración eléctrica mide la respuesta del TL072, no el ambiente:

1. Desconectá el micrófono de la entrada del TL072.
2. Conectá la salida verde del PCM2902 a la entrada del TL072 con un capacitor de **1 µF** y una resistencia en serie de **2.2 kΩ**.
3. Dejá la salida del TL072 conectada a la entrada rosa del PCM2902, con **10 µF** y un divisor ajustable.
4. Cerrá la ventana de la app si está abierta: el PCM2902 no soporta dos streams full-duplex simultáneos.
5. Bajá el volumen de Windows a ~50 % y ejecutá:

```powershell
py calibrar_demo.py --amplitude 0.1
```

Criterios:

- Sin advertencia de saturación.
- Coherencia mediana > 0.8 entre 100 Hz y 10 kHz.
- RMS de cada captura mayor que −60 dBFS.
- Se guardan `medicion_H.npy` y `medicion_gamma2.npy`.

El calibrador ahora rechaza capturas con nivel bajo, saturación o coherencia insuficiente y no guarda archivos nuevos en esos casos. Si satura, usá `--amplitude 0.05` o bajá el volumen. Si la coherencia es baja, revisá conexiones y ruido.

6. Reconectá el micrófono al TL072.

> `medicion_H.npy` y `medicion_gamma2.npy` que ya existían son la medición **sintética** de `generar_medicion_demo.py` (muescas en 1 kHz y 4 kHz, coherencia 1.0). No sirven para el circuito real. `calibrar_demo.py` solo las reemplazará después de una captura con nivel y coherencia aceptables; si falla, no guarda una calibración nueva.

## 3. Escuchar los dos escenarios

```powershell
py iniciar_demo.py --h-response medicion_H.npy --coherence medicion_gamma2.npy
```

En la interfaz:

- **Interfaz:** “Sistema”.
- **Muestreo:** 48 kHz. **FFT:** 2048.
- **Operación:** arranca en **Ecualizado**.
- Hablá al micrófono y escuchá en los auriculares.
- Cambiá a **Bypass (sin corrección)**: escuchás la señal cruda del micrófono + TL072.
- Volvé a **Ecualizado**: escuchás la corrección.
- El espectro naranja es la entrada; el verde es la salida; la curva azul es el filtro inverso.

Ajustes si hace falta:

- Filtro agresivo o ruidoso: subí “Estabilidad (ε)” a 0.05–0.1 o bajá “Ganancia máx” a 6–9 dB.
- `CLIP`: bajá el nivel de entrada o la ganancia máxima.
- Acople/aullido: auriculares cerrados, volumen bajo, alejá el micrófono de los auriculares.

## 4. Seguridad

- No conectes la salida del TL072 directa a la entrada de micrófono del PCM2902: usá acople y atenuación.
- No conectes la salida del TL072 directa al ADC del ESP32. Si es single-supply de 5 V, divisor 10k/10k a GPIO34. Si es dual-supply (±12 V), usá clamp/level shifter.
- El ESP32 es solo un monitor; el audio USB lo hace el PCM2902.

## 5. Comandos rápidos

```powershell
cd C:\Users\juan_\Desktop\Projects\fc-filtro-inverso

# Demo que funciona hoy: micrófono + parlantes de la laptop
py iniciar_demo_laptop.py --check
py iniciar_demo_laptop.py --h-response verificacion\laptop\medicion_H.npy --coherence verificacion\laptop\medicion_gamma2.npy --max-gain-db 6

# Calibrar la cadena acústica de la laptop
py calibrar_demo_laptop.py --amplitude 0.05 --sweeps 4
```

Comandos del circuito (bloqueados hasta verificar PCM2902 y protoboard):

```powershell
# Abrir la interfaz con el par PCM2902
py iniciar_demo.py

# Calibrar con el par PCM2902
py calibrar_demo.py --amplitude 0.1

# Abrir con la calibración del circuito
py iniciar_demo.py --h-response medicion_H.npy --coherence medicion_gamma2.npy
```
