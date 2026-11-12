# Mapeo de la protoboard: qué está confirmado y qué no

Este archivo registra **solo lo que el usuario reportó** sobre el prototipo.
No es un esquema eléctrico verificado y no debe usarse para conectar audio a
ciegas.

## Confirmado en Windows y en las pruebas

- Windows reconoce un **USB PnP Sound Device** con `VID_08BB&PID_2902`: es la
  interfaz PCM2902.
- Endpoints observados: **Micrófono (USB PnP Sound Device)** como entrada y
  **Altavoces (USB PnP Sound Device)** como salida.
- La entrada acepta 48 kHz; el nivel de Windows estuvo al 100 %, sin mute.
- La captura más reciente de la entrada dio **45,10 dB**; no se demostró
  una señal útil.
- El loopback verde→rosa falló: no apareció la componente de 1 kHz esperada.
- Windows no mostró **COM5** ni un CP2102. Los COM visibles eran COM3 y COM4,
  ambos Bluetooth. El segundo USB no está identificado con certeza como la
  conexión de datos del ESP32.

## Reportado por el usuario sobre la protoboard

### Conector llamado “altavoz”

Tiene cables marcados `GND`, `V+` y `Dot`.

- `GND` llega al riel negativo, cerca del extremo inferior izquierdo de la
  protoboard (vista desde arriba).
- `V+` llega al riel positivo, junto a ese `GND`.
- `Dot` llega a **A2**.

> No sabemos qué módulo es ni qué significa `Dot`. No asumir que `V+` es la
> alimentación del TL072 ni que `Dot` es una señal de audio.

### Módulo llamado “chip USB de datos”

- Tiene `3V3`, negativo/GND y dos cables hacia la parte inferior derecha de la
  protoboard.
- `UP` está conectado a **27a**.
- `GND` está a la izquierda de `3V3`.

### Chip negro de ocho patas

- Ocupa las filas **14e–17e** y **14f–17f**, cruzando la ranura central.
- La muesca semicircular apunta hacia las filas **13–12**.

> La forma y la muesca **no confirman** que sea un TL072. Hay que leer la
> referencia impresa en el encapsulado y compararla con el datasheet.

### Capacitor de 10 µF

- Pata larga (positiva, si es electrolítico) en **e19**.
- Pata corta (negativa) en **e22**.

> No está identificado qué lado del capacitor va a la señal, al divisor o a
> una polarización. La posición de las patas no alcanza para decidirlo.

### Conector de audio hacia la PC

- Tiene dos cables, uno **rojo** y uno **verde**.
- Ambos están **sin conexión asignada**.
- El rojo sale por debajo de una cajita; el verde, por un costado.

> El color y el recorrido no indican si son señal o tierra. Hay que trazar
> cada cable hasta punta, anillo y cuerpo del conector, y verificar
> continuidad.

## Lo que todavía no está identificado

- Qué módulo es el de `GND/V+/Dot` y qué significa `Dot`.
- Si el chip negro es realmente un **TL072** y cuál es su orientación física
  verificada.
- Si el riel llamado “negativo” es **GND** o una alimentación negativa.
- Las filas exactas de las resistencias del divisor y cómo se conectan con el
  capacitor.
- A qué contacto del conector de audio llegan el cable rojo y el verde.
- La polaridad y el valor real del capacitor (el usuario reportó 10 µF).
- Si el módulo “USB de datos” es un CP2102, un CH340 u otro, y si tiene datos
  conectados.

## Consecuencia para la conexión de audio

La ruta prevista es:

```text
Salida del circuito → acople y atenuación/protección → entrada rosa del PCM2902
Tierra de audio ─────────────────────────────────────→ GND común verificado
```

Con los datos actuales **no se puede señalar responsablemente una fila exacta
para conectar la señal**. `e19` y `e22` solo describen dónde están las patas
del capacitor; no dicen cuál lado corresponde a la salida del circuito o al
divisor.

## Corrección importante

Una numeración de patas anterior supuso que el chip era un **TL072 DIP-8**.
Esa numeración solo vale si se confirma el modelo y la orientación física; no
basta para decidir qué cable va a cada pata o fila.

## Seguridad

- Hubo un reporte previo de olor a quemado; después el prototipo se reemplazó
  o inspeccionó. Si reaparece olor a quemado o calentamiento anormal,
  desconectá y detené las pruebas.
- No conectes la protoboard al PCM2902 ni al ESP32 hasta identificar el chip,
  el GND real y el divisor.
- No hay multímetro disponible actualmente. Sin medición de continuidad y
  voltajes, este mapeo queda **bloqueado**.

## Qué medir cuando haya multímetro

1. Continuidad entre el riel “negativo” y el pin GND del conector USB.
2. Voltaje DC de cada riel respecto de ese GND.
3. Referencia impresa del chip de ocho patas y continuidad de cada pin a los
   rieles y a las resistencias.
4. Valor y conexión de las resistencias del divisor.
5. Polaridad y continuidad del capacitor de 10 µF.
6. Punta, anillo y cuerpo del conector de audio, y qué cable llega a cada uno.
7. Continuidad de `UP`, `3V3` y GND del módulo “USB de datos” hasta el chip.
