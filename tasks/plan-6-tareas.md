# Plan: cursor, login, grabaciones y graficas

Objetivo: entregar las 6 tareas pedidas sin tocar el callback de audio ni
`src/dsp`, con tests y un commit por tarea.

## Tareas

1. **Cursor compartido con dB**
   - `CursorReadout` mueve las 3 lineas y dibuja marcador + etiqueta por curva.
   - Etiqueta a la izquierda del punto y volteo a la derecha en la mitad derecha.
   - Soporta dB, dB SPL y bandas (`labels=`).
   - Al salir de las graficas se ocultan lineas, marcadores y etiquetas.
   - Tests en `tests/test_controls.py` y compatibilidad con el plot falso.

2. **Login, perfiles y dispositivos**
   - Logica pura en `src/audio/devices.py` y `src/storage/profiles.py`.
   - `LoginDialog` en `src/ui/login.py`; microfono manda la API, salida filtrada.
   - `perfiles.json` tolerante; recuerda usuario y dispositivos por nombre.
   - `main.py`: dialogo antes del motor, `--sin-login`, overrides de CLI y titulo.

3. **Grabaciones por usuario**
   - `sesiones/<nombre>_<documento>/<fecha_hora>/`.
   - `SessionRecorder(user=, devices=)` y metadata con usuario/dispositivos.
   - Export PNG/CSV por defecto en la carpeta del usuario.
   - Tests nuevos sin romper `test_recorder.py`.

4. **Sin huecos de ejes**
   - ViewBox sin padding, margenes 0-2 px y eje X desde
     `max(20 Hz, frecuencia del bin 1)`.
   - Mismo criterio en el espectrograma.
   - Actualizar tests que asumian 20 Hz exactos.

5. **Comparacion entrada/salida**
   - Sustituir "Respuesta del filtro" por `compare_plot` con entrada naranja y
     salida verde superpuestas en dB, eje X enlazado y cursor.
   - Conservar `_measured_db/_inverse_db/_product_db` para el CSV.
   - Actualizar tests de curvas y nombres.

6. **Titulos alineados**
   - `AXIS_WIDTH = 66` en los 3 plots; eje derecho vacio donde no se usa.
   - QSS: `margin-top: 16px`, `padding-top: 14px`, spacing raiz 8 px.
   - Verificacion visual con captura.

## Verificacion

- `py -m pytest` completo (fallos previos de fisica/SPL/barras se reportan).
- Capturas con `--sin-login` y con login.
