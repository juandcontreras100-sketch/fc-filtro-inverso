import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────────────────────
# Helpers de validación defensiva (Requerimiento 1)
# ──────────────────────────────────────────────────────────────
def _validate_inputs(
    h_freq: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    *,
    func_name: str,
) -> None:
    """
    Validaciones de cordura (sanity checks) para las funciones públicas.

    Lanza ValueError si:
    - h_freq y coherence no tienen el mismo shape.
    - Algún arreglo contiene NaN o Inf.
    - coherence tiene valores fuera de [0, 1].
    """
    if h_freq.shape != coherence.shape:
        raise ValueError(
            f"{func_name}: h_freq y coherence deben tener la misma forma; "
            f"recibido h_freq.shape={h_freq.shape}, coherence.shape={coherence.shape}"
        )

    if not np.all(np.isfinite(h_freq)):
        raise ValueError(f"{func_name}: h_freq contiene valores NaN o Inf")

    if not np.all(np.isfinite(coherence)):
        raise ValueError(f"{func_name}: coherence contiene valores NaN o Inf")

    if np.any(coherence < 0.0) or np.any(coherence > 1.0):
        raise ValueError(
            f"{func_name}: coherence debe estar en el rango físico [0, 1]; "
            f"encontrado min={np.min(coherence):.3f}, max={np.max(coherence):.3f}"
        )


def _safe_peak_power(h_freq: NDArray[np.complexfloating]) -> float:
    """
    Calcula max(|H|²) protegiendo contra underflow / señal nula (Requerimiento 2).

    Si la señal es identicamente cero (o bajo epsilon de máquina), devuelve
    np.finfo(float).eps para evitar división por cero en pasos posteriores.
    """
    peak = float(np.max(np.abs(h_freq) ** 2))
    eps_mach = np.finfo(float).eps
    return peak if peak > eps_mach else eps_mach


# ──────────────────────────────────────────────────────────────
# Funciones públicas originales — ahora robustecidas
# ──────────────────────────────────────────────────────────────
def epsilon_profile(
    h_freq: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    epsilon_base: float,
    k: float = 5.0,
) -> NDArray[np.floating]:
    """
    Perfil de regularización Tikhonov dependiente de frecuencia.

    epsilon(f) = epsilon_base * max(|H(f)|²) * (1 + k * (1 - gamma²(f)))

    Con coherencia perfecta (gamma² = 1) en toda la banda, se reduce a una
    fracción constante del pico de |H(f)|².

    Parámetros
    ----------
    h_freq : NDArray[complex]
        Respuesta en frecuencia compleja H(f) de forma (N,).
    coherence : NDArray[float]
        Coherencia gamma²(f) ∈ [0, 1] de forma (N,).
    epsilon_base : float
        Fracción base del pico de potencia (adimensional, típicamente 1e-4 – 1e-2).
    k : float, opcional
        Factor de penalización por baja coherencia (default 5.0).

    Retorna
    -------
    NDArray[float]
        epsilon(f) por bin de frecuencia, forma (N,).
    """
    _validate_inputs(h_freq, coherence, func_name="epsilon_profile")

    peak_power = _safe_peak_power(h_freq)
    return epsilon_base * peak_power * (1.0 + k * (1.0 - coherence))


def estimate_h_inverse(
    h_freq: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    epsilon_base: float,
    max_gain_db: float,
) -> NDArray[np.complexfloating]:
    """
    Filtro inverso regularizado de Wiener/Tikhonov con límite de ganancia.

        H_inv(f) = conj(H(f)) / (|H(f)|² + epsilon(f))

    Parámetros
    ----------
    h_freq : NDArray[complex]
        Respuesta en frecuencia compleja H(f).
    coherence : NDArray[float]
        Coherencia gamma²(f) ∈ [0, 1].
    epsilon_base : float
        Fracción base para regularización Tikhonov.
    max_gain_db : float
        Techo de ganancia en dB (ej. 12 dB → factor 4× lineal).

    Retorna
    -------
    NDArray[complex]
        Filtro inverso regularizado H_inv(f), forma (N,).
    """
    _validate_inputs(h_freq, coherence, func_name="estimate_h_inverse")

    eps = epsilon_profile(h_freq, coherence, epsilon_base)

    # Denominador protegido: |H|² + eps  (eps ≥ epsilon_base * eps_mach > 0)
    denom = np.abs(h_freq) ** 2 + eps
    h_inv = np.conj(h_freq) / denom

    # Límite duro de ganancia (clipping posterior, desacoplado de la regularización)
    max_allowed = 10 ** (max_gain_db / 20)
    magnitude = np.abs(h_inv)
    scale = np.ones_like(magnitude)
    nonzero = magnitude > 0
    scale[nonzero] = np.minimum(1.0, max_allowed / magnitude[nonzero])
    return h_inv * scale


# ──────────────────────────────────────────────────────────────
# NUEVAS FUNCIONES: Métricas de estabilidad (Requerimiento 3)
# ──────────────────────────────────────────────────────────────
def calculate_condition_number(
    h_freq: NDArray[np.complexfloating],
    epsilon: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Número de condición efectivo por bin de frecuencia.

    Para la inversión regularizada diagonal (|H|² + ε)⁻¹, el número de condición
    efectivo es la razón entre el valor singular máximo y el mínimo del sistema
    regularizado:

        κ_eff(f) = (|H(f)|² + ε(f)) / ε(f)

    Valores grandes indican bins donde la inversión es sensible a perturbaciones
    (muescas profundas, baja coherencia). Valores ≈ 1 indican inversión bien
    condicionada.

    Parámetros
    ----------
    h_freq : NDArray[complex]
        Respuesta en frecuencia H(f).
    epsilon : NDArray[float]
        Perfil de regularización epsilon(f) (salida de epsilon_profile).

    Retorna
    -------
    NDArray[float]
        κ_eff(f) por bin, forma (N,).
    """
    _validate_inputs(h_freq, epsilon, func_name="calculate_condition_number")
    power = np.abs(h_freq) ** 2
    return (power + epsilon) / epsilon


def estimate_error_bounds(
    h_freq: NDArray[np.complexfloating],
    h_inv: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """
    Error del residuo numérico (error a posteriori).

    Mide qué tan bien H_inv aproxima la inversa verdadera punto a punto:

        error(f) = |H(f) * H_inv(f) - 1|

    En bins con regularización fuerte (|H|² ≪ ε), el error tiende a 1
    (el filtro no corrige). En bins bien condicionados, el error tiende a 0.

    Parámetros
    ----------
    h_freq : NDArray[complex]
        Respuesta en frecuencia original H(f).
    h_inv : NDArray[complex]
        Filtro inverso estimado (salida de estimate_h_inverse).

    Retorna
    -------
    NDArray[float]
        Error absoluto del residuo por bin, forma (N,).
    """
    # Validación específica para esta función: h_inv NO es coherence
    if h_freq.shape != h_inv.shape:
        raise ValueError(
            f"estimate_error_bounds: h_freq y h_inv deben tener la misma forma; "
            f"recibido h_freq.shape={h_freq.shape}, h_inv.shape={h_inv.shape}"
        )
    if not np.all(np.isfinite(h_freq)):
        raise ValueError("estimate_error_bounds: h_freq contiene valores NaN o Inf")
    if not np.all(np.isfinite(h_inv)):
        raise ValueError("estimate_error_bounds: h_inv contiene valores NaN o Inf")

    residual = h_freq * h_inv - 1.0
    return np.abs(residual)


# ──────────────────────────────────────────────────────────────
# MÉTODOS NUMÉRICOS ITERATIVOS (Unidades 2 y 3 del Syllabus)
# ──────────────────────────────────────────────────────────────

def estimate_h_inverse_gradient(
    h_freq: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    epsilon_base: float,
    iterations: int,
    learning_rate: float,
) -> NDArray[np.complexfloating]:
    """
    Filtro inverso vía descenso del gradiente (Método del Gradiente - Unidad 3).

    Minimiza la función de costo regularizada (Tikhonov):
        J(H_inv) = |H · H_inv - 1|² + ε · |H_inv|²

    El gradiente analítico respecto a H_inv* (derivada de Wirtinger) es:
        ∇J = H* · (H · H_inv - 1) + ε · H_inv

    Actualización iterativa:
        H_inv^{k+1} = H_inv^k - α · ∇J(H_inv^k)

    Parámetros
    ----------
    h_freq : NDArray[complex]
        Respuesta en frecuencia compleja H(f) de forma (N,).
    coherence : NDArray[float]
        Coherencia gamma²(f) ∈ [0, 1] de forma (N,).
    epsilon_base : float
        Fracción base para regularización Tikhonov.
    iterations : int
        Número de iteraciones de descenso del gradiente.
    learning_rate : float
        Tasa de aprendizaje α (step size). Debe ser lo suficientemente
        pequeña para garantizar convergencia (ej. α < 2 / max(|H|² + ε)).

    Retorna
    -------
    NDArray[complex]
        Aproximación numérica de H_inv(f) tras 'iterations' iteraciones.
    """
    _validate_inputs(h_freq, coherence, func_name="estimate_h_inverse_gradient")

    if iterations <= 0:
        raise ValueError("estimate_h_inverse_gradient: iterations debe ser > 0")
    if learning_rate <= 0:
        raise ValueError("estimate_h_inverse_gradient: learning_rate debe ser > 0")

    eps = epsilon_profile(h_freq, coherence, epsilon_base)

    # Inicialización: solución de Wiener/Tikhonov como punto de partida (acelera convergencia)
    denom = np.abs(h_freq) ** 2 + eps
    h_inv = np.conj(h_freq) / denom

    # Descenso del gradiente
    for _ in range(iterations):
        # Residual: H * H_inv - 1
        residual = h_freq * h_inv - 1.0
        # Gradiente: conj(H) * residual + eps * H_inv
        grad = np.conj(h_freq) * residual + eps * h_inv
        # Actualización
        h_inv = h_inv - learning_rate * grad

    return h_inv


def calculate_group_delay(
    h_inv: NDArray[np.complexfloating],
    delta_f: float,
) -> NDArray[np.floating]:
    """
    Retardo de grupo mediante diferenciación numérica a 3 puntos (Unidad 2).

    El retardo de grupo es la derivada negativa de la fase no envuelta:
        τ_g(f) = -dφ/dω = -1/(2π) · dφ/df

    Se usa el método de diferencias centrales (orden 2, 3 puntos):
        f'(x_i) ≈ (f(x_{i+1}) - f(x_{i-1})) / (2h)    para puntos interiores
    Para los bordes se usan diferencias hacia adelante/atrás de 2do orden:
        f'(x_0) ≈ (-3f_0 + 4f_1 - f_2) / (2h)
        f'(x_{N-1}) ≈ (3f_{N-1} - 4f_{N-2} + f_{N-3}) / (2h)

    Parámetros
    ----------
    h_inv : NDArray[complex]
        Filtro inverso complejo H_inv(f) de forma (N,), N ≥ 3.
    delta_f : float
        Espaciado uniforme en frecuencia Δf [Hz] entre bins adyacentes.

    Retorna
    -------
    NDArray[float]
        Retardo de grupo τ_g(f) [segundos] por bin, forma (N,).
    """
    if h_inv.ndim != 1:
        raise ValueError("calculate_group_delay: h_inv debe ser array 1-D")
    n = h_inv.shape[0]
    if n < 3:
        raise ValueError("calculate_group_delay: se requieren al menos 3 puntos para diferencias a 3 puntos")
    if delta_f <= 0:
        raise ValueError("calculate_group_delay: delta_f debe ser > 0")

    # Fase no envuelta
    phase = np.unwrap(np.angle(h_inv))

    # Derivada numérica a 3 puntos (diferencias centrales)
    dphi_df = np.empty_like(phase, dtype=float)
    h = delta_f
    two_h = 2.0 * h

    # Punto inicial: diferencia hacia adelante de 2do orden
    dphi_df[0] = (-3.0 * phase[0] + 4.0 * phase[1] - phase[2]) / two_h
    # Punto final: diferencia hacia atrás de 2do orden
    dphi_df[-1] = (3.0 * phase[-1] - 4.0 * phase[-2] + phase[-3]) / two_h
    # Puntos interiores: diferencias centrales
    dphi_df[1:-1] = (phase[2:] - phase[:-2]) / two_h

    # Retardo de grupo: τ_g = -1/(2π) * dφ/df
    group_delay = -dphi_df / (2.0 * np.pi)
    return group_delay


def smooth_coherence_spline(
    coherence: NDArray[np.floating],
    freq_bins: NDArray[np.floating],
    new_freq_bins: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Suavizado y upsampling de coherencia via interpolación por splines (Unidad 3).

    Utiliza splines cúbicos (C²-continuos) para interpolar gamma²(f) en una
    malla de frecuencias más fina, suavizando ruido de medición y evitando
    artefactos de escalón en el cálculo de ε(f).

    Parámetros
    ----------
    coherence : NDArray[float]
        Coherencia gamma²(f) original ∈ [0, 1] de forma (N,).
    freq_bins : NDArray[float]
        Bins de frecuencia originales correspondientes a 'coherence', forma (N,).
    new_freq_bins : NDArray[float]
        Nueva malla de frecuencia objetivo (más fina), forma (M,).

    Retorna
    -------
    NDArray[float]
        Coherencia interpolada y suavizada en 'new_freq_bins', forma (M,),
        clampeada al rango físico [0, 1].
    """
    if coherence.shape != freq_bins.shape:
        raise ValueError(
            "smooth_coherence_spline: coherence y freq_bins deben tener la misma forma"
        )
    if not np.all(np.isfinite(coherence)) or not np.all(np.isfinite(freq_bins)):
        raise ValueError("smooth_coherence_spline: entradas contienen NaN o Inf")
    if np.any(coherence < 0.0) or np.any(coherence > 1.0):
        raise ValueError("smooth_coherence_spline: coherence debe estar en [0, 1]")
    if not np.all(np.diff(freq_bins) > 0):
        raise ValueError("smooth_coherence_spline: freq_bins debe ser estrictamente creciente")
    if not np.all(np.diff(new_freq_bins) > 0):
        raise ValueError("smooth_coherence_spline: new_freq_bins debe ser estrictamente creciente")

    # Spline cúbico usando scipy si está disponible, fallback a implementación manual
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(freq_bins, coherence, bc_type='natural')
        coherence_smooth = cs(new_freq_bins)
    except ImportError:
        # Implementación manual de spline cúbico natural (tridiagonal)
        coherence_smooth = _cubic_spline_interpolate(freq_bins, coherence, new_freq_bins)

    # Clampear a rango físico [0, 1]
    return np.clip(coherence_smooth, 0.0, 1.0)


def _cubic_spline_interpolate(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    x_new: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Interpolación spline cúbico natural (resolución tridiagonal tipo Gauss-Jordán - Unidad 3).

    Resuelve el sistema tridiagonal para segundas derivadas M_i:
        h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1} = 6 * ((y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1})

    Con condiciones de frontera naturales: M_0 = M_{n-1} = 0
    """
    n = len(x)
    h = np.diff(x)  # h_i = x_{i+1} - x_i

    # Sistema tridiagonal A * M = b
    # A es (n-2) x (n-2) para incógnitas M_1 ... M_{n-2}
    if n == 2:
        # Caso degenerado: interpolación lineal
        return np.interp(x_new, x, y)

    # Diagonales de la matriz tridiagonal
    a = h[:-1]           # sub-diagonal: h_{i-1} para i=1..n-2
    b = 2.0 * (h[:-1] + h[1:])  # diagonal principal: 2(h_{i-1}+h_i)
    c = h[1:]            # super-diagonal: h_i para i=1..n-2

    # Lado derecho
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    # Resolución tridiagonal (algoritmo de Thomas - O(n))
    # Forward elimination
    cp = c.copy()
    rp = rhs.copy()
    cp[0] = cp[0] / b[0]
    rp[0] = rp[0] / b[0]
    for i in range(1, n - 2):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = cp[i] / denom
        rp[i] = (rp[i] - a[i - 1] * rp[i - 1]) / denom

    # Back substitution
    M_inner = np.zeros(n - 2)
    M_inner[-1] = rp[-1]
    for i in range(n - 4, -1, -1):
        M_inner[i] = rp[i] - cp[i] * M_inner[i + 1]

    # Segundas derivadas completas (M_0 = M_{n-1} = 0 para spline natural)
    M = np.zeros(n)
    M[1:-1] = M_inner

    # Evaluar spline en x_new
    y_new = np.zeros_like(x_new)
    for j, xj in enumerate(x_new):
        # Encontrar intervalo [x_i, x_{i+1}] que contiene xj
        i = np.searchsorted(x, xj) - 1
        i = np.clip(i, 0, n - 2)
        hi = h[i]
        t = (xj - x[i]) / hi
        # Fórmula del spline cúbico
        y_new[j] = (
            (1 - t) * y[i] + t * y[i + 1]
            + hi**2 / 6.0 * (
                (1 - t) * ((1 - t)**2 - 1) * M[i]
                + t * (t**2 - 1) * M[i + 1]
            )
        )

    return y_new


# ──────────────────────────────────────────────────────────────
# OPERACIONES BÁSICAS: Integración y Búsqueda de Raíces (Unidad 2)
# ──────────────────────────────────────────────────────────────

def calculate_filter_energy(
    h_inv: NDArray[np.complexfloating],
    delta_f: float,
) -> float:
    """
    Energía total del filtro mediante integración numérica (Regla de Simpson 1/3 - Unidad 2).

    La energía es la integral de la magnitud al cuadrado:
        E = ∫ |H_inv(f)|² df

    Método: Regla de Simpson 1/3 compuesta para malla uniforme.
    Para n puntos (n-1 segmentos):
    - Si n es impar (n-1 par): Simpson 1/3 en todos los segmentos
    - Si n es par (n-1 impar): Simpson 1/3 hasta n-2, Trapecio en el último segmento

    Fórmula de Simpson 1/3 (3 puntos, paso h):
        ∫_{x_0}^{x_2} f(x) dx ≈ (h/3) * [f_0 + 4f_1 + f_2]

    Fórmula del Trapecio (2 puntos, paso h):
        ∫_{x_0}^{x_1} f(x) dx ≈ (h/2) * [f_0 + f_1]

    Parámetros
    ----------
    h_inv : NDArray[complex]
        Filtro inverso complejo H_inv(f) de forma (N,), N ≥ 3.
    delta_f : float
        Espaciado uniforme en frecuencia Δf [Hz] entre bins adyacentes.

    Retorna
    -------
    float
        Energía total E [unidad²·Hz] (escalar).
    """
    if h_inv.ndim != 1:
        raise ValueError("calculate_filter_energy: h_inv debe ser array 1-D")
    n = h_inv.shape[0]
    if n < 3:
        raise ValueError("calculate_filter_energy: se requieren al menos 3 puntos para Simpson")
    if delta_f <= 0:
        raise ValueError("calculate_filter_energy: delta_f debe ser > 0")

    # Integrando: |H_inv|²
    y = np.abs(h_inv) ** 2
    h = delta_f

    if n % 2 == 1:
        # n impar → n-1 segmentos par → Simpson 1/3 compuesto completo
        # S = (h/3) * [y_0 + y_{n-1} + 4*Σ(y_impares) + 2*Σ(y_pares interiores)]
        integral = (h / 3.0) * (
            y[0] + y[-1]
            + 4.0 * np.sum(y[1:-1:2])
            + 2.0 * np.sum(y[2:-2:2])
        )
    else:
        # n par → n-1 segmentos impar → Simpson 1/3 hasta n-2, Trapecio final
        # Simpson en los primeros n-1 puntos (índices 0 a n-2)
        integral_simpson = (h / 3.0) * (
            y[0] + y[-2]
            + 4.0 * np.sum(y[1:-2:2])
            + 2.0 * np.sum(y[2:-3:2])
        )
        # Trapecio en el último segmento [n-2, n-1]
        integral_trapezoid = (h / 2.0) * (y[-2] + y[-1])
        integral = integral_simpson + integral_trapezoid

    return float(integral)


def find_cutoff_frequency(
    freqs: NDArray[np.floating],
    magnitude_response: NDArray[np.floating],
    target_magnitude: float,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """
    Frecuencia de corte mediante Método de Bisección (Unidad 2).

    Encuentra la frecuencia f donde |H_inv(f)| = target_magnitude,
    resolviendo g(f) = |H_inv(f)| - target_magnitude = 0.

    Dado que los datos son discretos, se interpola linealmente entre bins
    para evaluar g(f) en cualquier f continua dentro del rango.

    Algoritmo de Bisección:
    1. Requiere un intervalo [a, b] donde g(a) * g(b) < 0 (cambio de signo)
    2. Punto medio c = (a + b) / 2
    3. Si |g(c)| < tol o (b - a) < tol → convergencia
    4. Si g(a) * g(c) < 0 → b = c, en caso contrario a = c
    5. Repetir hasta convergencia o max_iter

    Parámetros
    ----------
    freqs : NDArray[float]
        Bins de frecuencia (estrictamente creciente), forma (N,).
    magnitude_response : NDArray[float]
        Magnitud |H_inv(f)| correspondiente a freqs, forma (N,).
    target_magnitude : float
        Valor objetivo de magnitud (ej. 1/sqrt(2) ≈ 0.707 para -3 dB).
    tol : float, opcional
        Tolerancia de convergencia en frecuencia [Hz] (default 1e-5).
    max_iter : int, opcional
        Máximo número de iteraciones (default 100).

    Retorna
    -------
    float
        Frecuencia de corte f_c [Hz] donde |H_inv(f_c)| = target_magnitude.

    Lanza
    -----
    ValueError
        Si no hay cambio de signo en el rango, o si target_magnitude está
        fuera del rango de magnitude_response.
    """
    if freqs.shape != magnitude_response.shape:
        raise ValueError(
            "find_cutoff_frequency: freqs y magnitude_response deben tener la misma forma"
        )
    if freqs.ndim != 1 or magnitude_response.ndim != 1:
        raise ValueError("find_cutoff_frequency: entradas deben ser arrays 1-D")
    n = freqs.shape[0]
    if n < 2:
        raise ValueError("find_cutoff_frequency: se requieren al menos 2 puntos")
    if not np.all(np.diff(freqs) > 0):
        raise ValueError("find_cutoff_frequency: freqs debe ser estrictamente creciente")
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(magnitude_response)):
        raise ValueError("find_cutoff_frequency: entradas contienen NaN o Inf")
    if tol <= 0:
        raise ValueError("find_cutoff_frequency: tol debe ser > 0")
    if max_iter <= 0:
        raise ValueError("find_cutoff_frequency: max_iter debe ser > 0")

    # Función auxiliar: interpolación lineal de magnitude_response en f continua
    def g(f: float) -> float:
        # np.interp hace interpolación lineal y extrapolación constante en bordes
        # Limitamos al rango válido
        f_clipped = np.clip(f, freqs[0], freqs[-1])
        mag_interp = np.interp(f_clipped, freqs, magnitude_response)
        return mag_interp - target_magnitude

    # Verificar que target_magnitude está en el rango de la respuesta
    min_mag = np.min(magnitude_response)
    max_mag = np.max(magnitude_response)
    if target_magnitude < min_mag or target_magnitude > max_mag:
        raise ValueError(
            f"find_cutoff_frequency: target_magnitude={target_magnitude:.6f} "
            f"fuera del rango de la respuesta [{min_mag:.6f}, {max_mag:.6f}]"
        )

    # Buscar intervalo [a, b] con cambio de signo
    # Recorremos los segmentos discretos buscando cruce
    a = b = None
    for i in range(n - 1):
        g_i = magnitude_response[i] - target_magnitude
        g_ip1 = magnitude_response[i + 1] - target_magnitude
        if g_i == 0.0:
            return float(freqs[i])
        if g_i * g_ip1 < 0:
            a = float(freqs[i])
            b = float(freqs[i + 1])
            break

    if a is None or b is None:
        # Si no hay cruce estricto, puede ser que target_magnitude == max o min
        # y ocurra en un extremo. Devolvemos la frecuencia correspondiente.
        if abs(max_mag - target_magnitude) < abs(min_mag - target_magnitude):
            idx = np.argmax(magnitude_response)
        else:
            idx = np.argmin(magnitude_response)
        return float(freqs[idx])

    # Bisección
    fa = g(a)
    fb = g(b)
    if fa * fb > 0:
        # Por seguridad numérica (aunque ya verificamos cambio de signo)
        raise ValueError("find_cutoff_frequency: no hay cambio de signo válido en el intervalo")

    for _ in range(max_iter):
        c = (a + b) / 2.0
        fc = g(c)

        if abs(fc) < tol or (b - a) < tol:
            return float(c)

        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    # Si no convergió en max_iter, devolver mejor estimación
    return float((a + b) / 2.0)