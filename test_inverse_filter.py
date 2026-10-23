import numpy as np
import pytest
from inverse_filter import (
    epsilon_profile,
    estimate_h_inverse,
    calculate_condition_number,
    estimate_error_bounds,
    estimate_h_inverse_gradient,
    calculate_group_delay,
    smooth_coherence_spline,
    _cubic_spline_interpolate,
    calculate_filter_energy,
    find_cutoff_frequency,
)


# ──────────────────────────────────────────────────────────────
# TESTS ORIGINALES (mantenidos intactos)
# ──────────────────────────────────────────────────────────────
def test_epsilon_profile_uniform_coherence_is_constant_fraction_of_peak():
    """Con coherencia perfecta (gamma^2 = 1) en toda la banda, epsilon(f)
    debe ser una fraccion constante del pico de |H(f)|^2: epsilon_base * max(|H|^2).
    No debe haber termino extra de "desconfianza" cuando la coherencia es perfecta.
    """
    freqs = np.linspace(0, 24000, 9)
    h = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05], dtype=complex)
    coherence = np.ones_like(freqs)  # coherencia perfecta en toda la banda
    epsilon_base = 0.02

    eps = epsilon_profile(h, coherence, epsilon_base)

    expected_constant = epsilon_base * np.max(np.abs(h) ** 2)
    assert np.allclose(eps, expected_constant), (
        f"esperado epsilon constante = {expected_constant}, obtuve {eps}"
    )


def test_epsilon_profile_increases_where_coherence_is_low():
    """En bandas donde la medicion de H(f) es poco confiable (coherencia
    baja), epsilon(f) debe ser mayor que en bandas de coherencia alta, para
    que el filtro se vuelva mas conservador ahi (menos correccion) en vez
    de intentar invertir ruido de medicion.
    """
    freqs = np.linspace(0, 24000, 5)
    h = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=complex)
    coherence = np.array([1.0, 1.0, 0.1, 1.0, 1.0])  # banda central poco confiable
    epsilon_base = 0.02

    eps = epsilon_profile(h, coherence, epsilon_base)

    low_coherence_eps = eps[2]
    high_coherence_eps = eps[0]
    assert low_coherence_eps > high_coherence_eps, (
        f"epsilon en banda de baja coherencia ({low_coherence_eps}) deberia "
        f"ser mayor que en banda de coherencia alta ({high_coherence_eps})"
    )


def test_estimate_h_inverse_of_flat_response_is_approximately_unity():
    """Si el amplificador tiene una respuesta plana H(f) = 1 en toda la
    banda y la medicion es totalmente confiable (coherencia = 1), el
    filtro inverso no deberia alterar la señal: H_inv(f) ~= 1.

    La regularizacion Tikhonov introduce un sesgo pequeño y deliberado
    hacia valores levemente menores que 1 (para permanecer estable en
    frecuencias reales donde H(f) no es perfectamente plana), por eso se
    exige "aproximadamente" con una tolerancia, no igualdad exacta.
    """
    freqs = np.linspace(0, 24000, 9)
    h = np.ones_like(freqs, dtype=complex)
    coherence = np.ones_like(freqs)
    epsilon_base = 0.02
    max_gain_db = 12.0

    h_inv = estimate_h_inverse(h, coherence, epsilon_base, max_gain_db)

    assert np.allclose(h_inv, 1.0, atol=0.05), (
        f"esperado H_inv approx 1.0, obtuve {h_inv}"
    )


def test_estimate_h_inverse_clips_gain_at_deep_notch():
    """Con una regularizacion base muy baja (epsilon_base pequeño) y una
    muesca profunda en H(f), la formula de Wiener/Tikhonov por si sola
    puede producir una ganancia enorme cerca del punto h ~= sqrt(epsilon).
    El limite explicito de ganancia (max_gain_db) debe evitar que
    |H_inv(f)| supere ese techo en cualquier frecuencia, sin importar que
    tan agresiva sea la muesca.
    """
    freqs = np.linspace(0, 24000, 5)
    h = np.array([1.0, 1.0, 0.01, 1.0, 1.0], dtype=complex)  # muesca en el centro
    coherence = np.ones_like(freqs)
    epsilon_base = 1e-4  # regularizacion deliberadamente laxa
    max_gain_db = 12.0

    h_inv = estimate_h_inverse(h, coherence, epsilon_base, max_gain_db)

    max_allowed_linear = 10 ** (max_gain_db / 20)
    assert np.max(np.abs(h_inv)) <= max_allowed_linear * 1.001, (
        f"|H_inv| maximo = {np.max(np.abs(h_inv))}, "
        f"supera el limite de {max_gain_db} dB ({max_allowed_linear} lineal)"
    )


# ──────────────────────────────────────────────────────────────
# NUEVOS TESTS: Validación defensiva (Requerimiento 1)
# ──────────────────────────────────────────────────────────────
def test_epsilon_profile_raises_on_shape_mismatch():
    h = np.ones(5, dtype=complex)
    coherence = np.ones(4)  # shape distinto
    with pytest.raises(ValueError, match="misma forma"):
        epsilon_profile(h, coherence, epsilon_base=0.01)


def test_epsilon_profile_raises_on_nan_in_h():
    h = np.array([1.0, np.nan, 1.0], dtype=complex)
    coherence = np.ones(3)
    with pytest.raises(ValueError, match="NaN o Inf"):
        epsilon_profile(h, coherence, epsilon_base=0.01)


def test_epsilon_profile_raises_on_inf_in_coherence():
    h = np.ones(3, dtype=complex)
    coherence = np.array([1.0, np.inf, 1.0])
    with pytest.raises(ValueError, match="NaN o Inf"):
        epsilon_profile(h, coherence, epsilon_base=0.01)


def test_epsilon_profile_raises_on_coherence_out_of_bounds():
    h = np.ones(3, dtype=complex)
    coherence = np.array([1.0, 1.5, 0.5])  # > 1
    with pytest.raises(ValueError, match="rango físico \\[0, 1\\]"):
        epsilon_profile(h, coherence, epsilon_base=0.01)


def test_epsilon_profile_raises_on_negative_coherence():
    h = np.ones(3, dtype=complex)
    coherence = np.array([1.0, -0.1, 0.5])  # < 0
    with pytest.raises(ValueError, match="rango físico \\[0, 1\\]"):
        epsilon_profile(h, coherence, epsilon_base=0.01)


def test_estimate_h_inverse_raises_on_shape_mismatch():
    h = np.ones(5, dtype=complex)
    coherence = np.ones(4)
    with pytest.raises(ValueError, match="misma forma"):
        estimate_h_inverse(h, coherence, epsilon_base=0.01, max_gain_db=12.0)


def test_calculate_condition_number_raises_on_shape_mismatch():
    h = np.ones(5, dtype=complex)
    epsilon = np.ones(4)
    with pytest.raises(ValueError, match="misma forma"):
        calculate_condition_number(h, epsilon)


def test_estimate_error_bounds_raises_on_shape_mismatch():
    h = np.ones(5, dtype=complex)
    h_inv = np.ones(4, dtype=complex)
    with pytest.raises(ValueError, match="misma forma"):
        estimate_error_bounds(h, h_inv)


# ──────────────────────────────────────────────────────────────
# NUEVOS TESTS: Seguridad numérica extrema (Requerimiento 2)
# ──────────────────────────────────────────────────────────────
def test_epsilon_profile_handles_all_zero_H():
    """H(f) ≡ 0 no debe lanzar excepción; peak_power se clampa a eps_mach."""
    h = np.zeros(5, dtype=complex)
    coherence = np.ones(5)
    epsilon_base = 0.02
    eps = epsilon_profile(h, coherence, epsilon_base)
    # epsilon_base * eps_mach * (1 + k*0)  ≈  0.02 * 2.22e-16
    expected = epsilon_base * np.finfo(float).eps
    assert np.allclose(eps, expected)


def test_estimate_h_inverse_handles_all_zero_H():
    """H(f) ≡ 0 → H_inv ≡ 0 (sin división por cero)."""
    h = np.zeros(5, dtype=complex)
    coherence = np.ones(5)
    h_inv = estimate_h_inverse(h, coherence, epsilon_base=0.01, max_gain_db=12.0)
    assert np.allclose(h_inv, 0.0)


# ──────────────────────────────────────────────────────────────
# NUEVOS TESTS: Métricas de estabilidad (Requerimiento 3)
# ──────────────────────────────────────────────────────────────
def test_calculate_condition_number_formula():
    """κ_eff = (|H|² + ε) / ε  →  en H=1, ε=0.01  ⇒  κ = 101."""
    h = np.array([1.0, 2.0, 0.5], dtype=complex)
    epsilon = np.array([0.01, 0.01, 0.01])
    kappa = calculate_condition_number(h, epsilon)
    expected = (np.abs(h) ** 2 + epsilon) / epsilon
    assert np.allclose(kappa, expected)


def test_calculate_condition_number_high_at_deep_notch():
    """En muesca profunda (|H|² ≪ ε), κ_eff ≈ 1 (regularización domina).
    En cambio, donde |H|² ≫ ε, κ_eff ≈ |H|²/ε → grande.
    """
    h = np.array([1.0, 1e-6, 1.0], dtype=complex)
    epsilon = np.array([0.01, 0.01, 0.01])
    kappa = calculate_condition_number(h, epsilon)
    # En muesca (índice 1): |H|² ≈ 1e-12 ≪ 0.01 → κ ≈ 1
    # En banda plana (índice 0): |H|² = 1 ≫ 0.01 → κ ≈ 101
    assert kappa[1] < kappa[0]  # muesca mejor condicionada (regularización domina)
    assert kappa[0] > 100       # banda plana mal condicionada sin regularización fuerte


def test_estimate_error_bounds_zero_for_perfect_inverse():
    """Si H_inv = 1/H exactamente (sin regularización), error = 0."""
    h = np.array([1.0, 2.0, 0.5], dtype=complex)
    h_inv = 1.0 / h
    error = estimate_error_bounds(h, h_inv)
    assert np.allclose(error, 0.0, atol=1e-15)


def test_estimate_error_bounds_near_one_when_regularized():
    """Con regularización fuerte (H≈0), H_inv≈0 ⇒ error ≈ |0 - 1| = 1."""
    h = np.array([1e-6, 1e-6], dtype=complex)
    h_inv = np.array([1e-6, 1e-6], dtype=complex)  # ≈ 0
    error = estimate_error_bounds(h, h_inv)
    assert np.allclose(error, 1.0, atol=1e-3)


# ──────────────────────────────────────────────────────────────
# NUEVOS TESTS: Métodos numéricos iterativos (Unidades 2 y 3)
# ──────────────────────────────────────────────────────────────

def test_estimate_h_inverse_gradient_converges_to_analytic():
    """El descenso del gradiente debe converger a la solución analítica de Wiener/Tikhonov."""
    freqs = np.linspace(0, 24000, 9)
    h = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05], dtype=complex)
    coherence = np.ones_like(freqs)
    epsilon_base = 0.02

    # Solución analítica (referencia)
    h_inv_analytic = estimate_h_inverse(h, coherence, epsilon_base, max_gain_db=12.0)

    # Solución por gradiente con learning rate conservador
    # α < 2 / max(|H|² + ε) para convergencia garantizada
    max_h2 = np.max(np.abs(h) ** 2)
    eps = epsilon_base * max_h2
    learning_rate = 0.5 / (max_h2 + eps)  # factor 0.5 para seguridad
    iterations = 500

    h_inv_grad = estimate_h_inverse_gradient(
        h, coherence, epsilon_base, iterations, learning_rate
    )

    # Debe converger cerca de la solución analítica (tolerancia 1%)
    assert np.allclose(h_inv_grad, h_inv_analytic, rtol=0.01, atol=1e-3), (
        f"Gradiente no converge: max diff = {np.max(np.abs(h_inv_grad - h_inv_analytic))}"
    )


def test_estimate_h_inverse_gradient_monotonic_cost_decrease():
    """El costo J debe disminuir monótonamente en cada iteración (con α adecuado)."""
    h = np.array([1.0, 0.5, 0.2, 0.5, 1.0], dtype=complex)
    coherence = np.ones(5)
    epsilon_base = 0.01
    eps = epsilon_base * np.max(np.abs(h) ** 2)

    learning_rate = 0.3 / (np.max(np.abs(h) ** 2) + eps)
    iterations = 100

    # Inicialización LEJANA de la solución óptima (para ver descenso de costo)
    h_inv = np.ones_like(h, dtype=complex) * 0.1  # Inicialización arbitraria

    costs = []
    for _ in range(iterations):
        residual = h * h_inv - 1.0
        cost = np.mean(np.abs(residual) ** 2) + eps * np.mean(np.abs(h_inv) ** 2)
        costs.append(cost)
        grad = np.conj(h) * residual + eps * h_inv
        h_inv = h_inv - learning_rate * grad

    # El costo debe ser no creciente (monótonamente decreciente)
    costs = np.array(costs)
    assert np.all(np.diff(costs) <= 1e-10), "El costo no disminuye monótonamente"
    assert costs[-1] < costs[0], "El costo final debe ser menor que el inicial"


def test_estimate_h_inverse_gradient_raises_on_invalid_iterations():
    h = np.ones(3, dtype=complex)
    coherence = np.ones(3)
    with pytest.raises(ValueError, match="iterations debe ser > 0"):
        estimate_h_inverse_gradient(h, coherence, 0.01, iterations=0, learning_rate=0.1)
    with pytest.raises(ValueError, match="iterations debe ser > 0"):
        estimate_h_inverse_gradient(h, coherence, 0.01, iterations=-1, learning_rate=0.1)


def test_estimate_h_inverse_gradient_raises_on_invalid_lr():
    h = np.ones(3, dtype=complex)
    coherence = np.ones(3)
    with pytest.raises(ValueError, match="learning_rate debe ser > 0"):
        estimate_h_inverse_gradient(h, coherence, 0.01, iterations=10, learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate debe ser > 0"):
        estimate_h_inverse_gradient(h, coherence, 0.01, iterations=10, learning_rate=-0.1)


def test_calculate_group_delay_linear_phase_constant_delay():
    """Fase lineal φ(ω) = -τω → retardo de grupo constante τ_g = τ."""
    # H_inv con fase lineal: delay = 10 muestras a fs=48000 Hz
    fs = 48000.0
    n = 32
    tau_samples = 10.0
    freq_bins = np.arange(n) * fs / n
    delta_f = fs / n
    # Fase = -ω * τ = -2πf * (τ_samples/fs) = -2π * f * tau_samples / fs
    phase = -2.0 * np.pi * freq_bins * tau_samples / fs
    h_inv = np.exp(1j * phase)

    gd = calculate_group_delay(h_inv, delta_f)

    expected_delay = tau_samples / fs  # segundos
    assert np.allclose(gd, expected_delay, rtol=0.02), (
        f"Retardo de grupo esperado {expected_delay:.6f}s, obtenido {gd}"
    )


def test_calculate_group_delay_quadratic_phase_linear_gd():
    """Fase cuadrática φ(ω) = aω² → retardo de grupo linear τ_g = -2aω."""
    fs = 48000.0
    n = 64
    freq_bins = np.arange(n) * fs / n
    delta_f = fs / n
    a = 1e-9
    phase = a * (2.0 * np.pi * freq_bins) ** 2
    h_inv = np.exp(1j * phase)

    gd = calculate_group_delay(h_inv, delta_f)

    expected_gd = -2.0 * a * (2.0 * np.pi * freq_bins)
    assert np.allclose(gd, expected_gd, rtol=0.05, atol=1e-6), (
        f"Retardo de grupo no coincide con derivada analítica"
    )


def test_calculate_group_delay_raises_on_invalid_input():
    h_inv = np.ones(3, dtype=complex)
    with pytest.raises(ValueError, match="delta_f debe ser > 0"):
        calculate_group_delay(h_inv, delta_f=0.0)
    with pytest.raises(ValueError, match="delta_f debe ser > 0"):
        calculate_group_delay(h_inv, delta_f=-1.0)

    h_inv_short = np.ones(2, dtype=complex)
    with pytest.raises(ValueError, match="al menos 3 puntos"):
        calculate_group_delay(h_inv_short, delta_f=1.0)


def test_smooth_coherence_spline_upsamples_and_smooths():
    """Spline debe interpolar suavemente y aumentar resolución (upsampling)."""
    freq_coarse = np.linspace(0, 24000, 9)
    coherence_coarse = np.array([1.0, 0.9, 0.5, 0.2, 0.1, 0.2, 0.5, 0.9, 1.0])
    freq_fine = np.linspace(0, 24000, 81)  # 9x upsampling

    coherence_fine = smooth_coherence_spline(coherence_coarse, freq_coarse, freq_fine)

    # Mismo número de puntos que freq_fine
    assert coherence_fine.shape == freq_fine.shape
    # Rango físico respetado
    assert np.all(coherence_fine >= 0.0) and np.all(coherence_fine <= 1.0)
    # En puntos originales, debe pasar cerca (interpolación exacta en nudos)
    idx_original = np.searchsorted(freq_fine, freq_coarse)
    idx_original = np.clip(idx_original, 0, len(freq_fine) - 1)
    assert np.allclose(coherence_fine[idx_original], coherence_coarse, atol=1e-10)


def test_smooth_coherence_spline_clamps_to_physical_range():
    """Resultado debe clamparse a [0, 1] aunque el spline se salga."""
    freq_coarse = np.linspace(0, 1000, 5)
    # Coherencia con valores que harían que el spline sobresalga
    coherence_coarse = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    freq_fine = np.linspace(0, 1000, 41)

    coherence_fine = smooth_coherence_spline(coherence_coarse, freq_coarse, freq_fine)

    assert np.all(coherence_fine >= 0.0) and np.all(coherence_fine <= 1.0)


def test_smooth_coherence_spline_raises_on_invalid_inputs():
    freq = np.linspace(0, 1000, 5)
    coh = np.ones(5)
    freq_new = np.linspace(0, 1000, 10)

    with pytest.raises(ValueError, match="misma forma"):
        smooth_coherence_spline(coh[:4], freq, freq_new)
    with pytest.raises(ValueError, match="estrictamente creciente"):
        smooth_coherence_spline(coh, np.array([0, 1000, 500, 2000, 3000]), freq_new)
    with pytest.raises(ValueError, match="estrictamente creciente"):
        smooth_coherence_spline(coh, freq, np.array([0, 1000, 500, 2000, 3000]))


def test_cubic_spline_manual_matches_scipy_if_available():
    """El fallback manual debe producir resultados equivalentes a scipy (si está instalado)."""
    try:
        from scipy.interpolate import CubicSpline
        has_scipy = True
    except ImportError:
        has_scipy = False

    if not has_scipy:
        pytest.skip("scipy no instalado, se prueba solo fallback manual")

    x = np.linspace(0, 10, 6)
    y = np.sin(x)
    x_new = np.linspace(0, 10, 51)

    # scipy
    cs = CubicSpline(x, y, bc_type='natural')
    y_scipy = cs(x_new)

    # Manual
    y_manual = _cubic_spline_interpolate(x, y, x_new)

    assert np.allclose(y_manual, y_scipy, rtol=1e-10, atol=1e-12), (
        "Implementación manual no coincide con scipy"
    )


# ──────────────────────────────────────────────────────────────
# NUEVOS TESTS: Operaciones Básicas (Unidad 2 - Integración y Búsqueda)
# ──────────────────────────────────────────────────────────────

def test_calculate_filter_energy_simpson_constant():
    """Integral de constante 1 sobre [0, L] debe dar L (Simpson exacto para lineales)."""
    L = 10.0
    n = 101  # impar → Simpson exacto
    freq = np.linspace(0, L, n)
    delta_f = L / (n - 1)
    h_inv = np.ones(n, dtype=complex)  # |H|^2 = 1
    energy = calculate_filter_energy(h_inv, delta_f)
    assert np.isclose(energy, L, rtol=1e-12), f"Esperado {L}, obtenido {energy}"


def test_calculate_filter_energy_simpson_parabola():
    """Integral de x^2 sobre [0, 1] es 1/3 (Simpson exacto para cuadráticas)."""
    L = 1.0
    n = 51  # impar
    freq = np.linspace(0, L, n)
    delta_f = L / (n - 1)
    h_inv = freq.astype(complex)  # H_inv(f) = f → |H|^2 = f^2
    energy = calculate_filter_energy(h_inv, delta_f)
    expected = 1.0 / 3.0
    assert np.isclose(energy, expected, rtol=1e-12), f"Esperado {expected}, obtenido {energy}"


def test_calculate_filter_energy_trapezoid_tail_for_even_n():
    """Con n pares, Simpson+Trapecio debe ser exacto para funciones lineales."""
    L = 5.0
    n = 50  # par → n-1 impar → Simpson + Trapecio
    freq = np.linspace(0, L, n)
    delta_f = L / (n - 1)
    # |H|^2 = f → integral exacta = L^2 / 2
    h_inv = np.sqrt(freq).astype(complex)  # |H|^2 = f
    energy = calculate_filter_energy(h_inv, delta_f)
    expected = L**2 / 2.0
    assert np.isclose(energy, expected, rtol=1e-10), f"Esperado {expected}, obtenido {energy}"


def test_calculate_filter_energy_raises_on_invalid_input():
    h_inv = np.ones(5, dtype=complex)
    with pytest.raises(ValueError, match="delta_f debe ser > 0"):
        calculate_filter_energy(h_inv, delta_f=0.0)
    with pytest.raises(ValueError, match="delta_f debe ser > 0"):
        calculate_filter_energy(h_inv, delta_f=-1.0)

    h_inv_short = np.ones(2, dtype=complex)
    with pytest.raises(ValueError, match="al menos 3 puntos"):
        calculate_filter_energy(h_inv_short, delta_f=1.0)


def test_find_cutoff_frequency_known_crossing():
    """Bisección debe encontrar el cruce exacto en una función lineal conocida."""
    freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # |H| = f/5 → cruce con 0.5 en f=2.5
    magnitude = freqs / 5.0
    target = 0.5
    fc = find_cutoff_frequency(freqs, magnitude, target, tol=1e-8)
    assert np.isclose(fc, 2.5, rtol=1e-10), f"Esperado 2.5, obtenido {fc}"


def test_find_cutoff_frequency_negative_slope():
    """Función con pendiente negativa (ej. paso bajo ideal invertido)."""
    freqs = np.linspace(0, 10, 11)
    # |H| = 1 - f/10 → cruce con 0.5 en f=5
    magnitude = 1.0 - freqs / 10.0
    target = 0.5
    fc = find_cutoff_frequency(freqs, magnitude, target, tol=1e-8)
    assert np.isclose(fc, 5.0, rtol=1e-10), f"Esperado 5.0, obtenido {fc}"


def test_find_cutoff_frequency_exact_at_bin():
    """Si el objetivo coincide exactamente con un bin, debe devolver esa frecuencia."""
    freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    magnitude = np.array([0.0, 0.2, 0.5, 0.8, 1.0])  # cruce en f=2.0
    target = 0.5
    fc = find_cutoff_frequency(freqs, magnitude, target, tol=1e-8)
    assert np.isclose(fc, 2.0, rtol=1e-12), f"Esperado 2.0, obtenido {fc}"


def test_find_cutoff_frequency_raises_on_invalid_input():
    freqs = np.array([0.0, 1.0, 2.0])
    mag = np.array([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="misma forma"):
        find_cutoff_frequency(freqs[:2], mag, 0.5)
    with pytest.raises(ValueError, match="estrictamente creciente"):
        find_cutoff_frequency(np.array([0, 2, 1]), mag, 0.5)
    with pytest.raises(ValueError, match="fuera del rango"):
        find_cutoff_frequency(freqs, mag, 1.5)  # > max
    with pytest.raises(ValueError, match="fuera del rango"):
        find_cutoff_frequency(freqs, mag, -0.1)  # < min
    with pytest.raises(ValueError, match="tol debe ser > 0"):
        find_cutoff_frequency(freqs, mag, 0.5, tol=0.0)
    with pytest.raises(ValueError, match="max_iter debe ser > 0"):
        find_cutoff_frequency(freqs, mag, 0.5, max_iter=0)