"""Implements 1D particle-core model in continuous-focusing lattice [1].

[1] Wangler, T. P., et al. "Particle-core model for transverse dynamics of beam
    halo." Physical review special topics-accelerators and beams 1.8 (1998):
    084201.
"""

import numpy as np
import scipy.integrate


def get_eq_radius(emittance: float, k: float) -> float:
    return np.sqrt(emittance / k)


def get_eq_perveance(emittance: float, k: float, k0: float) -> float:
    radius = get_eq_radius(emittance, k)
    return (k0**2 - k**2) * radius**2


def get_eq_cov_matrix(radius: float, emittance: float) -> np.ndarray:
    """Return 4x4 covariance matrix for round beam at equilibrium."""
    cov_matrix = np.zeros((4, 4))
    cov_matrix[0, 0] = 0.25 * radius**2
    cov_matrix[2, 2] = cov_matrix[0, 0]
    cov_matrix[1, 1] = (0.25 * emittance) ** 2 / cov_matrix[0, 0]
    cov_matrix[3, 3] = cov_matrix[1, 1]
    return cov_matrix


def get_cov_matrix(radius: float, radius_prime: float, eta: float) -> np.ndarray:
    cov_matrix = np.zeros((2, 2))
    cov_matrix[0, 0] = radius**2
    cov_matrix[0, 1] = cov_matrix[1, 0] = radius * radius_prime
    cov_matrix[1, 1] = (eta / radius)**2 + radius_prime**2
    return cov_matrix * 0.25


def get_ellipse_params(cov_matrix: np.ndarray) -> tuple[float, float, float]:
    sii = cov_matrix[0, 0]
    sjj = cov_matrix[1, 1]
    sij = cov_matrix[0, 1]
    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)
    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2
    rx = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    ry = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))
    return (rx, ry, angle)


def ode_system(t: float, v: np.ndarray, eta: float) -> np.ndarray:
    """ODE system for 1D particle-core model in linear continuous-focusing lattice.

    Args:
        t: Dimensionless time coordinate.
            t = s * k0, where s is distance and k0 is lattice wavenumber.
        v: Vector of envelope and particle states.
            v[0] = Envelope radius R (scaled by equilibrium radius R0).
            v[1] = dR/dt.
            v[2] = particle x (scaled by R0).
            v[3] = dx/dt.
            ...
        eta: Tune depression ratio k / k0.

    Returns:
        dv/dt
    """
    r, rp, x, xp = v

    f_sc = 0.0
    if abs(x) <= r:
        f_sc = x / r**2
    else:
        f_sc = 1.0 / x

    vp = np.zeros_like(v)
    vp[0] = rp
    vp[1] = -r + (eta**2 / r**3) + ((1.0 - eta**2) / r)
    vp[2] = xp
    vp[3] = -x + (1.0 - eta**2) * f_sc
    return vp


def track(
    envelope: np.ndarray,
    particles: np.ndarray,
    eta: float,
    t_max: float,
    t_steps: float,
) -> dict[str, np.ndarray]:
    """Track particle-core system.

    Args:
        envelope: Initial envelope r and dr/dt.
        particles: Initial particle x and dx/dt coordinates, shape (n, 2).
        eta: Tune depression ratio.
        t_max: Evolution time.
        t_steps: Number of evaluation points along t axis.
    Returns:
        history: Dictionary with the following keys:
            - "t": time
            - "r": envelope r, shape (t_steps + 1).
            - "rp": envelope dr/dt, shape (t_steps + 1).
            - "particles": particle coordinates, shape (t_steps + 1, n, 2).
    """
    solutions = []
    for particle in particles:
        solution = scipy.integrate.solve_ivp(
            ode_system,
            t_span=(0.0, t_max),
            t_eval=np.linspace(0.0, t_max, t_steps + 1),
            y0=np.hstack([envelope, particle]),
            args=(eta,),
            method="LSODA",
            rtol=1e-8,
            atol=1e-8,
        )
        solutions.append(solution)

    history = {}
    history["t"] = solution.t
    history["r"] = solution.y[0]
    history["rp"] = solution.y[1]
    history["particles"] = np.zeros((t_steps + 1, particles.shape[0], particles.shape[1]))
    for index, solution in enumerate(solutions):
        history["particles"][:, index, :] = np.transpose(solution.y[2:])
    return history


def track_strobe(
    envelope: np.ndarray,
    particles: np.ndarray,
    eta: float,
    periods: int,
    phase: str = "min",
) -> dict[str, np.ndarray]:
    """Track particle-core system - evaluate at minima of core radius.

    Args:
        envelope: Initial envelope r and dr/dt.
        particles: Initial particle x and dx/dt coordinates, shape (n, 2).
        eta: Tune depression ratio.
        periods: Number of envelope oscillation periods (approximate).
        phase: Whether to evaluate system at minimum or maximum beam size {"min", "max"}.
            Uses `events` parameter of `scipy.integrate.solve_ivp` to determine
            evaluation points rather than setting integration period. (Accounts for
            slight differences from theoretical envelope oscillation period.)
    Returns:
        history: Dictionary with the following keys:
            - "t": time
            - "r": envelope r, shape (t_steps + 1).
            - "rp": envelope dr/dt, shape (t_steps + 1).
            - "particles": particle coordinates, shape (t_steps + 1, n, 2).
    """

    def event_func(t: float, y: np.ndarray, eta: float) -> float:
        return y[1] if phase == "min" else -y[1]

    event_func.direction = 1

    wavenumber = np.sqrt(2.0 * (1.0 + eta**2))  # breathing mode (approximate)
    wavelength = 2.0 * np.pi / wavenumber
    t_max = periods * wavelength

    envelope = np.copy(envelope)
    if envelope[0] == 1.0:
        envelope[0] += 0.01

    solutions = []
    for particle in particles:
        solution = scipy.integrate.solve_ivp(
            ode_system,
            t_span=(0.0, t_max),
            y0=np.hstack([envelope, particle]),
            args=(eta,),
            events=event_func,
            method="LSODA",
            rtol=1e-8,
            atol=1e-8,
        )
        solutions.append(solution)

    history = {}
    history["t"] = solution.t_events[0]
    history["r"] = solution.y_events[0][:, 0]
    history["rp"] = solution.y_events[0][:, 1]
    history["particles"] = np.zeros((solution.y_events[0].shape[0], particles.shape[0], particles.shape[1]))
    for index, solution in enumerate(solutions):
        history["particles"][:, index, :] = solution.y_events[0][:, 2:]
    return history
