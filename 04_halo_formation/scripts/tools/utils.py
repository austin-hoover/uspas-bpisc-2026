import math
import numpy as np
import scipy.constants
import scipy.fft


def samp_dist(
    size: int,
    name: str = "kv",
    cov_matrix: np.ndarray = None,
    seed: int = None,
    dim: int = 4,
):
    """Sample particles from ellipsoidally symmetric distribution function.

    Parameters:
        name: Distribution name {"kv", "waterbag", "gauss"}.
        cov_matrix: Covariance matrix, shape (dim, dim).
        seed: Random number generator seed.
        dim: Number of dimensions.

    Returns:
        Particle coordinates, shape (size, dim).
    """
    rng = np.random.default_rng(seed)

    if name == "gauss":
        x = rng.normal(size=(size, dim))
    elif name == "kv":
        x = rng.normal(size=(size, dim))
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.std(x, axis=0)
    elif name == "waterbag":
        x = rng.normal(size=(size, 4))
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        r = rng.uniform(0.0, 1.0, size=size) ** (1.0 / dim)
        x = x * r[:, None]
        x = x / np.std(x, axis=0)
    else:
        raise ValueError

    if cov_matrix is not None:
        x = x @ np.linalg.cholesky(cov_matrix)
    return x


def get_classical_radius() -> float:
    """Return classical proton radius [m]."""
    m = scipy.constants.proton_mass
    q = scipy.constants.elementary_charge
    c = scipy.constants.speed_of_light
    eps0 = scipy.constants.epsilon_0
    return q**2 / (4.0 * math.pi * eps0 * m * c**2)


def intensity_from_perveance(
    perveance: float, kin_energy: float, rest_energy: float, length: float
) -> float:
    """Return beam intensity from perveance, energy, mass, and length.""" 
    gamma = 1.0 + (kin_energy / rest_energy)
    beta = math.sqrt(1.0 - (1.0 / gamma) ** 2)
    classical_radius = get_classical_radius()
    return length * perveance * beta**2 * gamma**3 / (2.0 * classical_radius)
