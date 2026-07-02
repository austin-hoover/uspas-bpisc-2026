import numpy as np
from tqdm import tqdm
from tqdm import trange

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.bunch_utils import collect_bunch
from orbit.envelope import Envelope
from orbit.envelope import EnvelopeTracker
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


def samp_dist_norm_kv(size: int) -> np.ndarray:
    coords = np.random.normal(size=(size, 4))
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / np.std(coords, axis=0)
    return coords
    

def samp_dist_norm_gauss(size: int) -> np.ndarray:
    return np.random.normal(size=(size, 4))


def samp_dist(size: int, name: str = "kv", cov_matrix: np.ndarray = None):
    if name == "kv":
        coords = samp_dist_norm_kv(size)
    elif name == "gauss":
        coords = samp_dist_norm_gauss(size)
    else:
        raise ValueError(f"Invalid distribution name '{name}'")
        
    if cov_matrix is not None:
        coords = coords @ np.linalg.cholesky(cov_matrix)
    return coords
    

def track_env_tbt(envelope: Envelope, lattice: AccLattice, turns: int, copy: bool = False) -> dict[str, np.ndarray]:    
    history_keys = ["rms_x", "rms_y"]
    history = {}
    for key in history_keys:
        history[key] = np.zeros(turns + 1)

    if copy:
        envelope_out = envelope.copy()
    else:
        envelope_out = envelope

    tracker = EnvelopeTracker(lattice, space_charge="2d")
    
    for i in range(turns + 1):
        if i > 0:
            tracker.track(envelope_out)
            
        cov_matrix = envelope_out.cov_matrix
        history["rms_x"][i] = 1000.0 * np.sqrt(cov_matrix[0, 0])
        history["rms_y"][i] = 1000.0 * np.sqrt(cov_matrix[2, 2])

    history["r"] = np.sqrt(2.0) * np.sqrt(history["rms_x"]**2 + history["rms_y"]**2)
    return history



def track_bunch_tbt(bunch: Bunch, lattice: AccLattice, turns: int, copy: bool = False) -> dict[str, np.ndarray]:    
    history_keys = ["rms_x", "rms_y"]
    history = {}
    for key in history_keys:
        history[key] = np.zeros(turns + 1)

    if copy:
        bunch_out = Bunch()
        bunch.copyBunchTo(bunch_out)
    else:
        bunch_out = bunch
        
    for i in trange(turns + 1):
        if i > 0:
            lattice.trackBunch(bunch_out)

        # twiss_calc = BunchTwissAnalysis()
        # twiss_calc.analyzeBunch(bunch_out)
        
        # cov_matrix = np.zeros((6, 6))
        # for i in range(6):
        #     for j in range(6):
        #         cov_matrix[i, j] = cov_matrix[j, i] = twiss_calc.getCorrelation(i, j)

        bunch_coords = collect_bunch(bunch_out)["coords"]
        cov_matrix = np.cov(bunch_coords.T)

        history["rms_x"][i] = 1000.0 * np.sqrt(cov_matrix[0, 0])
        history["rms_y"][i] = 1000.0 * np.sqrt(cov_matrix[2, 2])

    history["r"] = np.sqrt(2.0) * np.sqrt(history["rms_x"]**2 + history["rms_y"]**2)
    return history



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # --- System Parameters ---
# eta = 0.3          # Tune-depression ratio for Figure 5
# mu = 0.62          # Initial core radius / mismatch parameter
# tau_max = 5000     # Time span per particle

# # --- Equations of Motion ---
# def beam_halo_system(tau, y, eta):
#     """
#     y[0] = r (core radius)
#     y[1] = r' (dr/dtau)
#     y[2] = x (particle displacement)
#     y[3] = x' (dx/dtau)
#     """
#     r, r_prime, x, x_prime = y
    
#     # Core envelope equation
#     d2r_dtau2 = -r + (eta**2 / r**3) + ((1 - eta**2) / r)
    
#     # Space-charge force for the single particle
#     if abs(x) < r:
#         f_sc = x / r**2
#     else:
#         f_sc = 1 / x
        
#     # Particle equation
#     d2x_dtau2 = -x + (1 - eta**2) * f_sc
    
#     return [r_prime, d2r_dtau2, x_prime, d2x_dtau2]

# # --- Event Function ---
# # Triggers when the core radius is at a minimum (r' crosses 0 from negative to positive)
# def core_rmin_event(tau, y, eta):
#     return y[1] 

# core_rmin_event.direction = 1  

# # --- Initial Conditions (32 particles) ---
# # The paper states they are uniformly distributed along the positive horizontal and vertical axes.
# # We create 16 particles on the X-axis and 16 on the Y-axis.
# x_axis_particles = [(x_val, 0.0) for x_val in np.linspace(0.1, 4.0, 16)]
# y_axis_particles = [(0.0, yp_val) for yp_val in np.linspace(0.1, 4.0, 16)]
# initial_particles = x_axis_particles + y_axis_particles

# # --- Run Simulations and Plot ---
# plt.figure(figsize=(8, 6))

# print("Simulating 32 particles. This may take a few seconds...")

# for x0, x_prime_0 in initial_particles:
#     y0 = [mu, 0.0, x0, x_prime_0]
    
#     sol = solve_ivp(
#         beam_halo_system, 
#         [0, tau_max], 
#         y0, 
#         args=(eta,), 
#         events=core_rmin_event, 
#         method='LSODA', 
#         rtol=1e-8, 
#         atol=1e-8
#     )
    
#     # Extract and plot the stroboscopic points for this particle
#     if len(sol.y_events[0]) > 0:
#         events_state = sol.y_events[0]
#         x_strobe = events_state[:, 2]
#         x_prime_strobe = events_state[:, 3]
        
#         # We use a small dot size to recreate the dense, ink-like look of the paper's phase plots
#         plt.scatter(x_strobe, x_prime_strobe, s=1, color='red', alpha=0.5)

# plt.title(f'Figure 5: Stroboscopic Plot ($\eta={eta}$, $\mu={mu}$)')
# plt.xlabel(r'$X / R_0$')
# plt.ylabel(r"$X' / k_0 R_0$")
# plt.xlim(-4.5, 4.5)
# plt.ylim(-4.0, 4.0)
# plt.grid(True, linestyle=':', alpha=0.6)
# plt.tight_layout()
# plt.show()