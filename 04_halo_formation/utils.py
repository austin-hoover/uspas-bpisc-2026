import numpy as np

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.bunch_utils import collect_bunch
from orbit.envelope import Envelope
from orbit.envelope import EnvelopeTracker
from orbit.lattice import AccLattice
from orbit.lattice import AccNode


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
        
    for i in range(turns + 1):
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