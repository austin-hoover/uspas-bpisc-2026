"""Benchmark envelope ODE vs. PIC tracking."""

import argparse
import copy
import math
import os
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate
import scipy.signal
from tqdm import trange

from orbit.core.bunch import Bunch
from orbit.core.bunch import BunchTwissAnalysis
from orbit.core.orbit_utils import BunchExtremaCalculator
from orbit.core.spacecharge import SpaceChargeCalc2p5D
from orbit.bunch_utils import collect_bunch
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccNode
from orbit.space_charge.sc2p5d import SC2p5D_AccNode
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.teapot import ContinuousLinearFocusingTEAPOT
from orbit.teapot import TEAPOT_Lattice
from orbit.utils.consts import mass_proton

# local
from tools import pcm
from tools import utils
from tools.diag import BunchHistogramPlotter

plt.style.use("../../style.mplstyle")


# Arguments
# ----------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--sigma0", type=float, default=80.0, help="phase advance")
parser.add_argument("--eta", type=float, default=0.9, help="tune depression ratio")
parser.add_argument("--mismatch", type=float, default=0.6095, help="r/r0")
parser.add_argument("--emittance", type=float, default=1.0, help="4 times rms emittance")
parser.add_argument("--kin-energy", type=float, default=0.0025, help="[GeV]")
parser.add_argument("--bunch-length", type=float, default=10.0, help="[m]")
parser.add_argument("--latt-length", type=float, default=1.0, help="[m]")
parser.add_argument("--latt-parts", type=float, default=30)

parser.add_argument("--periods", type=int, default=10)
parser.add_argument("--dist", type=str, default="kv", choices=["kv", "waterbag", "gauss"])
parser.add_argument("--nparts", type=int, default=50_000)

parser.add_argument("--sc-grid", type=int, default=64)

parser.add_argument("--plot-xmax", type=float, default=3.5)
parser.add_argument("--plot-bins", type=int, default=150)
parser.add_argument("--plot-vmin", type=float, default=-3.0, help="log cutoff / peak")
parser.add_argument(
    "--plot-smooth", type=int, default=1, help="smoothed binning (much slower if 0)"
)
parser.add_argument("--plot-pcm", type=int, default=1, help="plot pcm prediction")

parser.add_argument("--runname", type=str, default=None)
args = parser.parse_args()


# Setup
# ----------------------------------------------------------------------------------

runname = args.runname
if runname is None:
    runname = time.strftime("%y%m%d_%H%M%S")

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem, runname)
os.makedirs(output_dir, exist_ok=True)

histories = {"pcm": {}, "bunch": {}}


# Parameters
# ----------------------------------------------------------------------------------

sigma0 = np.radians(args.sigma0)
k0 = sigma0 * args.latt_parts

eta = args.eta
sigma = sigma0 * eta
k = k0 * eta

emittance = args.emittance * 1.0e-06  # [mm mrad] (4-rms)

eq_radius = pcm.get_eq_radius(emittance, k)
perveance = pcm.get_eq_perveance(emittance, k, k0)

kin_energy = args.kin_energy  # [GeV]
rest_energy = mass_proton  # [GeV]

radius = eq_radius * args.mismatch

cov_matrix = np.zeros((6, 6))
cov_matrix[0:4, 0:4] = pcm.get_eq_cov_matrix(radius, emittance)
cov_matrix[4, 4] = args.bunch_length**2 / 12.0
cov_matrix_init = cov_matrix.copy()

intensity = utils.intensity_from_perveance(perveance, kin_energy, rest_energy, args.bunch_length)


# Track envelope
# ----------------------------------------------------------------------------------

wavenumber = np.sqrt(2.0 * (k**2 + k0**2))
wavelength = (2.0 * np.pi) / wavenumber

history = pcm.track(
    envelope=np.array([args.mismatch, 0.0]),
    particles=np.array([[2.8, 0.0]]),
    eta=args.eta,
    t_max=(args.periods * wavelength * k0),
    t_steps=(args.periods * 100),
)
for key in history:
    history[key] = np.array(history[key])

history["x_rms"] = 0.5 * history["r"]
history["y_rms"] = 0.5 * history["r"]
histories["pcm"] = copy.deepcopy(history)


# Initialize lattice and bunch
# ----------------------------------------------------------------------------------

particles = np.zeros((args.nparts, 6))
particles[:, :4] = utils.samp_dist(
    size=args.nparts, name=args.dist, cov_matrix=cov_matrix_init[:4, :4]
)
particles[:, 4] = args.bunch_length * np.random.uniform(-0.5, 0.5, size=args.nparts)

bunch = Bunch()
bunch.mass(mass_proton)
bunch.getSyncParticle().kinEnergy(kin_energy)
bunch.macroSize(intensity / args.nparts)
for i in range(particles.shape[0]):
    bunch.addParticle(*particles[i])

lattice = TEAPOT_Lattice()
lattice.addNode(
    ContinuousLinearFocusingTEAPOT(
        length=(args.periods * wavelength),
        kq=k0**2,
        nparts=(args.periods * args.latt_parts),
    )
)

sc_calc = SpaceChargeCalc2p5D(args.sc_grid, args.sc_grid, 1)
sc_path_length_min = 0.001
sc_nodes = setSC2p5DAccNodes(lattice, sc_path_length_min, sc_calc)


# Track bunch
# ----------------------------------------------------------------------------------


class BunchMonitor:
    def __init__(self, stride: float = 0.01) -> None:
        self.stride = stride
        self.last_position = 0.0

        self.twiss_calc = BunchTwissAnalysis()
        self.history = {}
        for key in ["s", "x_rms", "y_rms"]:
            self.history[key] = []

    def __call__(self, params_dict: dict) -> None:
        bunch = params_dict["bunch"]
        position = params_dict["path_length"]

        if position - self.last_position < self.stride:
            return

        self.last_position = position

        self.twiss_calc.analyzeBunch(bunch)
        x_rms = np.sqrt(self.twiss_calc.getCorrelation(0, 0))
        y_rms = np.sqrt(self.twiss_calc.getCorrelation(2, 2))

        self.history["s"].append(params_dict["path_length"])
        self.history["x_rms"].append(x_rms)
        self.history["y_rms"].append(y_rms)

        print("s={:0.2f} {:0.3f} {:0.3f}".format(position, x_rms * 1000.0, y_rms * 1000.0))


monitor = BunchMonitor(stride=0.005)
action_container = AccActionsContainer("monitor")
action_container.addAction(monitor, AccNode.ENTRANCE)
action_container.addAction(monitor, AccNode.EXIT)

params_dict = {}
params_dict["bunch"] = bunch

lattice.trackBunch(bunch, paramsDict=params_dict, actionContainer=action_container)

history = monitor.history
for key in history:
    history[key] = np.array(history[key])

history["t"] = history["s"] * k0
history["x_rms"] /= eq_radius
history["y_rms"] /= eq_radius

histories["bunch"] = copy.deepcopy(history)

# Analysis
# ----------------------------------------------------------------------------------

plot_kws = {
    "pcm": dict(color="black"),
    "bunch": dict(lw=0, marker=".", color="red"),
}

fig, ax = plt.subplots(figsize=(8, 2.5))
for model in ["pcm", "bunch"]:
    history = histories[model]
    ax.plot(history["t"], 2.0 * history["x_rms"], label=model, **plot_kws[model])
ax.set_xlabel("period")
ax.set_ylabel("rms size / r0")
ax.set_ylim(0.0, ax.get_ylim()[1] * 1.25)
plt.savefig(os.path.join(output_dir, "fig_rms.png"))
plt.close()
