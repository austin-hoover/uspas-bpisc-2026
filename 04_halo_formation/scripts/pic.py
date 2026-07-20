"""Particle-in-cell (PIC) simulation.

Note that this script spends a lot of time calculating histograms + making plots.
"""

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
from orbit.space_charge.sc2p5d import SC2p5D_AccNode
from orbit.space_charge.sc2p5d import setSC2p5DAccNodes
from orbit.teapot import ContinuousLinearFocusingTEAPOT
from orbit.teapot import TEAPOT_Lattice
from orbit.utils.consts import mass_proton

# local
from tools import pcm
from tools import utils
from tools.diag import BunchHistogramPlotter

plt.style.use("style.mplstyle")


# Arguments
# ----------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--sigma0", type=float, default=80.0, help="phase advance")
parser.add_argument("--eta", type=float, default=0.5, help="tune depression ratio")
parser.add_argument("--mismatch", type=float, default=0.6095, help="r/r0")
parser.add_argument("--emittance", type=float, default=1.0, help="4 times rms emittance")
parser.add_argument("--kin-energy", type=float, default=0.0025, help="[GeV]")
parser.add_argument("--bunch-length", type=float, default=10.0, help="[m]")
parser.add_argument("--latt-length", type=float, default=1.0, help="[m]")
parser.add_argument("--latt-parts", type=float, default=30)

parser.add_argument("--periods", type=int, default=25)
parser.add_argument("--dist", type=str, default="kv", choices=["kv", "waterbag", "gauss"])
parser.add_argument("--nparts", type=int, default=50_000)

parser.add_argument("--sc-grid", type=int, default=64)

parser.add_argument("--plot-xmax", type=float, default=3.5)
parser.add_argument("--plot-bins", type=int, default=150)
parser.add_argument("--plot-vmin", type=float, default=-3.0, help="log cutoff / peak")
parser.add_argument("--plot-smooth", type=int, default=1, help="smoothed binning (slow if 0 and lots of particles)")
parser.add_argument("--plot-pcm", type=int, default=1, help="plot pcm prediction")
parser.add_argument("--strobe-period", type=float, default=1.0)

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

# Small correction to envelope oscillation frequency (breathing mode)
env_wave_pred = (2.0 * np.pi) / np.sqrt(2.0 * (1.0 + eta**2))  # scaled

periods = 20
history = pcm.track(
    envelope=np.array([args.mismatch, 0.0]),
    particles=np.array([[2.8, 0.0]]),
    eta=args.eta,
    t_max=(periods * env_wave_pred),
    t_steps=(periods * 100),
)

idx, _ = scipy.signal.find_peaks(history["r"])
env_wave_avg = np.mean(np.diff(history["t"][idx]))
env_wave_std = np.std(np.diff(history["t"][idx]))

print("wavelength * k0 (pred) = {:0.4f}".format(env_wave_pred))
print("wavelength * k0 (calc) = {:0.4f} +- {:0.4f}".format(env_wave_avg, env_wave_std))

env_wave = env_wave_avg / k0

# Generate stroboscopic plot for later comparisons
history_pcm = pcm.track_strobe(
    envelope=np.array([args.mismatch, 0.0]),
    particles=np.array([[2.8, 0.0]]),  # ~separatrix
    eta=args.eta,
    periods=500,
)


# Initialize lattice and bunch
# ----------------------------------------------------------------------------------

# Sample phase space coordinates
particles = np.zeros((args.nparts, 6))
particles[:, :4] = utils.samp_dist(
    size=args.nparts, name=args.dist, cov_matrix=cov_matrix_init[:4, :4]
)
particles[:, 4] = args.bunch_length * np.random.uniform(-0.5, 0.5, size=args.nparts)

# Create bunch
bunch = Bunch()
bunch.mass(mass_proton)
bunch.getSyncParticle().kinEnergy(kin_energy)
bunch.macroSize(intensity / args.nparts)
for i in range(particles.shape[0]):
    bunch.addParticle(*particles[i])

# Create lattice
lattice = TEAPOT_Lattice()
lattice.addNode(
    ContinuousLinearFocusingTEAPOT(
        length=(env_wave * args.strobe_period), 
        kq=k0**2, 
        nparts=args.latt_parts
    )
)

# Add 2D space charge nodes to lattice
sc_calc = SpaceChargeCalc2p5D(args.sc_grid, args.sc_grid, 1)
sc_path_length_min = 0.001
sc_nodes = setSC2p5DAccNodes(lattice, sc_path_length_min, sc_calc)


# Track bunch
# ----------------------------------------------------------------------------------

# Initialize Twiss calculator
twiss_calc = BunchTwissAnalysis()

# Initialize extrema calculator
extrema_calc = BunchExtremaCalculator()

# Initialize histogram calculators/plotters
dims = ["x", "xp", "y", "yp"]
labels = [r"$x / r_0$", r"$x' / k_0 r_0$", r"$y / r_0$", r"$y' / k_0 r_0$"]

scale = np.array([eq_radius, eq_radius * k0, eq_radius, eq_radius * k0])
xmax = args.plot_xmax * scale
limits = list(zip(-xmax, xmax))

plotter = BunchHistogramPlotter(
    limits=limits,
    labels=labels,
    scale=scale,
    bins=args.plot_bins,
    smooth=args.plot_smooth,
    interp=None,
)

# Initialize history arrays
history = {}
for key in ["rms_x", "rms_y", "eps_x", "eps_y", "max_x", "max_y", "min_x", "min_y"]:
    history[key] = []

# Track bunch
for period in trange(args.periods + 1):
    if period > 0:
        lattice.trackBunch(bunch)

    # Analysis: covariance matrix
    twiss_calc.analyzeBunch(bunch)
    cov_matrix = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            cov_matrix[i, j] = cov_matrix[j, i] = twiss_calc.getCorrelation(i, j)

    x_rms = np.sqrt(cov_matrix[0, 0])
    y_rms = np.sqrt(cov_matrix[2, 2])
    eps_x = np.sqrt(np.linalg.det(cov_matrix[0:2, 0:2]))
    eps_y = np.sqrt(np.linalg.det(cov_matrix[2:4, 2:4]))

    # Analysis: extrema
    x_min, x_max, y_min, y_max, z_min, z_max = extrema_calc.extremaXYZ(bunch)

    # Update history arrays
    history["rms_x"].append(x_rms)
    history["rms_y"].append(y_rms)
    history["min_x"].append(x_min)
    history["min_y"].append(y_min)
    history["max_x"].append(x_max)
    history["max_y"].append(y_max)
    history["eps_x"].append(eps_x)
    history["eps_y"].append(eps_y)

    # Plot: 2D histograms
    for axis in [(0, 1), (2, 3), (0, 2)]:
        fig, ax = plotter.plot(
            bunch,
            axis=axis,
            log=True,
            mask=True,
            cmap="Greys",
            vmin=args.plot_vmin,
            fig_kws=dict(figsize=(3, 3)),
            text=f"Period = {period:02.0f}",
        )
        key = f"{dims[axis[0]]}_{dims[axis[1]]}"

        output_subdir = os.path.join(output_dir, f"fig_hist_{key}")
        os.makedirs(output_subdir, exist_ok=True)

        plt.savefig(os.path.join(output_subdir, f"fig_hist_{key}_{period:02.0f}.png"))
        plt.close()

    # Plot: 2D histograms with PCM overlay
    if args.plot_pcm:
        for axis in [(0, 1), (2, 3)]:
            fig, ax = plotter.plot(
                bunch,
                axis=axis,
                log=True,
                mask=True,
                cmap="Greys",
                vmin=args.plot_vmin,
                text=f"Period = {period:02.0f}",
                fig_kws=dict(figsize=(3, 3)),
            )

            # Get particles (x, x')
            particles_pcm = history_pcm["particles"].copy()
            particles_pcm = particles_pcm.reshape(-1, particles_pcm.shape[-1])

            # Rotate to specified phase of envelope oscillation.
            if args.strobe_period % 1 != 0:
                phase = 2.0 * np.pi * args.strobe_period * period                
                matrix = np.array([
                    [ np.cos(phase), np.sin(phase)],
                    [-np.sin(phase), np.cos(phase)]
                ])
                particles_pcm = particles_pcm @ matrix.T
            
            ax.scatter(
                particles_pcm[:, 0],
                particles_pcm[:, 1],
                ec="none",
                s=1,
                c="red",
            )
            key = f"{dims[axis[0]]}_{dims[axis[1]]}"
            output_subdir = os.path.join(output_dir, f"fig_hist_pcm_{key}")
            os.makedirs(output_subdir, exist_ok=True)
            plt.savefig(os.path.join(output_subdir, f"fig_hist_{key}_{period:02.0f}.png"))
            plt.close()

    # Plot: corner
    fig, axs = plotter.plot_corner(
        bunch,
        log=True,
        mask=True,
        cmap="Greys",
        vmin=args.plot_vmin,  # log
        diag_vmin=(args.plot_vmin - 1.0),  # log
        text=f"Period = {period:02.0f}",
    )
    output_subdir = os.path.join(output_dir, f"fig_corner")
    os.makedirs(output_subdir, exist_ok=True)
    plt.savefig(os.path.join(output_subdir, f"fig_corner_{period:02.0f}.png"))
    plt.close()


# Analysis
# ----------------------------------------------------------------------------------

for key in history:
    history[key] = np.array(history[key])

# Plot rms beam sizes
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.axhline(eq_radius * args.mismatch)
ax.plot(2.0 * history["rms_x"] / eq_radius, label="x", marker=".")
ax.plot(2.0 * history["rms_y"] / eq_radius, label="y", marker=".")
ax.set_xlabel("period")
ax.set_ylabel("rms size / r0")
ax.set_ylim(0.0, ax.get_ylim()[1] * 1.25)
plt.savefig(os.path.join(output_dir, "fig_rms.png"))
plt.close()

# Plot xmax / rms
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.axhline(eq_radius * args.mismatch)
ax.plot(history["max_x"] / (2.0 * history["rms_x"]), label="x", marker=".")
ax.plot(history["max_y"] / (2.0 * history["rms_y"]), label="y", marker=".")
ax.set_xlabel("period")
ax.set_ylabel("x_max / x_rms")
ax.set_ylim(0.0, ax.get_ylim()[1] * 1.25)
plt.savefig(os.path.join(output_dir, "fig_max_over_rms.png"))
plt.close()
