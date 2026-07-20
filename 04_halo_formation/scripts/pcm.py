"""Track PCM through continous focusing channel.

- Continuous-time plots for three different particles: resonant, non-resonant 
  inside core, non-resonant outside core.
- Stroboscopic plot for each particle.
- Stroboscopic plot with initial particles spanning x axis.
"""

import argparse
import math
import os
import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate
import scipy.signal

from tools import pcm

plt.style.use("style.mplstyle")


# Args
# ----------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--eta", type=float, default=0.5)
parser.add_argument("--r0", type=float, default=0.6095)
parser.add_argument("--periods", type=int, default=1000)
parser.add_argument("--n", type=int, default=50)
parser.add_argument("--xmax", type=float, default=4.0)
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


# Continuous-time plots: three particles
# ----------------------------------------------------------------------------------

# Initialize particle-core system
envelope = np.array([args.r0, 0.0])
particles = np.array([[0.748, 0.0], [0.25, 0.0], [3.25, 0.0]])

# Track
history = pcm.track(envelope, particles, eta=args.eta, t_max=134.0, t_steps=5000)

# Plot particle trajectories in position space
for index in range(history["particles"].shape[1]):
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.fill_between(history["t"], -history["r"], +history["r"], color="black", alpha=0.25, ec="none")
    ax.plot(history["t"], history["particles"][:, index, 0], color="black")
    ax.set_xlabel(r"$k_0 s$")
    ax.set_ylabel(r"$x / r_0$")
    ax.set_xlim(0.0, history["t"][-1])
    ax.set_ylim(-args.xmax, args.xmax)
    plt.savefig(os.path.join(output_dir, f"fig_continuous_x_{index}"))
    plt.close()

# Plot particle trajectories in phase space
for index in range(history["particles"].shape[1]):
    fig, ax = plt.subplots(figsize=(args.xmax, args.xmax))
    ax.plot(
        history["particles"][:, index, 0],
        history["particles"][:, index, 1],
        color="black",
        lw=0.5
    )
    for sign in [-1.0, +1.0]:
        ax.plot(
            history["r"] * sign,
            history["rp"],
            color="red",
            lw=0.5,
            alpha=0.1,
        )
    ax.set_xlabel(r"$x / r_0$")
    ax.set_ylabel(r"$x' / k_0 r_0$")
    ax.set_xlim(-args.xmax, args.xmax)
    ax.set_ylim(-args.xmax, args.xmax)
    plt.savefig(os.path.join(output_dir, f"fig_continuous_x_xp_{index}"))
    plt.close()


# Wavelength correction
# ----------------------------------------------------------------------------------

if (args.r0 != 1.0):
    idx, _ = scipy.signal.find_peaks(history["r"])
    env_wavelength_avg = np.mean(np.diff(history["t"][idx]))
    env_wavelength_std = np.std(np.diff(history["t"][idx]))
    env_wavelength_pred = (2.0 * np.pi) / np.sqrt(2.0 * (1.0 + args.eta**2))
    
    print("Envelope wavelength (calc) = {:0.4f} +- {:0.4f}".format(env_wavelength_avg, env_wavelength_std))
    print("Envelope wavelength (pred) = {:0.4f}".format(env_wavelength_pred))


# Stoboscopic plot: three particles
# ----------------------------------------------------------------------------------
    
history = pcm.track_strobe(envelope, particles, eta=args.eta, periods=args.periods)

for index in range(particles.shape[0]):
    fig, ax = plt.subplots(figsize=(args.xmax, args.xmax))
    ax.scatter(
        history["particles"][:, index, 0],
        history["particles"][:, index, 1],
        color="black",
        ec="none",
        s=1,
    )
    ax.set_xlabel(r"$x / r_0$")
    ax.set_ylabel(r"$x' / k_0 r_0$")
    ax.set_xlim(-args.xmax, args.xmax)
    ax.set_ylim(-args.xmax, args.xmax)
    plt.savefig(os.path.join(output_dir, f"fig_strobe_{index:02.0f}"))
    plt.close()


# Stoboscopic plot: x-axis span
# ----------------------------------------------------------------------------------

# Distribute particles along x axis.
particles = np.zeros((args.n, 2))
particles[:, 0] = np.linspace(0.0, 3.5, particles.shape[0])

# Track and save data at each core oscillation minimum.
history = pcm.track_strobe(envelope, particles, eta=args.eta, periods=args.periods)

# Plot x-x'.
fig, ax = plt.subplots(figsize=(args.xmax, args.xmax))
ax.scatter(
    history["particles"][:, :, 0],
    history["particles"][:, :, 1],
    color="black",
    ec="none",
    s=1,
)
ax.set_xlabel(r"$x / r_0$")
ax.set_ylabel(r"$x' / k_0 r_0$")
ax.set_xlim(-args.xmax, args.xmax)
ax.set_ylim(-args.xmax, args.xmax)
plt.savefig(os.path.join(output_dir, "fig_strobe"))

# Repeat at different envelope phase (core oscillation maximum).
history = pcm.track_strobe(envelope, particles, eta=args.eta, periods=args.periods, phase="max")

fig, ax = plt.subplots(figsize=(args.xmax, args.xmax))
ax.scatter(
    history["particles"][:, :, 0],
    history["particles"][:, :, 1],
    color="black",
    ec="none",
    s=1,
)
ax.set_xlabel(r"$x / r_0$")
ax.set_ylabel(r"$x' / k_0 r_0$")
ax.set_xlim(-args.xmax, args.xmax)
ax.set_ylim(-args.xmax, args.xmax)
plt.savefig(os.path.join(output_dir, "fig_strobe_max"))

# Plot initial test particle distribution.
fig, ax = plt.subplots(figsize=(args.xmax, args.xmax))
ax.scatter(
    particles[:, 0],
    particles[:, 1],
    color="black",
    ec="none",
    s=1,
)
ax.set_xlabel(r"$x / r_0$")
ax.set_ylabel(r"$x' / k_0 r_0$")
ax.set_xlim(-args.xmax, args.xmax)
ax.set_ylim(-args.xmax, args.xmax)
plt.savefig(os.path.join(output_dir, "fig_strobe_test"))