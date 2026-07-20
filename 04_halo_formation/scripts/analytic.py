"""Plot results of Gluckstern's analytic halo formation model [1].

[1] Gluckstern, Robert L. "Analytic model for halo formation in high current
    ion linacs." Physical review letters 73.9 (1994): 1247.
"""

import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("style.mplstyle")


# Setup
# --------------------------------------------------------------------------------

path = pathlib.Path(__file__)
output_dir = os.path.join("outputs", path.stem)
os.makedirs(output_dir, exist_ok=True)


# Plot eps * cos(psi) vs. w
# --------------------------------------------------------------------------------

delta = 0.35
eps = 0.1

c_eps_min = (2.0 / 3.0) * (delta + eps) ** 2
c_eps_max = (2.0 / 3.0) * (delta - eps) ** 2
c_step = abs(c_eps_min - c_eps_max) / 4.0

n_lines = 9
c_max = c_eps_min
c_min = c_max - (n_lines - 1) * c_step
c_vals = np.linspace(c_min, c_max, n_lines)

labeled_points = {}  # (w, psi)
labeled_points["Q"] = [
    (4.0 / 3.0) * (delta - eps),
    0.0,
]
labeled_points["S"] = [
    (4.0 / 3.0) * (delta + eps),
    np.pi,
]
labeled_points["P"] = [
    (4.0 / 3.0) * (eps + delta - 2.0 * np.sqrt(eps * delta)),
    np.pi,
]
labeled_points["R"] = [
    (4.0 / 3.0) * (eps + delta + 2.0 * np.sqrt(eps * delta)),
    np.pi,
]

fig, ax = plt.subplots(figsize=(5, 3))
ax.axhline(+eps, color="black", ls="--", lw=0.8)
ax.axhline(-eps, color="black", ls="--", lw=0.8)
for i, c in enumerate(c_vals):
    w = np.linspace(0.01, 1.5, 500)
    eps_cos_psi = delta - (3.0 / 8.0) * w - c / w
    ax.plot(w, eps_cos_psi, color="black", lw=None, alpha=0.3)
    if i == 4:
        ax.plot(w, eps_cos_psi, color="black", lw=None)
for label, (w, psi) in labeled_points.items():
    ax.scatter(w, eps * np.cos(psi), zorder=999, s=20)
ax.annotate(r"$+\epsilon$", xy=(1.3, +eps * 1.3), verticalalignment="center")
ax.annotate(r"$-\epsilon$", xy=(1.3, -eps * 0.7), verticalalignment="center")
ax.set_ylim(-0.2, 0.5)
ax.set_xlim(0.0, 1.4)
ax.set_xlabel(r"$w$")
ax.set_ylabel(r"$\epsilon \cos{\Psi}$")
plt.savefig(os.path.join(output_dir, "pcm_eps_cos_vs_w"))
plt.close()


# Plot Hamiltonian contours (polar)
# --------------------------------------------------------------------------------


def hamiltonian(w: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return w * delta - (3.0 / 8.0) * w**2 - eps * w * np.cos(psi)


n = 500
w = np.linspace(0.001, 1.5, n)
psi = np.linspace(0, 2.0 * np.pi, n)

w_mesh, psi_mesh = np.meshgrid(w, psi)
c_mesh = hamiltonian(w_mesh, psi_mesh)

x_mesh = w_mesh * np.cos(psi_mesh)
y_mesh = w_mesh * np.sin(psi_mesh)

fig, ax = plt.subplots(figsize=(4, 4))
ax.contour(x_mesh, y_mesh, c_mesh, levels=c_vals, linestyles="-", colors="black", alpha=0.3)
ax.contour(x_mesh, y_mesh, c_mesh, levels=[c_eps_max], linestyles="-", colors="black")

for label, (w, psi) in labeled_points.items():
    ax.scatter(w * np.cos(psi), w * np.sin(psi), zorder=99999, s=20)
ax.set_xlim(-1.6, 1.1)
ax.set_ylim(-1.4, 1.4)
ax.set_xlabel(r"$w \; \cos(\Psi_R)$")
ax.set_ylabel(r"$w \; \sin(\Psi_R)$")
plt.savefig(os.path.join(output_dir, "pcm_contours_polar"))
plt.close()


# Plot Hamiltonian contours (unfolded)
# --------------------------------------------------------------------------------

n = 500
w = np.linspace(0.001, 1.5, n)
psi = np.linspace(0, 4.0 * np.pi, n)

w_mesh, psi_mesh = np.meshgrid(w, psi)
c_mesh = hamiltonian(w_mesh, psi_mesh)

x_mesh = np.sqrt(w_mesh) * np.sin(psi_mesh / 2.0)
y_mesh = np.sqrt(w_mesh) * np.cos(psi_mesh / 2.0)

fig, ax = plt.subplots(figsize=(4, 4))
ax.contour(x_mesh, y_mesh, c_mesh, levels=c_vals, linestyles="-", colors="black", alpha=0.3)
ax.contour(x_mesh, y_mesh, c_mesh, levels=[c_eps_max], linestyles="-", colors="black")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i, (label, (w, psi)) in enumerate(labeled_points.items()):
    for sign in [-1.0, +1.0]:
        ax.scatter(
            sign * np.sqrt(w) * np.sin(psi / 2.0),
            sign * np.sqrt(w) * np.cos(psi / 2.0),
            zorder=999,
            color=colors[i],
            s=20,
        )

ax.set_xlabel(r"$\sqrt{w} \; \cos(\Psi_R / 2)$")
ax.set_ylabel(r"$\sqrt{w} \; \sin(\Psi_R / 2)$")
plt.savefig(os.path.join(output_dir, "pcm_contours_unfolded"))
plt.close()
