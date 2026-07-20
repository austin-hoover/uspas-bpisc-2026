import os
import numpy as np
import matplotlib.pyplot as plt

from plotting import CornerGrid

plt.style.use("../../style.mplstyle")


rng = np.random.default_rng(0)

ndim = 6
X = rng.normal(size=(1_000, ndim))
M = rng.normal(size=(ndim, ndim))
X = X @ M.T
X = X / np.std(X, axis=0)


limits = ndim * [(-4.0, 4.0)]

grid = CornerGrid(
    ndim=6,
    figsize=(8.0, 8.0), 
    labels=[r"$x$", r"$p_x$", r"$y$", r"$p_y$", r"$z$", r"$p_z$"],
)
grid.set_limits(limits)
grid.plot_scatter(X[:1000], s=0.1, c="grey", bins=50, diag_kws=dict(color="grey", lw=1.5))
grid.plot_rms(X, color="red")
for ax in grid.axs.flat:
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig("../images/corner.png")
