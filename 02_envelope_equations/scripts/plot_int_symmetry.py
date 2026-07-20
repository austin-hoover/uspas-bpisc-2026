import matplotlib.pyplot as plt
import numpy as np

plt.style.use("../../style.mplstyle")


fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(4, 2))
for ax, y in zip(axs, [1.0, 0.0]):
    ax.fill_between([0, 1], [0, 1], y, color="grey")
    ax.plot([0, 1], [0, 1], color="black", lw=2)
    ax.set_aspect(1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for loc in ["top", "right"]:
        ax.spines[loc].set_visible(False)
axs[0].set_xlabel(r"$r$", fontsize="x-large")
axs[0].set_ylabel(r"$r'$", fontsize="x-large")
plt.savefig("../images/int_symmetry")
