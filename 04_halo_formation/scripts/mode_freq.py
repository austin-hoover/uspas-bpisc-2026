import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("style.mplstyle")


k = np.sqrt(np.linspace(0.0, 1.0))
k_mode_1 = np.sqrt(2.0 + 2.0 * k**2)
k_mode_2 = np.sqrt(1.0 + 3.0 * k**2)

fig, ax = plt.subplots(figsize=(5.0, 3.5))
ax.plot(k**2, k_mode_1**2, color="black")
ax.plot(k**2, k_mode_2**2, color="black")
ax.set_xlabel(r"Particle $\left( {k} / {k_0} \right)^2$") 
ax.set_ylabel(r"Envelope $\left( {k_{\pm}} / {k_0} \right)^2$", rotation=0, ha="right") 
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 4.0)
ax.annotate(r"$k_+$ (breathing)", xy=(0.15, 3.0))
ax.annotate(r"$k_-$ (quadrupole)", xy=(0.45, 2.0))
ax.grid(True, alpha=0.1)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/fig_mode_freq")