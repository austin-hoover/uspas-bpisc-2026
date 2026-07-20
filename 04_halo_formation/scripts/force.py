import numpy as np
import matplotlib.pyplot as plt

from tools import pcm

plt.style.use("style.mplstyle")


def force(x: float, k: float, R: float, Q: float) -> float:
    if x == 0:
        return 0.0

    f = -(k**2) * x
    if x > R:
        f -= (Q / R**2) * (1.0 - (R / x)**2) * x
    return f


def force_simple(x: float, k: float, R: float, Q: float) -> float:
    return -(k**2) * x - (Q / R**4) * x**3


k0 = np.radians(80.0)
k = k0 * 0.5
kp = np.sqrt(2.0 * (k0**2 + k**2))

eps = 1e-6  # [m rad]
R = pcm.get_eq_radius(eps, k)
Q = pcm.get_eq_perveance(eps, k, k0)

x = np.linspace(0.0, 1.5 * R, 100)
f1 = np.array([force(_x, k, R, Q) for _x in x])
f2 = np.array([force_simple(_x, k, R, Q) for _x in x])
f3 = -(k0**2) * x
f4 = -(k**2) * x

x /= R
f1 /= (k0 * R)
f2 /= (k0 * R)
f3 /= (k0 * R)
f4 /= (k0 * R)


grey = (0, 0, 0, 0.1)

fig, ax = plt.subplots(figsize=(4.0, 3.5))
ax.plot(x, f3, color=grey, ls="-", label="f3")
ax.plot(x, f4, color=grey, ls="-", label="f4")
ax.plot(x, f1, color="black", ls="-", label="f1")
ax.plot(x, f2, color="black", ls="--", label="f2")
ax.set_xlabel(r"$x / R_0$")
ax.set_ylabel(r"$x'' / k_0 R_0$")
ax.set_xlim(0.0, ax.get_xlim()[1])

i = -15
ax.annotate(
    "Exact",
    xy=(x[i], f1[i]),
    xytext=(x[i], f1[i] - 0.35),
    horizontalalignment="center",
    verticalalignment="center",
    arrowprops=dict(
        arrowstyle="->",
        alpha=0
    )
)

i = -20
ax.annotate(
    r"$-Q x^3 / R_0^4$",
    xy=(x[i], f2[i]),
    xytext=(x[i] - 0.2, f2[i]),
    horizontalalignment="center",
    verticalalignment="center",
    arrowprops=dict(
        arrowstyle="->",
        alpha=0,
    )
)

i = -70
ax.annotate(
    r"$-k_0^2 x$",
    xy=(x[i], f3[i]),
    xytext=(x[i], f3[i] - 0.3),
    horizontalalignment="center",
    verticalalignment="center",
    color=(0, 0, 0, 0.25),
    arrowprops=dict(
        arrowstyle="->",
        color=grey,
        alpha=0,
    )
)

i = -10
ax.annotate(
    r"$-k^2 x$",
    xy=(x[i], f4[i]),
    xytext=(x[i], f4[i] + 0.2),
    horizontalalignment="center",
    verticalalignment="center",
    color=(0, 0, 0, 0.25),
    arrowprops=dict(
        arrowstyle="->",
        color=grey,
        alpha=0,
    )
)

ax.set_ylim(-3.0, 0.3)
plt.savefig("outputs/fig_force")


