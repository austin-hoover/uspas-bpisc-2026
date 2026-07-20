import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calc_rms_ellipse_params(cov_matrix: np.ndarray) -> tuple[float, float, float]:
    i, j = (0, 1)

    sii = cov_matrix[i, i]
    sjj = cov_matrix[j, j]
    sij = cov_matrix[i, j]

    angle = -0.5 * np.arctan2(2.0 * sij, sii - sjj)

    _sin = np.sin(angle)
    _cos = np.cos(angle)
    _sin2 = _sin**2
    _cos2 = _cos**2

    c1 = np.sqrt(abs(sii * _cos2 + sjj * _sin2 - 2 * sij * _sin * _cos))
    c2 = np.sqrt(abs(sii * _sin2 + sjj * _cos2 + 2 * sij * _sin * _cos))

    return (c1, c2, angle)


def plot_ellipse(
    rx: float = 1.0,
    ry: float = 1.0,
    angle: float = 0.0,
    center: tuple[float, float] = None,
    ax=None,
    **kws,
):
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    kws.setdefault("lw", 1.25)

    center = (0.0, 0.0)

    d1 = rx * 2.0
    d2 = ry * 2.0
    angle = -np.degrees(angle)

    ax.add_patch(patches.Ellipse(center, d1, d2, angle=angle, **kws))
    return ax


def plot_rms_ellipse(cov_matrix: np.ndarray, level: float = 1.0, ax=None, **ellipse_kws):
    rx, ry, angle = calc_rms_ellipse_params(cov_matrix)
    plot_ellipse(rx * level, ry * level, angle=angle, ax=ax, **ellipse_kws)
    return ax


def plot_hist_1d(
    values: np.ndarray,
    edges: np.ndarray,
    blur: float = None,
    log: bool = False,
    ax=None,
    **kws,
):
    kws.setdefault("lw", 1.5)
    kws.setdefault("color", "black")

    if blur:
        values = scipy.ndimage.gaussian_filter(values, sigma=blur)
    if log:
        values = np.log10(values + 1e-15)
        kws["baseline"] = None

    ax.stairs(values, edges, **kws)


def plot_hist_2d(
    values: np.ndarray,
    edges: list[np.ndarray],
    blur: float = None,
    mask: bool = False,
    log: bool = False,
    ax=None,
    **kws,
):
    kws.setdefault("linewidth", 0.0)
    kws.setdefault("rasterized", True)
    kws.setdefault("shading", "auto")

    values = values.copy()
    values = values / np.max(values)
    if mask:
        values = np.ma.masked_less_equal(values, 0.0, None)
    if log:
        if mask:
            values = np.ma.log10(values)
        else:
            values = np.log10(values + 1e-12)

    ax.pcolormesh(edges[0], edges[1], values.T, **kws)


class CornerGrid:
    def __init__(
        self,
        ndim: int,
        labels: list[str] = None,
        figsize: tuple[float, float] = None,
    ) -> None:

        self.ndim = ndim
        self.labels = labels

        if self.labels is None:
            self.labels = ndim * [""]

        if figsize is None:
            figsize = (2.0 * ndim, 2.0 * ndim)

        self.fig, self.axs = plt.subplots(ncols=ndim, nrows=ndim, sharex=False, sharey=False, figsize=figsize)
        for j in range(ndim):
            for i in range(j):
                self.axs[i, j].axis("off")

        for ax in self.axs.flat:
            for loc in ["top", "right"]:
                ax.spines[loc].set_visible(False)

        for i in range(0, ndim - 1):
            for j in range(0, ndim):
                self.axs[i, j].set_xticklabels([])
        for i in range(0, ndim):
            for j in range(1, ndim):
                self.axs[i, j].set_yticklabels([])

        for i, label in enumerate(self.labels):
            self.axs[-1, i].set_xlabel(label)

        for i, label in enumerate(self.labels[1:], start=1):
            self.axs[i, 0].set_ylabel(label)

        self.axs[0, 0].set_yticklabels([])
        self.axs[0, 0].set_ylabel(None)

        self.fig.align_ylabels()
        self.fig.align_xlabels()

        self.limits = None

    def set_diag_ylim(self, ymin: float = None) -> None:
        ymax = 0.0
        if ymin is None:
            ymin = np.inf
            for i in range(self.ndim):
                ymax = max(ymax, self.axs[i, i].get_ylim()[1])
                ymin = min(ymin, self.axs[i, i].get_ylim()[0])
        for i in range(self.ndim):
            self.axs[i, i].set_ylim(ymin, ymax)

    def set_limits(self, limits: list[tuple[float, float]]) -> None:
        self.limits = limits
        for j in range(self.ndim):
            for ax in self.axs[:, j]:
                ax.set_xlim(limits[j])
        for i in range(1, self.ndim):
            for ax in self.axs[i, :i]:
                ax.set_ylim(limits[i])

    def plot_scatter(
        self,
        particles: np.ndarray,
        bins: int = 10,
        diag_kws: dict = None,
        **plot_kws,
    ) -> None:

        if diag_kws is None:
            diag_kws = dict()
            
        for i in range(self.ndim):
            for j in range(i):
                self.axs[i, j].scatter(particles[:, j], particles[:, i], **plot_kws)
                
        for i in range(self.ndim):
            values, edges = np.histogram(particles[:, i], bins=bins, range=self.limits[i])
            plot_hist_1d(values, edges, ax=self.axs[i, i], **diag_kws)

        self.set_limits(self.limits)

    def plot_rms(self, particles: np.ndarray, **plot_kws) -> None:            
        for i in range(self.ndim):
            for j in range(i):
                cov_matrix = np.cov(particles[:, j], particles[:, i])
                plot_rms_ellipse(cov_matrix, level=2.0, ax=self.axs[i, j], **plot_kws)

    def plot_hist(self, particles: np.ndarray, bins: int = 10, diag_kws: dict = None, **plot_kws) -> None:
        if diag_kws is None:
            diag_kws = dict()

        for i in range(self.ndim):
            for j in range(i):
                axis = (j, i)
                values, edges = np.histogramdd(particles[:, axis], bins=bins, range=[self.limits[k] for k in axis])
                plot_hist_2d(values, edges, ax=self.axs[i, j], **plot_kws)
                
        for i in range(self.ndim):
            values, edges = np.histogram(particles[:, i], bins=bins, range=self.limits[i])
            plot_hist_1d(values, edges, ax=self.axs[i, i], **diag_kws)
