"""PyORBIT diagnostics."""
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt

from orbit.core.bunch import Bunch
from orbit.core.spacecharge import Grid1D
from orbit.core.spacecharge import Grid2D
from orbit.bunch_utils import collect_bunch

from .plot import plot_hist_1d
from .plot import plot_hist_2d
from .plot import CornerGrid


def get_grid_points(grid_coords: list[np.ndarray]) -> np.ndarray:
    if len(grid_coords) == 1:
        return grid_coords[0]
    return np.vstack([c.ravel() for c in np.meshgrid(*grid_coords, indexing="ij")]).T


def grid_edges_to_coords(grid_edges: np.ndarray) -> np.ndarray:
    return 0.5 * (grid_edges[:-1] + grid_edges[1:])


def grid_coords_to_edges(grid_coords: np.ndarray) -> np.ndarray:
    delta = grid_coords[1] - grid_coords[0]
    grid_edges = np.zeros(grid_coords.shape[0] + 1)
    grid_edges[0] = grid_coords[0] - 0.5 * delta
    grid_edges[1:] = grid_coords + 0.5 * delta
    return grid_edges


def make_grid(shape: tuple[int, ...], limits: list[tuple[float, float]]) -> Grid2D:
    if len(shape) == 1:
        return Grid1D(shape[0], limits[0][0], limits[0][1])
    elif len(shape) == 2:
        return Grid2D(
            shape[0] + 1,
            shape[1] + 1,
            limits[0][0],
            limits[0][1],
            limits[1][0],
            limits[1][1],
        )
    else:
        raise ValueError


class Histogram:
    def __init__(
        self, values: np.ndarray, edges: np.ndarray = None, coords: np.ndarray = None
    ) -> None:
        self.values = np.copy(values)

        self.coords = coords
        self.edges = edges
        if self.coords is None and self.edges is not None:
            self.coords = [grid_edges_to_coords(e) for e in self.edges]
        if self.edges is None and self.coords is not None:
            self.edges = [grid_coords_to_edges(c) for c in self.coords]

        self.coords = [np.copy(_) for _ in self.coords]
        self.edges = [np.copy(_) for _ in self.edges]

    def copy(self):
        return copy.deepcopy(self)


class BunchHistogramCalculator:
    def __init__(
        self,
        axis: tuple[int, ...],
        shape: tuple[int, ...],
        limits: list[tuple[float, float]],
        smooth: bool = True,
        interp: str = None,
        normalize: bool = True,
    ) -> None:
        """Constructor.

        Args:
            axis: Axis on which to compute the histogram.
            shape: Number of bins along each axis.
            limits: Min/max coordinates along each axis.
            smooth: If False, return regular histogram without any smoothing,
                calculated using NumPy. This will take longer beacuse we have
                to create the NumPy array from the bunch, and it will not work
                with MPI.
            interp: Interpolation method {"bilinear", "nine-point", None}.
            normalize: Whehter to normalize values to PDF.
        """
        self.axis = axis
        self.ndim = len(axis)
        self.normalize = normalize
        self.smooth = smooth
        self.interp = interp

        self.node = None

        if self.ndim > 2:
            raise NotImplementedError

        self.dims = ["x", "xp", "y", "yp", "z", "dE"]
        self.dims = [self.dims[i] for i in self.axis]

        self.grid_shape = shape
        self.grid_limits = limits
        self.grid_edges = [
            np.linspace(self.grid_limits[i][0], self.grid_limits[i][1], self.grid_shape[i] + 1)
            for i in range(self.ndim)
        ]
        self.grid_coords = [grid_edges_to_coords(e) for e in self.grid_edges]
        self.grid_values = np.zeros(shape)
        self.grid_points = get_grid_points(self.grid_coords)
        self.grid = make_grid(self.grid_shape, self.grid_limits)

        self.inv_cell_volume = np.prod([e[1] - e[0] for e in self.grid_edges])

    def bin_bunch(self, bunch: Bunch) -> None:
        macrosize = bunch.macroSize()
        if macrosize == 0:
            bunch.macroSize(1.0)

        if not self.smooth:
            particles = collect_bunch(bunch)["coords"]
            self.grid_values, _ = np.histogramdd(particles[:, self.axis], self.grid_edges)
        elif self.interp == "bilinear":
            self.grid.binBunchBilinear(bunch, *self.axis)
        else:
            self.grid.binBunch(bunch, *self.axis)

        bunch.macroSize(macrosize)

    def compute_histogram(self, bunch: Bunch) -> np.ndarray:
        self.bin_bunch(bunch)

        if self.smooth:
            values = np.zeros(self.grid_points.shape[0])
            if self.interp == "bilinear":
                for i, point in enumerate(self.grid_points):
                    values[i] = self.grid.getValueBilinear(*point)
            elif self.interp == "nine-point":
                for i, point in enumerate(self.grid_points):
                    values[i] = self.grid.getValue(*point)
            else:
                for i, indices in enumerate(np.ndindex(*self.grid_shape)):
                    values[i] = self.grid.getValueOnGrid(*indices)
        else:
            values = self.grid_values

        if self.normalize:
            values_sum = np.sum(values)
            if values_sum > 0.0:
                values /= values_sum
            values *= self.inv_cell_volume

        return values.reshape(self.grid_shape)

    def __call__(self, bunch: Bunch) -> Histogram:
        self.grid.setZero()
        self.grid_values = self.compute_histogram(bunch)
        return Histogram(values=self.grid_values, edges=self.grid_edges)


class BunchHistogramPlotter:
    def __init__(
        self,
        limits: list[tuple[float, float]],
        labels: list[str],
        scale: list[float],
        bins: int,
        smooth: bool = False,
        interp: str = None,
    ) -> None:
        self.ndim = len(limits)
        self.limits = limits
        self.labels = labels
        self.scale = scale
        self.smooth = smooth
        self.interp = interp

        self.hist_calcs = {}
        for i in range(self.ndim):
            for j in range(i):
                axis = (j, i)
                self.hist_calcs[axis] = BunchHistogramCalculator(
                    axis=axis,
                    shape=(bins, bins),
                    limits=[limits[k] for k in axis],
                    smooth=smooth,
                    interp=interp,
                    normalize=True,
                )
        for i in range(self.ndim):
            self.hist_calcs[i] = BunchHistogramCalculator(
                axis=(i,),
                shape=(bins,),
                limits=(limits[i],),
                smooth=smooth,
                interp=interp,
                normalize=True,
            )

        self.hists = None

    def calc_hists(self, bunch: Bunch) -> None:
        hists = {}
        if self.smooth:
            for axis, hist_calc in self.hist_calcs.items():
                hists[axis] = hist_calc(bunch)
        else:
            particles = collect_bunch(bunch)["coords"]
            for axis, hist_calc in self.hist_calcs.items():
                edges = hist_calc.grid_edges
                values, _ = np.histogramdd(particles[:, axis], edges)
                if hist_calc.normalize:
                    values_sum = np.sum(values)
                    if values_sum > 0.0:
                        values /= values_sum
                    values *= hist_calc.inv_cell_volume
                hists[axis] = Histogram(values=values, edges=edges)
        self.hists = hists

    def plot(self, bunch: Bunch, axis: tuple[int, int], text: str = None, fig_kws: dict = None, **plot_kws) -> tuple:
        """Plot 2D projection."""
        if fig_kws is None:
            fig_kws = {}

        if plot_kws is None:
            plot_kws = {}

        hist = self.hist_calcs[axis](bunch)

        values = hist.values
        values = values / np.max(values)

        edges = hist.edges
        edges[0] /= self.scale[axis[0]]
        edges[1] /= self.scale[axis[1]]

        fig, ax = plt.subplots(**fig_kws)
        plot_hist_2d(values, edges, ax=ax, **plot_kws)
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        if text:
            ax.set_title(text, fontsize="medium")
        return (fig, ax)

    def plot_corner(
        self,
        bunch: Bunch,
        ndim: int = 4,
        log: bool = False,
        vmin: float = None,
        diag_vmin: float = None,
        figsize: float = 8.0,
        text: str = None,
        diag_kws: dict = None,
        **plot_kws
    ) -> tuple:
        """Corner plot of 2D projections + 1D projections on diagonal."""

        assert ndim in (4, 6)

        if diag_kws is None:
            diag_kws = {}

        diag_kws["log"] = log
        plot_kws["log"] = log
        plot_kws["vmin"] = vmin

        grid = CornerGrid(
            ndim=ndim,
            limits=self.limits,
            labels=self.labels,
            figsize=(figsize, figsize),
        )

        fig = grid.fig
        axs = grid.axs

        self.calc_hists(bunch)

        for i in range(ndim):
            for j in range(i):
                axis = (j, i)
                hist = self.hists[axis].copy()

                values = hist.values
                values = values / np.max(values)

                edges = hist.edges
                edges[0] /= self.scale[axis[0]]
                edges[1] /= self.scale[axis[1]]

                plot_hist_2d(values, edges, ax=axs[i, j], **plot_kws)

        for i in range(ndim):
            hist = self.hists[i]

            values = np.squeeze(hist.values)
            values = values / np.max(values)

            edges = hist.edges[0]
            edges /= self.scale[i]

            plot_hist_1d(values, edges, ax=axs[i, i], **diag_kws)

        grid.set_diag_ylim(diag_vmin)

        if log:
            for i in range(ndim):
                ax = axs[i, i]
                ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.5)

        if text:
            axs[0, 1].annotate(
                text, 
                xy=(0.5, 0.5),
                horizontalalignment="center",
                verticalalignment="center",
            )
            
        return (fig, axs)
