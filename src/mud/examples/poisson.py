"""
Poisson Problem Example

The functions here implement the poisson problem example found in [ref] section
6.1.
"""
import logging
import random
from typing import List, Union

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from scipy.interpolate import interp1d  # type: ignore
from scipy.optimize import minimize  # type: ignore

from mud.base import SpatioTemporalProblem
from mud.plot import mud_plot_params, plot_vert_line, save_figure

_logger = logging.getLogger(__name__)

# Matplotlib plotting options
plt.rcParams.update(mud_plot_params)


def load_poisson_prob(
    prob_data_file,
    std_dev=0.05,
    seed=21,
):
    """
    Load Poisson Problem Dataset


    Parameters
    ----------
    prob_data_file : pathlib.Path, str
        Path to pickled data file storing data from solving poisson problem.
    std_dev : float, default = 0.05
        Standard deviation of noise to add to true values stored in dataset
        to create fake "measurements".
    seed : int, default = 21
        For reproducible results, leave seed as is or set to a value. Otherwise,
        set to None.

    Returns
    -------
    data, poisson_prob : Tuple(dict, mud.base.SpatioTemporalProblem)
        Returns a pair of the raw dictionary loaded from the pickled dataset
        and the associated SpatioTemporalProblem class object associated with it.
    """

    poisson_prob = SpatioTemporalProblem()
    data = poisson_prob.load(
        str(prob_data_file),
        lam="lam",
        data="data",
        true_vals="true_vals",
        measurements=None,
        std_dev=0.05,
        sample_dist="uniform",
        domain="domain",
        lam_ref=None,
        sensors="sensors",
        times=np.array([0]),
    )

    poisson_prob.measurements_from_reference(std_dev=0.05, seed=seed)

    return data, poisson_prob


def plot_solution_spline(lam, aff=1000, plot_true=True, ax=None, **kwargs):
    """
    Poisson Problem Plot Boundary Solution Spline

    Parameters
    ----------

    Returns
    ----------
    """

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    if plot_true:
        x = np.linspace(0, 1, aff)
        a = 197.65032
        g = a * np.power(x, 2) * np.power((x - 1), 5)
        ax.plot(x, g, lw=5, c="k", label="$g$")

    intervals = list(np.linspace(0, 1, 2 + len(lam))[1:-1])
    ax.plot([0] + intervals + [1], [0] + list(lam) + [0], **kwargs)

    ax.set_xlabel("$x_2$")
    ax.set_ylabel(r"$g(x_2,\lambda)$")
    ax.set_xlim([0, 1])
    ax.set_ylim([-4, 0])

    return ax


def spline_objective_function_2d(lam, aff=10000):
    """
    Spline Objective Function

    Parameters
    ----------
    lam1: float
        $lambda_1$ parameter value - location of first knot at x=1/3.
    lam2: float
        $lambda_2$ parameter value - location of second knot not at x=2/3.

    Returns
    -------

    """
    x = np.linspace(0, 1, aff)
    a = 197.65032
    g = a * np.power(x, 2) * np.power((x - 1), 5)
    spline = interp1d([0, 1 / 3, 2 / 3, 1], [0, lam[0], lam[1], 0])
    vals = spline(x)

    return np.linalg.norm(g - vals)


# TODO: Document group_idxs, markers, colors
def run_2d_poisson_sol(
    data_file: str,
    sigma: float = 0.05,
    seed: int = None,
    plot_fig: Union[List[str], str] = "all",
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = False,
    order: str = "random",
    group_idxs: List[List[int]] = [[0, 5], [5, 50], [50, -1]],
    markers: List[str] = ["*", "+", "."],
    colors: List[str] = ["red", "white", "k"],
    param1_kwargs: dict = {},
    param2_kwargs: dict = {},
):
    """
    Run 2D Poisson problem

    Parameters
    ----------
    data_file : str
        Path to pickled data file storing data from solving poisson problem.
    sigma: float, default=0.05
        N(0, sigma) error added to true solution surface to produce
        "measurements".
    seed: int, default = None
        To fix results for reproducibility. Set to None to randomize results.
    plot_fig : str, default='all'
        Which figures to produce. Possible options are response, qoi, or all to
        produce all figures.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool, default=False
        Set to True to close figure and only save it.

    Returns
    -------
    posson_prob, mud_prob, axes: Tuple[mud.base.SpatioTemporalProblem,
                                       mud.base.DensityProblem,
                                       List[matplotlib.pyplot.Axes]]
        Tuple of SpatioTemporalProblem data structure storing poisson problem
        as configured by inputs, and DensityProblem object used to solve the
        inverse problem, along with the list of axes objects where desired
        plots were plotted on.
    """
    res = minimize(spline_objective_function_2d, x0=[-3, -1])
    closest = res["x"]

    raw_data, poisson_prob = load_poisson_prob(data_file, std_dev=sigma, seed=seed)
    if order == "random":
        idx_o = list(
            random.sample(range(poisson_prob.n_sensors), poisson_prob.n_sensors)
        )
    elif order == "sorted":
        idx_o = list(
            np.lexsort((poisson_prob.sensors[:, 1], poisson_prob.sensors[:, 0]))
        )
    else:
        idx_o = list(np.arange(0, poisson_prob.n_sensors, 1))
    num_components = 2
    mud_prob = poisson_prob.mud_problem(
        method="pca", num_components=num_components, sensors_mask=idx_o
    )
    _ = mud_prob.estimate()
    plot_fig = list(plot_fig) if type(plot_fig) != list else plot_fig
    axes = []
    if "response" in plot_fig or "all" in plot_fig:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)

        # Plot response surface from solving Eq. 30.
        # u stores the mesh and values as solved by fenics
        mesh, vals = raw_data["u"]
        tcf = ax.tricontourf(mesh[:, 0], mesh[:, 1], vals, levels=20, vmin=-0.5, vmax=0)
        fig.colorbar(tcf)

        # Plot points used for each ordering
        for idx, oi in enumerate(group_idxs):
            start_idx, end_idx = group_idxs[idx][0], group_idxs[idx][1]
            mask = idx_o[start_idx:end_idx]
            poisson_prob.sensor_scatter_plot(
                ax=ax,
                mask=mask,
                color=colors[idx],
                marker=markers[idx],
            )

        # Label and format figure
        _ = plt.xlim(0, 1)
        _ = plt.ylim(0, 1)
        _ = plt.title("Response Surface")
        _ = plt.xlabel("$x_1$")
        _ = plt.ylabel("$x_2$")

        ax = fig.add_subplot(1, 2, 2)
        # Plot closest solution in sample set to the reference solution
        plot_solution_spline(
            closest,
            ax=ax,
            lw=5,
            c="green",
            ls="--",
            label=r"$\hat{g}(\lambda^\dagger)$",
            zorder=50,
        )
        # Plot first 100 lambda initial
        for i, lam in enumerate(poisson_prob.lam[0:50]):
            plot_solution_spline(
                lam,
                plot_true=False,
                ax=ax,
                lw=1,
                c="purple",
                alpha=0.1,
            )

        plot_vert_line(ax, 1 / 3.0, color="b", linestyle="--", alpha=0.6)
        plot_vert_line(ax, 2 / 3.0, color="b", linestyle="--", alpha=0.6)

        ax.set_title("Boundary Condition")
        ax.set_ylabel("")
        ax.legend(
            ["$g(x_2)$", r"$\hat{g}(x_2,\lambda^\dagger)$", r"$\hat{g}(x_2,\lambda_i)$"]
        )

        fig.tight_layout()

        save_figure(
            "response_surface",
            save_path,
            close_fig=close_fig,
            dpi=dpi,
            bbox_inches="tight",
        )
        axes.append(ax)
    if "qoi" in plot_fig or "all" in plot_fig:
        fig = plt.figure(figsize=(10, 5))
        for i in range(num_components):
            ax = fig.add_subplot(1, 2, i + 1)
            mud_prob.plot_params_2d(ax=ax, y=i, contours=True, colorbar=True)
            if i == 1:
                ax.set_ylabel("")
        save_figure(
            "learned_qoi", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
        )
        axes.append(ax)
    if "densities" in plot_fig or "all" in plot_fig:
        ax1 = mud_prob.plot_param_space(param_idx=0, **param1_kwargs)
        save_figure(
            "lam1", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
        )

        ax2 = mud_prob.plot_param_space(param_idx=1, **param2_kwargs)
        save_figure(
            "lam2", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
        )
        axes.append([ax1, ax2])

    return (poisson_prob, mud_prob, axes)


def run_2d_poisson_trials(
    data_file: pd.DataFrame,
    N_vals: List[int] = [5, 50, 500],
    order: str = "random",
    ylim1: List[float] = [-0.1, 3.5],
    ylim2: List[float] = [-0.1, 2.5],
    xlim1: List[float] = [-4.5, 0.5],
    xlim2: List[float] = [-4.5, 0.5],
    annotate_location_1: List[float] = [-2.6, 1.2, 0.8],
    annotate_location_2: List[float] = [-3.5, 0.83, 0.53],
    sigma: float = 0.05,
    seed: int = None,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = False,
):
    """
    Run Poisson Problem 2D Solution Trials

    Run trials for solving Poisson boundary estimation problem using a 2D linear
    spline with increasing number of sensors over the solution space.

    Parameters
    ----------
    data_file : str
        Path to pickled data file storing data from solving poisson problem.
    N_vals: List[int], default=[5, 50, 500]
        Number of sensors to use in each trial. Each N value will produce two
        figures, one for the updated parameter distribution and MUD solution for
        each parameter value.
    ylim1: List[float], default=[-0.1, 3.5]
        y limits to use in plot for first parameter space. Set to something
        consistent so that plots for all N values will be clearly visible.
    ylim2: List[float], default=[-0.1, 2.5]
        y limits to use in plot for second parameter space. Set to something
        consistent so that plots for all N values will be clearly visible.
    xlim1: List[float], default=[-4.5, 0.5]
        x limits to use in plot for second parameter space. Set to something
        consistent so that plots for all N values will be clearly visible.
    xlim2: List[float], default=[-4.5, 0.5]
        x limits to use in plot for second parameter space. Set to something
        consistent so that plots for all N values will be clearly visible.
    sigma: float, default=1e-1
        N(0, sigma) error added to produce "measurements" from linear operator.
    seed: int, default = None
        To fix results for reproducibility. Set to None to randomize results.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool, default=False
        Set to True to close figure and only save it.

    Returns
    ----------
    poisson_prob, probs, axes : Tuple[mud.base.SpatioTemporalProblem,
                                     List[mud.base.DensityProblem],
                                     List[Tuple[matplotlib.pyplot.Axes,
                                                matplotlib.pyplot.Axes]]]
        Tuple of poisson problem from loaded data, list of mud problems solved
        for each N value, and the pair of axes objects that each mud problem
        was plotted to.
    """
    if seed is not None:
        random.seed(seed)

    res = minimize(spline_objective_function_2d, x0=[-3, -1])
    closest = res["x"]
    raw_data, poisson_prob = load_poisson_prob(data_file, std_dev=sigma, seed=seed)
    x_range = np.array([xlim1, xlim2])
    axes = []
    probs = []
    if order == "random":
        idx_o = list(
            random.sample(range(poisson_prob.n_sensors), poisson_prob.n_sensors)
        )
    elif order == "sorted":
        idx_o = list(
            np.lexsort((poisson_prob.sensors[:, 1], poisson_prob.sensors[:, 0]))
        )
    else:
        idx_o = list(np.arange(0, poisson_prob.n_sensors, 1))
    for N in N_vals:
        mud_prob = poisson_prob.mud_problem(
            method="pca", num_components=2, sensors_mask=idx_o[0:N]
        )
        _ = mud_prob.estimate()

        fig = plt.figure(figsize=(9, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        mud_prob.plot_param_space(
            ax=ax1, x_range=x_range, param_idx=0, mud_opts=None, true_opts=None
        )
        ax1.set_ylim(ylim1)
        mud_prob.plot_param_space(ax=ax1, true_val=closest, in_opts=None, up_opts=None)
        ax1.set_xlabel(r"$\lambda_1$")
        # annotate_location_1 = [-2.8, 1.2, 0.8]
        if annotate_location_1 is not None:
            x = annotate_location_1[0]
            y1 = annotate_location_1[1]
            y2 = annotate_location_1[2]
            ax1.text(x, y1, f"$N = {N}$", fontsize=18)
            e_r = mud_prob.expected_ratio()
            ax1.text(x, y2, rf"$\mathbb{{E}}(r) = {e_r:0.4}$", fontsize=18)
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        mud_prob.plot_param_space(
            ax=ax2,
            x_range=x_range,
            param_idx=1,
            mud_opts=None,
            true_opts=None,
        )
        ax2.set_ylim(ylim2)
        mud_prob.plot_param_space(
            ax=ax2, param_idx=1, true_val=closest, in_opts=None, up_opts=None
        )
        ax2.set_xlabel(r"$\lambda_2$")
        # annotate_location_2 = [-3.5, 0.83, 0.53]
        if annotate_location_2 is not None:
            x = annotate_location_2[0]
            y1 = annotate_location_2[1]
            y2 = annotate_location_2[2]
            ax2.text(x, y1, f"$N = {N}$", fontsize=18)
            e_r = mud_prob.expected_ratio()
            ax2.text(x, y2, rf"$\mathbb{{E}}(r) = {e_r:0.4}$", fontsize=18)
        ax2.legend()

        save_figure(
            f"solution_n{N}",
            save_path,
            close_fig=close_fig,
            dpi=dpi,
            bbox_inches="tight",
        )

        axes.append([ax1, ax2])
        probs.append(mud_prob)

    return (poisson_prob, probs, axes)
