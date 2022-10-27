"""
ADCIRC Parameter Estimation Example

Using SpatioTemporalProblem class to aggregate temporal data from ADCIRC for solving
a two parameter estimation problem.
"""
from typing import List, Tuple

import matplotlib.colors as colors  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.tri as mtri  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from mud.base import SpatioTemporalProblem
from mud.plot import mud_plot_params, save_figure

# Matplotlib plotting options
plt.rcParams.update(mud_plot_params)


def load_adcirc_prob(
    prob_data_file,
    std_dev=0.05,
    seed=21,
    sensors=np.array([[0, 0]]),
):
    """
    Load ADCIRC Problem
    """

    adcirc_prob = SpatioTemporalProblem()
    data = adcirc_prob.load(
        str(prob_data_file),
        lam="lam",
        data="data",
        true_vals="true_vals",
        measurements=None,
        std_dev=0.05,
        sample_dist="uniform",
        domain="domain",
        lam_ref="lam_ref",
        sensors=sensors,
        times="times",
    )

    adcirc_prob.measurements_from_reference(std_dev=0.05, seed=seed)

    return data, adcirc_prob


def tri_mesh_plot(
    adcirc_grid_data,
    value="wind_speed_mult_0",
    stations=[0],
    zoom=None,
    colorbar_cutoff=-10,
    save_path: str = None,
    close_fig: bool = False,
    dpi: int = 500,
):

    triangles = mtri.Triangulation(
        adcirc_grid_data["X"], adcirc_grid_data["Y"], adcirc_grid_data["triangles"]
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)

    # Plot values and grid on top of it
    if value == "DP":
        name = "jet_terrain"
        new_map = colors.LinearSegmentedColormap.from_list(
            name, plt.cm.gist_rainbow_r(np.linspace(0.3, 0.9, 256))
        )
        cutoff_val = colorbar_cutoff
        # make the norm:  Note the center is offset so that the land has more
        divnorm = colors.SymLogNorm(
            linthresh=0.03, linscale=0.03, vmin=cutoff_val, vmax=2.0
        )
        depth = -adcirc_grid_data["DP"]
        depth[np.where(depth < cutoff_val)] = cutoff_val
        tcf = plt.tricontourf(triangles, depth, cmap=new_map, norm=divnorm, levels=100)
        cb = fig.colorbar(
            tcf, shrink=0.35, label="Bathymetry (m)", ticks=[cutoff_val, -5, -2, 0, 2]
        )
        cb.ax.set_yticklabels([rf"$<{cutoff_val}$", "-5", "-2", "0", "2"])
    elif value in adcirc_grid_data.keys():
        tcf = plt.tricontourf(triangles, adcirc_grid_data["wind_speed_mult_0"])
        plt.colorbar(tcf, fraction=0.031, pad=0.04, label="Wind Speed Multiplier")
    else:
        raise ValueError(f"Unable to find {value} in adcirc grid data set")
    plt.triplot(triangles, linewidth="0.5", color="k")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    if zoom is not None:
        ax.set_xlim([zoom[0][0] - zoom[0][1], zoom[0][0] + zoom[0][1]])
        ax.set_ylim([zoom[1][0] - zoom[1][1], zoom[1][0] + zoom[1][1]])
        ax.set_aspect("equal")

    if stations is not None:
        for i, s in enumerate(stations):
            plt.scatter(
                adcirc_grid_data["XEL"][s],
                adcirc_grid_data["YEL"][s],
                marker="x",
                color="r",
                label=f"Recording Station {i}",
            )
    ax.legend()

    save_figure(
        f"si-{value}", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
    )


def adcirc_ts_plot(
    adcirc_prob,
    time_windows=[
        ["2018-01-11 01:00:00", "2018-01-11 07:00:00"],
        ["2018-01-04 11:00:00", "2018-01-04 14:00:00"],
        ["2018-01-07 00:00:00", "2018-01-09 00:00:00"],
    ],
    wind_speeds=None,
    labels=[
        ["2018-01-10 14:00:00", 1.8],
        ["2018-01-04 00:00:00", 1.8],
        ["2018-01-07 20:00:00", 1.8],
    ],
    save_path=None,
    close_fig=False,
    dpi: int = 500,
):
    """
    ADCIRC Full Time-Series Plot

    Plot full time series Data at a node with optional labeling for time windows.
    """
    ax = adcirc_prob.plot_ts(max_plot=50)

    if wind_speeds is not None:
        ax2 = ax.twinx()
        ax2.plot(wind_speeds[0], wind_speeds[1])
        ax2.legend(["Wind Speed"])
        ax2.set_ylabel("Wind Speed (m/s)")
        _ = ax2.set_title("")

    # Time windows
    ylims = ax.get_ylim()
    color = "g"
    linestyles = ["--", ":", "-."]
    for i, t in enumerate(time_windows):
        ax.plot(pd.to_datetime([t[0], t[0]]), ylims, f"{color}{linestyles[i]}")
        ax.plot(pd.to_datetime([t[1], t[1]]), ylims, f"{color}{linestyles[i]}")
        if labels is not None and len(labels) == len(time_windows):
            ax.text(
                pd.to_datetime(labels[i][0]),
                labels[i][1],
                f"$T_{i+1}$",
                fontsize=16,
                color=color,
            )
    ax.set_ylim(ylims)
    myFmt = mdates.DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend(["Observed", "Simulated"])
    ax.set_ylabel("Water Elevation (m)")
    ax.set_xlabel("Time")
    _ = ax.set_title("")

    save_figure(
        "adcirc_full_ts", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
    )


def adcirc_time_window(
    adcirc_prob,
    time_window: Tuple[str, str],
    method="pca",
    num_components: int = 1,
    max_plot: int = 50,
    msize: int = 10,
    ylims: List[float] = None,
    title: str = None,
    save_path: str = None,
    plot_figs: List[str] = ["all"],
    close_fig: bool = False,
    dpi: int = 500,
):
    """
    ADCIRC Time-Window Plots

    Plots relevant to solving MUD problems using a given time window of ADCIRC data
    """
    t_mask = np.logical_and(
        pd.to_datetime(adcirc_prob.times) > time_window[0],
        pd.to_datetime(adcirc_prob.times) < time_window[1],
    )
    ndata = len([x for x in t_mask if x])

    prob = adcirc_prob.mud_problem(
        method=method, num_components=num_components, times_mask=t_mask
    )
    prob.estimate()
    if title is not None:
        title = " ".join(
            [
                rf"{title} :",
                rf"$\mathbb{{E}}(r_{num_components}) =",
                rf"{prob.expected_ratio():0.4}$",
            ]
        )
    if "ts-pca" in plot_figs or "all" in plot_figs:
        pca_vector_plot(
            adcirc_prob, t_mask, msize=msize, max_plot=max_plot, title=title
        )
        save_figure(
            f"pca_vecs_{num_components}_{ndata}",
            save_path,
            close_fig=close_fig,
            dpi=dpi,
            bbox_inches="tight",
        )
    if "updated_dist" in plot_figs or "all" in plot_figs:
        updated_dist_plot(prob, lam_ref=adcirc_prob.lam_ref, title=title, ylims=ylims)
        save_figure(
            f"updated_dist_{num_components}_{ndata}",
            save_path,
            close_fig=close_fig,
            dpi=dpi,
            bbox_inches="tight",
        )

    if "learned_qoi" in plot_figs or "all" in plot_figs:
        fig = plt.figure(figsize=(12, 5))

        for i in range(num_components):
            ax = fig.add_subplot(1, 2, i + 1)
            prob.plot_params_2d(ax=ax, y=i, colorbar=True)
            if i == 1:
                ax.set_ylabel("")

        if title is not None:
            _ = fig.suptitle(f"{title}", fontsize=20)
        save_figure(
            "qoi_{num_components}_{ndata}",
            save_path,
            close_fig=close_fig,
            dpi=dpi,
            bbox_inches="tight",
        )

    return prob


def updated_dist_plot(density_prob, lam_ref=None, title=None, ylims=None):
    """Plot updated distributiosn for param"""
    fig = plt.figure(figsize=(12, 5))

    # Make domain an argumet?
    # domain = np.array([[0.02, 0.12], [0.0012, 0.0038]])
    ylims = [None] * density_prob.n_params if ylims is None else ylims
    for p_idx in range(density_prob.n_params):
        ax = fig.add_subplot(1, density_prob.n_params, p_idx + 1)
        opts = {
            "ax": ax,
            "param_idx": p_idx,
            "up_opts": {
                "linestyle": "--",
                "label": rf"$\pi_\mathrm{{update}}^{{({p_idx})}}$",
            },
            "mud_opts": {
                "linestyle": "--",
                "label": rf"$\lambda_\mathrm{{MUD}}^{{({p_idx})}}$",
            },
            "true_val": lam_ref,
            "true_opts": {"color": "r"},
            "ylim": ylims[p_idx],
        }
        density_prob.plot_param_space(**opts)
        ax.legend(loc="upper right")
        ax.set_ylim([0.2, ax.get_ylim()[1]])
    if title is not None:
        _ = fig.suptitle(f"{title}", fontsize=20)


def pca_vector_plot(adcirc_prob, t_mask, msize=10, max_plot=50, title=None):
    """Plot pca vectors along with time series for a window of data"""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Time Series
    adcirc_prob.plot_ts(
        ax=ax, times=t_mask, max_plot=max_plot, meas_kwargs={"s": msize}
    )
    ax.legend(["Observed", "Samples"], loc="upper left")
    ax.set_ylabel("Water Elevation (m)")

    # PCA Vectors
    ax2 = ax.twinx()
    colors = ["blue", "orange", "black"]
    for i, vec in enumerate(adcirc_prob.pca["vecs"]):
        ax2.scatter(
            adcirc_prob.times[t_mask],
            vec,
            color=colors[i],
            marker="o",
            s=msize,
            label=f"$p^{{({i})}}$",
        )
    ax2.legend(loc="lower right")
    ax2.set_ylim([-0.3, 0.3])

    if title is not None:
        _ = fig.suptitle(f"{title}", fontsize=20)
