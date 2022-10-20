"""
MUD vs MAP Comparison Example

Functions for running 1-dimensional polynomial inversion problem.
"""
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from scipy.stats import norm

from mud.base import SpatioTemporalProblem

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

plt.backend = "Agg"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["font.size"] = 16
plt.rcParams["text.usetex"] = True
plt.rcParams[
    "text.latex.preamble"
] = r"\usepackage{bm} \usepackage{amsfonts} \usepackage{amsmath}"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["axes.titlesize"] = 26
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


def load_adcirc_prob(
    prob_data_file,
    std_dev=0.05,
    seed=21,
    sensors=None,
):

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

    return data, load_adcirc_prob


def plot_adcirc_ts(time_windows: List[Tuple[str, str]] = None):
    """
    Plot ADCIRC Time Series

    """

    # Decent results from 11, 21
    raw_data, adcirc_prob = load_adcirc_prob(df, std_dev=0.05, seed=21)
    ax = adcirc_prob.plot_ts(max_plot=50)
    if "met_data" in raw_data.keys():
        ax2 = ax.twinx()
        raw_data["met_data"]["wind_speed"] = np.sqrt(
            raw_data["met_data"]["u_wind"] ** 2 + raw_data["met_data"]["v_wind"] ** 2
        )
        raw_data["met_data"]["wind_speed"].plot(ax=ax2)
        ax2.legend(["Wind Speed"])
        ax2.set_ylabel("Wind Speed (m/s)")
        _ = ax2.set_title("")

    for i, (t_start, t_end) in enumerate(time_windows):
        ylims = ax.get_ylim()
        ax.plot(pd.to_datetime([t_start, t_start]), ylims, "g--")
        ax.plot(pd.to_datetime([t_end, t_end]), ylims, "g--")

    _ = ax.set_ylim(ylims)
    myFmt = mdates.DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend(["Observed", "Simulated"])
    ax.set_ylabel("Water Elevation (m)")
    ax.set_xlabel("Time")
    _ = ax.set_title("")

    save_figure(
        "adcirc_full_ts", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
    )


def adcdirc_time_window(
    df,
    time_window: Tuple[str, str],
    method="pca",
    num_components: int = 1,
    ax: plt.Axes = None,
    max_plot: int = 50,
    msize: int = 10,
):

    raw_data, adcirc_prob = load_adcirc_prob(df, std_dev=0.05, seed=21)

    t_mask = np.logical_and(
        pd.to_datetime(raw_data["times"]) > time_window[0],
        pd.to_datetime(raw_data["times"]) < time_window[1],
    )

    ndata = len([x for x in t1_mask if x])

    prob = adcirc_prob.mud_problem(
        method=method, num_components=num_components, times_mask=t_mask
    )
    mud_pt = prob.estimate()
    exp_r = prob.exp_r()
    exp_r_str = f"$\mathbb{{E}}(r_1) = {exp_r:0.4}$"
    if "lam_ref" in raw_data.keys():
        err = np.linalg.norm(raw_data["lam_ref"] - mud_pt)

    if "ts-pca" in plot_figs or "all" in plot_figs:
        if ax is None:
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
                raw_data["times"][t_mask],
                vec,
                color=color[i],
                marker="o",
                s=msize,
                label=f"$p^{{({i})}}$",
            )
        ax2.legend(loc="lower right")
        ax2.set_ylim([-0.3, 0.3])

        save_figure(
            "adcirc_t{pca_t1_vecs",
            save_path,
            close_fig=close_fig,
            dpi=dip,
            bbox_inches="tight",
        )
    if "updated_dist" in plot_figs or "all" in plot_figs:

        fig = plt.figure(figsize=(12, 5))

        # Make domain an argumet?
        # domain = np.array([[0.02, 0.12], [0.0012, 0.0038]])
        for p_idx in range(prob.n_params):
            ax = fig.add_subplot(1, prob.n_params, p_idx + 1)
            opts = {
                "ax": ax,
                "param_idx": p_idx,
                "up_opts": {
                    "linestyle": "--",
                    "label": f"$\pi_\\text{{update}}^{{({p_idx})}}$",
                },
                "mud_opts": {
                    "linestyle": "--",
                    "label": f"$\lambda_\\text{{MUD}}^{{({p_idx})}}$",
                },
            }
            if "lam_ref" in raw_data.keys():
                opts["true_val"] = raw_data["lam_ref"]
                opts["true_opts"] = {"color": "r"}
            prob.plot_param_space(**opts)
            ax.legend()

        save_figure(
            "updated_dist", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight"
        )

    if "learned_qoi" in plot_figs or "all" in plot_figs:
        fig = plt.figure(figsize=(9, 4))

        for i in range(num_components):
            ax = fig.add_subplot(1, 2, i + 1)
            prob.plot_params_2d(ax=ax, y=i, colorbar=True)
            if i == 1:
                ax.set_ylabel("")

        save_figure("qoi", save_path, close_fig=close_fig, dpi=dpi, bbox_inches="tight")
