"""
MUD Examples CLI

CLI for running MUD examples
"""
import ast
import json
import re
from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from prettytable import PrettyTable  # type: ignore
from wget import download  # type: ignore

from .adcirc import adcirc_time_window, adcirc_ts_plot, load_adcirc_prob, tri_mesh_plot
from .comparison import run_comparison_example
from .fenics import fin_flag, run_fenics
from .linear import run_contours, run_high_dim_linear, run_wme_covariance
from .poisson import run_2d_poisson_sol, run_2d_poisson_trials


def print_res(res, fields, search=None, match=r".", filter_fun=None):
    """
    Print results

    Prints dictionary keys in list `fields` for each dictionary in res,
    filtering on the search column if specified with regular expression
    if desired.

    Parameters
    ----------
    res : List[dict]
        List of dictionaries containing response of an AgavePy call
    fields : List[string]
        List of strings containing names of fields to extract for each element.
    search : string, optional
        String containing column to perform string patter matching on to
        filter results.
    match : str, default='.'
        Regular expression to match strings in search column.
    output_file : str, optional
        Path to file to output result table to.

    Examples
    --------

    Printing list of dictionaries in a pretty table:

    >>> vals = [{'a': 'foo'}, {'a': 'bar'}]
    >>> print(print_res(vals, fields=['a']))
    +-----+
    |  a  |
    +-----+
    | foo |
    | bar |
    +-----+

    Filtering results based off of regex matchingL

    >>> print(print_res(vals, fields=['a'], search='a', match='foo'))
    +-----+
    |  a  |
    +-----+
    | foo |
    +-----+

    """
    # Initialize Table
    x = PrettyTable(float_format="0.2")
    x.field_names = fields

    # Build table from results
    filtered_res = []
    for r in res:
        if filter_fun is not None:
            r = filter_fun(r)
        if search is not None:
            if re.search(match, r[search]) is not None:
                x.add_row([r[f] for f in fields])
                filtered_res.append(dict([(f, r[f]) for f in fields]))
        else:
            x.add_row([r[f] for f in fields])
            filtered_res.append(dict([(f, r[f]) for f in fields]))

    return str(x)


@click.group(short_help="MUD examples problems")
@click.option(
    "-s/-ns",
    "--show/--no-show",
    default=False,
    help="Whether to show figures or not.",
    show_default=True,
)
@click.option("--seed", default=None, type=int, help="Seed for fixing results.")
@click.option(
    "--save-path", default=".", help="Path to save figures to.", show_default=True
)
@click.option(
    "--dpi",
    default=500,
    help="Resolution in dpi to use for output images.",
    show_default=True,
)
@click.pass_context
def examples(ctx, show, seed, save_path, dpi):
    ctx.ensure_object(dict)
    ctx.obj["show"] = show
    ctx.obj["seed"] = seed
    ctx.obj["save_path"] = save_path
    ctx.obj["dpi"] = dpi
    if save_path is not None:
        Path(save_path).mkdir(exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
    pass


@examples.command(short_help="MUD vs MAP comparison example.")
@click.option(
    "-p", "--power", default=5, help="Power of exponential function to invert."
)
@click.option(
    "-n",
    "--num_samples",
    default=1000,
    help="Number of samples to use to solve inverse problems",
)
@click.option("-m", "--mu", default=0.25, help="True observed value.")
@click.option("-s", "--sigma", default=0.1, help="Assumed noise in measured data.")
@click.option(
    "-d",
    "--domain",
    default=[-1, 1],
    multiple=True,
    help="Assumed domain of possible values for lambda.",
)
@click.option(
    "--n-vals",
    default=[1, 5, 10, 20],
    multiple=True,
    help="".join(
        [
            "Values for N, the number of data-points to use to",
            " solve inverse problems, to use. Each N value will produce",
            "a separate plot.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-l/-nl",
    "--latex-labels/--no-latex-labels",
    default=False,
    help="Whether to use latex labels in plot.",
    show_default=True,
)
@click.pass_context
def comparison(
    ctx,
    n_vals: List[int] = [1, 5, 10, 20],
    latex_labels: bool = True,
    power: int = 5,
    num_samples: int = 1000,
    mu: float = 0.25,
    sigma: float = 0.1,
    domain: List[int] = [-1, 1],
):
    """
    Run MUD vs MAP Comparison Example

    Entry-point function for running simple polynomial example comparing
    Bayesian and Data-Consistent solutions. Inverse problem involves inverting
    the polynomial QoI map Q(lam) = lam^5.
    """
    _ = run_comparison_example(
        N_vals=n_vals,
        latex_labels=latex_labels,
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
        p=power,
        num_samples=num_samples,
        mu=mu,
        sigma=sigma,
        domain=domain,
    )
    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(short_help="MUD vs MAP linear solution contour plots.")
@click.option(
    "-f",
    "--lin_prob_file",
    default=None,
    help="Path to json config file for linear problem.",
    type=str,
    show_default=True,
)
@click.option(
    "-p",
    "--plot_fig",
    default=["all"],
    multiple=True,
    help="Figures to plot.",
    show_default=True,
)
@click.pass_context
def contours(
    ctx,
    lin_prob_file=None,
    plot_fig=["all"],
):
    """ """
    if lin_prob_file is not None:
        with open(lin_prob_file, "r") as fp:
            lin_prob = json.load(fp)

        for k in lin_prob.keys():
            lin_prob[k] = np.array(lin_prob[k])
    else:
        lin_prob = {}

    _ = run_contours(
        plot_fig, save_path=ctx.obj["save_path"], dpi=ctx.obj["dpi"], **lin_prob
    )

    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(
    short_help="Spectral properties of updated covariance for linear WME Maps."
)
@click.option(
    "-i",
    "--dim_input",
    default=20,
    show_default=True,
    help="Input dimension of linear map (number of rows in A).",
)
@click.option(
    "-o",
    "--dim_output",
    default=5,
    show_default=True,
    help="Output dimension of linear map (number of columns in A).",
)
@click.option(
    "-s",
    "--sigma",
    default=1e-1,
    show_default=True,
    help='N(0, sigma) error added to produce "measurements" from linear operator.',
)
@click.option(
    "-n",
    "--num_data",
    default=[10, 100, 1000, 10000],
    help="".join(
        [
            "List of number of data points to collect in",
            " constructing Q_WME map to view how the spectral ",
            "properties of the updated covariance change as ",
            "more data is included in the Q_WME map.",
        ]
    ),
    show_default=True,
    multiple=True,
)
@click.pass_context
def wme_covariance(
    ctx,
    dim_input: int = 20,
    dim_output: int = 5,
    sigma: float = 1e-1,
    num_data=[10, 100, 1000, 10000],
):
    """
    Weighted Mean Error Map Updated Covariance

    Reproduces Figure 4 from [ref], showing the spectral properties of the
    updated covariance for a the Weighted Mean Error map on a randomly
    generated linear operator as more data from repeated measurements is used
    to construct the QoI map.
    """
    _ = run_wme_covariance(
        dim_input=dim_input,
        dim_output=dim_output,
        sigma=sigma,
        Ns=num_data,
        seed=ctx.obj["seed"],
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
        close_fig=False,
    )

    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(
    short_help="MUD/MAP/Least squares convergence for increasing dimension and rank."
)
@click.option(
    "-i",
    "--dim_input",
    default=100,
    show_default=True,
    help="Input dimension of linear map (number of rows in A).",
)
@click.option(
    "-o",
    "--dim_output",
    default=100,
    show_default=True,
    help="Output dimension of linear map (number of columns in A).",
)
@click.pass_context
def high_dim_linear(
    ctx,
    dim_input=100,
    dim_output=100,
):
    """
    Run High Dimension Linear Example

    Reproduces Figure 6 from [ref], showing the relative error between the true
    parameter value and the MUD, MAP and least squares solutions to linear
    gaussian inversion problems for increasing dimension and rank of a randomly
    generated linear map A.
    """
    _ = run_high_dim_linear(
        dim_input=dim_input,
        dim_output=dim_output,
        seed=ctx.obj["seed"],
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
        close_fig=False,
    )
    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(short_help="Poisson 2D parameter estimation problem solution.")
@click.argument("data_file")
@click.option(
    "-s",
    "--sigma",
    default=1e-1,
    show_default=True,
    help='N(0, sigma) error added to true solution surface to produce "measurements".',
)
@click.option(
    "-p",
    "--plot_fig",
    default=["all"],
    multiple=True,
    help="Figures to plot.",
    show_default=True,
)
@click.pass_context
def poisson_solve(
    ctx,
    data_file,
    sigma=0.05,
    plot_fig=["all"],
):
    """
    Run Poisson Example

    Reproduces Figure 7 and 8 from [ref].
    """
    res = run_2d_poisson_sol(
        data_file=data_file,
        sigma=sigma,
        seed=ctx.obj["seed"],
        plot_fig=plot_fig,
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
        close_fig=False,
    )

    mud_sol = res[1].estimate()
    print(mud_sol)

    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(short_help="Poisson 2D parameter estimation problem solution.")
@click.argument("data_file")
@click.option(
    "-n",
    "--n_vals",
    default=[5, 50, 500],
    multiple=True,
    help="".join(
        [
            "Values for N, the number of sensors to use to",
            " solve inverse problems, to use. Each N value will produce",
            "two separate plots, one for each parameter.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-y1",
    "--ylim1",
    default=[-0.1, 3.5],
    multiple=True,
    help="".join(
        [
            "y limits to use in plot for first parameter space.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-y2",
    "--ylim2",
    default=[-0.1, 2.5],
    multiple=True,
    help="".join(
        [
            "y limits to use in plot for second param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-x1",
    "--xlim1",
    default=[-4.5, 0.5],
    multiple=True,
    help="".join(
        [
            "x limits to use in plot for first param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-x2",
    "--xlim2",
    default=[-4.5, 0.5],
    multiple=True,
    help="".join(
        [
            "x limits to use in plot for second param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-s",
    "--sigma",
    default=1e-1,
    show_default=True,
    help='N(0, sigma) error added to true solution surface to produce "measurements".',
)
@click.option(
    "-p",
    "--plot_fig",
    default=["all"],
    multiple=True,
    help="Figures to plot.",
    show_default=True,
)
@click.pass_context
def poisson_trials(
    ctx,
    data_file,
    n_vals: List[int] = [5, 50, 500],
    ylim1: List[float] = [-0.1, 3.5],
    ylim2: List[float] = [-0.1, 2.5],
    xlim1: List[float] = [-4.5, 0.5],
    xlim2: List[float] = [-4.5, 0.5],
    sigma=0.05,
    plot_fig=["all"],
):
    """
    Run Poisson Example

    Reproduces Figure 7 and 8 from [ref].
    """
    n_vals = list(n_vals)
    n_vals.sort()
    res = run_2d_poisson_trials(
        data_file,
        N_vals=n_vals,
        ylim1=ylim1,
        ylim2=ylim2,
        xlim1=xlim1,
        xlim2=xlim2,
        sigma=sigma,
        seed=ctx.obj["seed"],
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
        close_fig=False,
    )
    runs = []
    for i, p in enumerate(res[1]):
        runs.append({"N": n_vals[i], "mud_pt": p.estimate(), "r": p.expected_ratio()})
    print(print_res(runs, fields=["N", "mud_pt", "r"]))

    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(short_help="Generate Poisson problem dataset using FEniCS.")
@click.argument("num_samples", type=int)
@click.argument("num_sensors", type=int)
@click.option(
    "--mins",
    type=float,
    default=[-4, -4],
    multiple=True,
    show_default=True,
    help="".join(
        [
            "Minimum value for input domain ranges. Note: ",
            "length of vector determines dimension.",
        ]
    ),
)
@click.option(
    "--maxs",
    default=[0, 0],
    multiple=True,
    type=float,
    show_default=True,
    help="".join(
        [
            "Minimum value for input domain ranges. Note: ",
            "length of vector determines dimension.",
        ]
    ),
)
@click.option(
    "--sensor_low",
    default=[0, 0],
    type=float,
    multiple=True,
    show_default=True,
    help="".join(["x, y minimum values in response surface for sensor" "locations"]),
)
@click.option(
    "--sensor_high",
    default=[1, 1],
    type=float,
    multiple=True,
    show_default=True,
    help="".join(["x, y minimum values in response surface for sensor" "locations"]),
)
@click.option(
    "-y1",
    "--ylim1",
    default=[-0.1, 3.5],
    multiple=True,
    help="".join(
        [
            "y limits to use in plot for first parameter space.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-y2",
    "--ylim2",
    default=[-0.1, 2.5],
    multiple=True,
    help="".join(
        [
            "y limits to use in plot for second param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-x1",
    "--ylim1",
    default=[-4.5, 0.5],
    multiple=True,
    help="".join(
        [
            "x limits to use in plot for first param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-x2",
    "--xlim2",
    default=[-4.5, 0.5],
    multiple=True,
    help="".join(
        [
            "x limits to use in plot for second param plot.",
            " Set to something consistent so that plots for ",
            "all N values will be clearly visible.",
        ]
    ),
    show_default=True,
)
@click.option(
    "-g",
    "--gamma",
    default=-3,
    show_default=True,
    type=int,
    help="Parameter controlling min value of true boundary condition.",
)
@click.pass_context
def poisson_generate(
    ctx,
    num_samples,
    num_sensors,
    mins=[-4, -4],
    maxs=[0, 0],
    sensor_low=[0, 0],
    sensor_high=[1, 1],
    ylim1=[-0.1, -3.5],
    ylim2=[-0.1, 2.5],
    xlim1=[-4.1, 0.5],
    xlim2=[-4.1, 0.5],
    gamma=-3,
):
    """
    Poisson Forward Solver using Fenics
    """
    path = "." if ctx.obj["save_path"] is None else ctx.obj["save_path"]
    try:
        res, p = run_fenics(
            num_samples,
            num_sensors,
            mins=mins,
            maxs=maxs,
            sensor_low=sensor_low,
            sensor_high=sensor_high,
            gamma=gamma,
            save_path=path,
            seed=ctx.obj["seed"],
        )
        print(f"{p}")
    except ModuleNotFoundError as e:
        print(f"Unable to run fenics - {e}")

    return None


@examples.command(short_help="ADCIRC 2D parameter estimation problem.")
@click.argument("data_file")
@click.option(
    "-n",
    "--num-components",
    type=int,
    default=2,
    help="Number of principal components to use. In this case set to 1 or 2.",
    show_default=True,
)
@click.option(
    "-t1",
    "--t-start",
    type=str,
    help="Time window(s) start times",
    multiple=True,
)
@click.option(
    "-t2",
    "--t-end",
    type=str,
    help="Time window(s) end times",
    multiple=True,
)
@click.option(
    "-lx",
    "--labels-x",
    type=str,
    help="Time x value to place time-window labels at.",
    multiple=True,
)
@click.option(
    "-ly",
    "--labels-y",
    type=float,
    help="Height y value to place time-window labels at.",
)
@click.option(
    "-p1",
    "--p1_ylims",
    type=float,
    help="ylimits to use for distribution plots of first param, for each window.",
    multiple=True,
)
@click.option(
    "-p2",
    "--p2_ylims",
    type=float,
    help="ylimits to use for distribution plots of second param, for each window.",
    multiple=True,
)
@click.option(
    "-s",
    "--sigma",
    default=1e-1,
    show_default=True,
    help='N(0, sigma) error added to true time series to produce "measurements".',
)
@click.option(
    "-mv",
    "--mesh-value",
    default="wind_speed_mult_0",
    show_default=True,
    help="Data over ADCIRC grid to be plotted.",
)
@click.option(
    "-mz",
    "--mesh-zoom",
    default="[[-72.48, 0.47], [40.70, 0.32]]",
    show_default=True,
    help="[[long_center, long_width], [latitude_center, latitude_width]] zoom box",
)
@click.option(
    "-mc",
    "--mesh-cb-cutoff",
    default=-10,
    show_default=True,
    help="Cutoff value for bathymetry plots so ignoring deeper waters.",
)
@click.option(
    "-p",
    "--plot_fig",
    default=["all"],
    multiple=True,
    help="Figures to plot.",
    show_default=True,
)
@click.pass_context
def adcirc_solve(
    ctx,
    data_file,
    num_components=1,
    t_start=["2018-01-11 01:00:00", "2018-01-04 11:00:00", "2018-01-07 00:00:00"],
    t_end=["2018-01-11 07:00:00", "2018-01-04 14:00:00", "2018-01-09 00:00:00"],
    labels_x=["2018-01-10 14:00:00", "2018-01-04 00:00:00", "2018-01-07 20:00:00"],
    labels_y=1.8,
    p1_ylims=[],
    p2_ylims=[],
    sigma=0.05,
    mesh_value="wind_speed_mult_0",
    mesh_zoom=None,
    mesh_cb_cutoff=-10,
    plot_fig=["all"],
):
    """
    Run ADCIRC Example

    Reproduces Figure 9 - 12
    """
    raw_data, adcirc_prob = load_adcirc_prob(
        data_file, std_dev=sigma, seed=ctx.obj["seed"]
    )
    time_windows = list(zip(t_start, t_end))
    labels = [[x, labels_y] for x in labels_x]
    ylims = list(zip(p1_ylims, p2_ylims))
    ylims = ylims if len(ylims) == len(time_windows) else None
    if "full_ts" in plot_fig or "all" in plot_fig:
        adcirc_ts_plot(
            adcirc_prob,
            time_windows=time_windows,
            wind_speeds=[raw_data["wind_speed"][0], raw_data["wind_speed"][1]],
            labels=labels,
            save_path=ctx.obj["save_path"],
            dpi=ctx.obj["dpi"],
            close_fig=False,
        )
    if "mesh" in plot_fig or "all" in plot_fig:
        tri_mesh_plot(
            raw_data["grid_data"],
            value=mesh_value,
            zoom=ast.literal_eval(mesh_zoom),
            colorbar_cutoff=mesh_cb_cutoff,
            save_path=ctx.obj["save_path"],
            dpi=ctx.obj["dpi"],
            close_fig=False,
        )

    t_res = []
    for i, t in enumerate(time_windows):
        ylim = None if ylims is None else ylims[i]
        res = adcirc_time_window(
            adcirc_prob,
            t,
            num_components=num_components,
            plot_figs=plot_fig,
            title=rf"$T_{{{i+1}}}$",
            ylims=ylim,
            save_path=ctx.obj["save_path"],
            dpi=ctx.obj["dpi"],
        )
        t_res.append(
            {
                "t_start": t[0],
                "t_end": t[1],
                "mud_pt": res.estimate(),
                "r": res.expected_ratio(),
            }
        )

    if len(time_windows) > 0:
        print(print_res(t_res, fields=["t_start", "t_end", "mud_pt", "r"]))

    if ctx.obj["show"]:
        plt.show()

    plt.close("all")


@examples.command(short_help="Reproduce figures from mud paper.")
@click.pass_context
def mud_paper(ctx):
    """Reproduce MUD Paper figures"""
    fpath = Path.cwd() / "mud_paper"
    fpath = Path(ctx.obj["save_path"]) if ctx.obj["save_path"] is not None else fpath
    fpath.mkdir(exist_ok=True)
    figs_path = fpath / "figures"
    figs_path.mkdir(exist_ok=True)
    data_path = fpath / "data"
    data_path.mkdir(exist_ok=True)

    def _set_seed():
        if ctx.obj["seed"] is not None:
            np.random.seed(ctx.obj["seed"])

    # Fig 1
    _set_seed()
    run_comparison_example(N_vals=[1], save_path=str(figs_path), dpi=ctx.obj["dpi"])
    plt.close("all")

    # Fig 2
    _set_seed()
    run_comparison_example(
        N_vals=[5, 10, 20], save_path=str(figs_path), dpi=ctx.obj["dpi"]
    )
    plt.close("all")

    # Fig 3
    _set_seed()
    run_contours("all", save_path=str(figs_path), dpi=ctx.obj["dpi"])
    plt.close("all")

    # Fig 4
    _set_seed()
    run_wme_covariance(save_path=str(figs_path), dpi=ctx.obj["dpi"])
    plt.close("all")

    # Fig 5
    _set_seed()
    lin_prob = {
        "A": [[1, 1]],
        "b": [[0]],
        "y": [[1]],
        "mean_i": [[0.25], [0.25]],
        "cov_i": [[1.0, -0.5], [-0.5, 0.5]],
        "cov_o": [[0.5]],
        "alpha": 1.0,
    }
    run_contours("comparison", save_path=str(figs_path), dpi=ctx.obj["dpi"], **lin_prob)
    plt.close("all")

    # Fig 6
    _set_seed()
    run_high_dim_linear(save_path=str(figs_path), dpi=ctx.obj["dpi"])
    plt.close("all")

    p_fname = "poisson_data.pkl"
    p_path = data_path / p_fname
    p_ds_url = f"https://github.com/mindthemath/mud-data/raw/main/{p_fname}"
    if not p_path.exists():
        if not fin_flag:
            download(p_ds_url)
            (Path.cwd() / p_fname).rename(p_path)
        else:
            res, p = run_fenics(
                1000,
                500,
                save_path=".",
                seed=ctx.obj["seed"],
            )
            print(f"{p}")
    plt.close("all")

    # Figs 7, 8
    _set_seed()
    res = run_2d_poisson_sol(
        data_file=p_path,
        seed=ctx.obj["seed"],
        plot_fig=["response", "qoi"],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")

    # Fig 9
    n_vals = [5, 50, 500]
    res = run_2d_poisson_trials(
        p_path,
        N_vals=n_vals,
        seed=ctx.obj["seed"],
        save_path=ctx.obj["save_path"],
        dpi=ctx.obj["dpi"],
    )
    runs = []
    for i, p in enumerate(res[1]):
        runs.append({"N": n_vals[i], "mud_pt": p.estimate(), "r": p.expected_ratio()})
    print(print_res(runs, fields=["N", "mud_pt", "r"]))
    plt.close("all")

    a_fname = "adcirc-si.pkl"
    a_path = data_path / a_fname
    a_ds_url = f"https://github.com/mindthemath/mud-data/raw/main/{a_fname}"
    if not (a_path).exists():
        download(a_ds_url)
        Path(a_fname).rename(a_path)

    raw_data, adcirc_prob = load_adcirc_prob(a_path, std_dev=0.05, seed=21)
    t1 = ["2018-01-11 01:00:00", "2018-01-11 07:00:00"]
    t2 = ["2018-01-04 11:00:00", "2018-01-04 14:00:00"]
    t3 = ["2018-01-07 00:00:00", "2018-01-09 00:00:00"]

    tri_mesh_plot(
        raw_data["grid_data"],
        zoom=[[-72.48, 0.47], [40.70, 0.32]],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    tri_mesh_plot(
        raw_data["grid_data"],
        value="DP",
        zoom=[[-72.5, 0.1], [40.85, 0.04]],
        colorbar_cutoff=-10,
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")

    # Fig 10
    adcirc_ts_plot(
        adcirc_prob,
        wind_speeds=raw_data["wind_speed"],
        time_windows=[t1, t2, t3],
        labels=[
            ["2018-01-10 14:00:00", 1.8],
            ["2018-01-04 00:00:00", 1.8],
            ["2018-01-07 20:00:00", 1.8],
        ],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")

    # Fig 11
    adcirc_time_window(
        adcirc_prob,
        t1,
        num_components=1,
        plot_figs="updated_dist",
        title=r"$T_1$",
        ylims=[65, 1600],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    adcirc_time_window(
        adcirc_prob,
        t1,
        num_components=2,
        plot_figs="updated_dist",
        title=r"$T_1$",
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")

    # Fig 12
    adcirc_time_window(
        adcirc_prob,
        t2,
        num_components=1,
        plot_figs="updated_dist",
        title=r"$T_2$",
        ylims=[40.0, 4000],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    adcirc_time_window(
        adcirc_prob,
        t2,
        num_components=2,
        plot_figs="updated_dist",
        title=r"$T_2$",
        ylims=[35, 10000],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")

    # Fig 13
    adcirc_time_window(
        adcirc_prob,
        t3,
        num_components=1,
        plot_figs="updated_dist",
        title=r"$T_3$",
        ylims=[30, 3000],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    adcirc_time_window(
        adcirc_prob,
        t3,
        num_components=2,
        plot_figs="updated_dist",
        title=r"$T_3$",
        ylims=[160, 14000],
        save_path=str(figs_path),
        dpi=ctx.obj["dpi"],
    )
    plt.close("all")
