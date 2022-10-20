"""
MUD Examples CLI

CLI for running MUD examples
"""
import pdb
import click
import json
import numpy as np
from typing import List
import matplotlib.pyplot as plt

from mud.util import print_res
from .comparison import run_comparison_example
from .linear import run_contours, run_wme_covariance, run_high_dim_linear
from .poisson import run_2d_poisson_sol, run_2d_poisson_trials

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


@click.group(short_help="MUD examples problems")
@click.option("-s/-ns", "--show/--no-show", default=True,
              help="Whether to show figures or not.", show_default=True)
@click.option("--seed", default=None, type=int,
              help="Seed for fixing results.")
@click.pass_context
def examples(ctx, show, seed):
    ctx.ensure_object(dict)
    ctx.obj['show'] = show
    ctx.obj['seed'] = seed
    if seed is not None:
        np.random.seed(seed)
    pass

@examples.command(short_help="MUD vs MAP comparison example.")
@click.option("-p", "--power", default=5,
              help="Power of exponential function to invert.")
@click.option("-n", "--num_samples", default=1000,
              help="Number of samples to use to solve inverse problems")
@click.option("-m", "--mu", default=0.25,
              help="True observed value.")
@click.option("-s", "--sigma", default=0.1,
              help="Assumed noise in measured data.")
@click.option("-d", "--domain", default=[-1, 1], multiple=True,
              help="Assumed domain of possible values for lambda.")
@click.option("--n-vals", default=[1, 5, 10, 20], multiple=True,
              help="".join(["Values for N, the number of data-points to use to",
              " solve inverse problems, to use. Each N value will produce",
              "a separate plot."]), show_default=True)
@click.option("-l/-nl", "--latex-labels/--no-latex-labels", default=True,
              help="Whether to use latex labels in plot.", show_default=True)
@click.option("--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
@click.pass_context
def comparison(
        ctx,
        n_vals: List[int] = [1, 5, 10, 20],
        latex_labels: bool = True,
        save_path: str = None,
        dpi: int = 500,
        power: int = 5,
        num_samples : int = 1000,
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
    res = run_comparison_example(N_vals=n_vals, latex_labels=latex_labels,
                              save_path=save_path, dpi=dpi,
                              p=power, num_samples=num_samples, mu=mu, sigma=sigma,
                              domain=domain)
    if ctx.obj['show']:
        plt.show()

    plt.close('all')


@examples.command(short_help="MUD vs MAP linear solution contour plots.")
@click.option("-f", "--lin_prob_file", default=None,
              help="Path to json config file for linear problem.",
              show_default=True)
@click.option("-p", "--plot_fig", default=['all'], multiple=True,
              help="Figures to plot.", show_default=True)
@click.option("-s", "--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("-d", "--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
@click.pass_context
def contours(
        ctx,
        lin_prob_file = None,
        plot_fig = ['all'],
        save_path = None,
        dpi = 500,
        ):
    """

    """
    if lin_prob_file is not None:
        with open(lin_prob_file, 'r') as fp:
            lin_prob = json.load(fp)

        for k in lin_prob.keys():
            if k in ['A', 'cov_i']:
                lin_prob[k] = np.array(lin_prob[k])
            elif k in ['b', 'y', 'mean_i', 'cov_o']:
                lin_prob[k] = np.array([lin_prob[k]])
    else:
        lin_prob = {}

    _ = run_contours(plot_fig, save_path=save_path, dpi=dpi, **lin_prob)

    if ctx.obj['show']:
        plt.show()

    plt.close('all')

@examples.command(short_help="Spectral properties of updated covariance for linear WME Maps.")
@click.option("-i", "--dim_input", default=20, show_default=True,
              help="Input dimension of linear map (number of rows in A).")
@click.option("-o", "--dim_output", default=5, show_default=True,
              help="Output dimension of linear map (number of columns in A).")
@click.option("-s", "--sigma", default=1e-1, show_default=True,
              help="N(0, sigma) error added to produce \"measurements\" from linear operator.")
@click.option("-n", "--num_data", default=[10, 100, 1000, 10000],
              help="".join(["List of number of data points to collect in",
                            " constructing Q_WME map to view how the spectral ",
                            "properties of the updated covariance change as ",
                            "more data is included in the Q_WME map."]),
              show_default=True, multiple=True)
@click.option("--seed", default=None, show_default=True,
              help="Random seed to use to make reproducible results.")
@click.option("--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
@click.pass_context
def wme_covariance(
        ctx,
        dim_input: int = 20,
        dim_output: int = 5,
        sigma: float = 1e-1,
        num_data = [10, 100, 1000, 10000],
        seed: int = None,
        save_path: str = None,
        dpi: int = 500,
):
    """
    Weighted Mean Error Map Updated Covariance

    Reproduces Figure 4 from [ref], showing the spectral properties of the
    updated covriance for a the Weighted Mean Error map on a randomly
    generated linear operator as more data from repeated measurements is used
    to constructthe QoI map.
    """
    _ = run_wme_covariance(
            dim_input=dim_input,
            dim_output=dim_output,
            sigma=sigma,
            Ns=num_data,
            seed=seed,
            save_path=save_path,
            dpi=dpi,
            close_fig=False)

    if ctx.obj['show']:
        plt.show()

    plt.close('all')

@examples.command(short_help="MUD/MAP/Least squares convergence for increasing dimension and rank of linear maps.")
@click.option("-i", "--dim_input", default=100, show_default=True,
              help="Input dimension of linear map (number of rows in A).")
@click.option("-o", "--dim_output", default=100, show_default=True,
              help="Output dimension of linear map (number of columns in A).")
@click.option("--seed", default=None, show_default=True,
              help="Random seed to use to make reproducible results.")
@click.option("--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
@click.pass_context
def high_dim_linear(
        ctx,
        dim_input=100,
        dim_output=100,
        seed: int = None,
        save_path: str = None,
        dpi: int = 500,
):
    """
    Run High Dimension Linear Example

    Reproduces Figure 6 from [ref], showing the relative error between the true
    parameter value and the MUD, MAP and least squares solutions to linear
    gaussian inversion problems for increasing dimension and rank of a randomly
    generated linear map A.
    """
    res = run_high_dim_linear(dim_input=dim_input,
                              dim_output=dim_output,
                              seed=ctx.obj['seed'],
                              save_path=save_path,
                              dpi=dpi,
                              close_fig=False)
    if ctx.obj['show']:
        plt.show()

    plt.close('all')

@examples.command(short_help="Poisson 2D parameter estimation problem solution.")
@click.argument("data_file")
@click.option("-s", "--sigma", default=1e-1, show_default=True,
              help="N(0, sigma) error added to true solution surface to produce \"measurements\".")
@click.option("--seed", default=None, show_default=True,
              help="Random seed to use to make reproducible results.")
@click.option("-p", "--plot_fig", default=['all'], multiple=True,
              help="Figures to plot.", show_default=True)
@click.option("--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
@click.pass_context
def poisson_solve(
        ctx,
        data_file,
        sigma=0.05,
        seed=None,
        plot_fig=['all'],
        save_path=None,
        dpi=500,
):
    """
    Run Poisson Example

    Reproduces Figure 7 and 8 from [ref].
    """
    res = run_2d_poisson_sol(
        data_file=data_file,
        sigma=sigma,
        seed=seed,
        plot_fig=plot_fig,
        save_path=save_path,
        dpi=dpi,
        close_fig=False)

    mud_sol = res[1].estimate()
    print(mud_sol)

    if ctx.obj['show']:
        plt.show()

    plt.close('all')

@examples.command(short_help="Poisson 2D parameter estimation problem solution.")
@click.argument("data_file")
@click.option("-n", "--n_vals", default=[5, 50, 500], multiple=True,
              help="".join(["Values for N, the number of sensors to use to",
              " solve inverse problems, to use. Each N value will produce",
              "two separate plots, one for each parameter."]),
              show_default=True)
@click.option("-y1", "--ylim1", default=[-0.1, 3.5], multiple=True,
              help="".join(["y limits to use in plot for first parameter space.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-y2", "--ylim2", default=[-0.1, 2.5], multiple=True,
              help="".join(["y limits to use in plot for second param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-x1", "--xlim1", default=[-4.5, 0.5], multiple=True,
              help="".join(["x limits to use in plot for first param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-x2", "--xlim2", default=[-4.5, 0.5], multiple=True,
              help="".join(["x limits to use in plot for second param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-s", "--sigma", default=1e-1, show_default=True,
              help="N(0, sigma) error added to true solution surface to produce \"measurements\".")
@click.option("--seed", default=None, show_default=True,
              help="Random seed to use to make reproducible results.")
@click.option("-p", "--plot_fig", default=['all'], multiple=True,
              help="Figures to plot.", show_default=True)
@click.option("--save-path", default=None,
              help="Path to save figures to.", show_default=True)
@click.option("--dpi", default=500,
              help="Resolution in dpi to use for output images.",
              show_default=True)
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
        seed=None,
        plot_fig=['all'],
        save_path=None,
        dpi=500,
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
            seed=seed,
            save_path=save_path,
            dpi=dpi,
            close_fig=False)
    runs = []
    for i, p in enumerate(res[1]):
        runs.append({'N': n_vals[i], 'mud_pt': p.estimate(), 'r': p.exp_r()})
    print(print_res(runs, fields=['N', 'mud_pt', 'r']))

    if ctx.obj['show']:
        plt.show()

    plt.close('all')


@examples.command(short_help="Generate Poisson problem dataset using FEniCS.")
@click.argument("num_samples", type=int)
@click.argument("num_sensors", type=int)
@click.option("--mins", default=[-4, -4], multiple=True, show_default=True,
              help="".join(["Minimum value for input domain ranges. Note: ",
                            "length of vector determines dimension."]))
@click.option("--maxs", default=[0, 0], multiple=True, show_default=True,
              help="".join(["Minimum value for input domain ranges. Note: ",
                            "length of vector determines dimension."]))
@click.option("--sensor_low", default=[0, 0], multiple=True, show_default=True,
              help="".join(["x, y minimum values in response surface for sensor"
                            "locations"]))
@click.option("--sensor_high", default=[1, 1], multiple=True, show_default=True,
              help="".join(["x, y minimum values in response surface for sensor"
                            "locations"]))
@click.option("-y1", "--ylim1", default=[-0.1, 3.5], multiple=True,
              help="".join(["y limits to use in plot for first parameter space.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-y2", "--ylim2", default=[-0.1, 2.5], multiple=True,
              help="".join(["y limits to use in plot for second param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-x1", "--ylim1", default=[-4.5, 0.5], multiple=True,
              help="".join(["x limits to use in plot for first param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-x2", "--xlim2", default=[-4.5, 0.5], multiple=True,
              help="".join(["x limits to use in plot for second param plot.",
                            " Set to something consistent so that plots for ",
                            "all N values will be clearly visible."]),
              show_default=True)
@click.option("-g", "--gamma", default=-3, show_default=True, type=int,
              help="Parameter conroling min value of true boundary condition.")
@click.option("--seed", default=None, show_default=True,
              help="Random seed to use to make reproducible results.")
@click.option("--save_dir", default='.',
              help="Path to save data generated to.", show_default=True)
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
        seed=None,
        save_dir='.',
):
    """
    Poisson Forward Solver using Fenics
    """
    try:
        from .fenics import run_fenics
    except Exception as e:
        raise ModuleNotFoundError(f"Fenics package not found - {e}")

    res, p = run_fenics(
            num_samples,
            num_sensors,
            mins=mins,
            maxs=maxs,
            sensor_low=sensor_low,
            sensor_high=sensor_high,
            gamma=gamma,
            save_path=save_dir,
            seed=seed)

    print(f'{p}')

    return res
