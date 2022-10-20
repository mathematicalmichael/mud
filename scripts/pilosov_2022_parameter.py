import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from matplotlib import cm
from scipy.stats import norm

from mud.base import *
from mud.examples import (polynomial_1D, random_linear_problem,
                          random_linear_wme_problem)
from mud.util import rank_decomposition

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

plt.backend = "Agg"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["font.size"] = 16
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"

_logger = logging.getLogger(__name__)


def _save_fig(
    fname: str, save_path: str = None, dpi: int = 500, close_fig: bool = True
):
    """
    Save Figure Utility

    Utility to save figure to a given folder path if specified.

    Parameters
    ----------
    fname: str
        Name of image, with extension.
    save_path: str, optional
        Directory to save figure to. Assumed to exist. If not specified then the
        figure is saved to the current working directory.
    dpi: int, default=500
        Resolution to save image with.
    close_fig: bool, default=True
        Whether to close the figure after saving it.


    """
    if save_path is not None:
        fname = Path(save_path) / fname
        plt.savefig(str(fname), dpi=dpi)
    if close_fig:
        plt.close()

def comparison_plot(
        d_prob: DensityProblem,
        b_prob: BayesProblem,
        space: str='param',
        ax: plt.Axes= None,
        plot_version: int = 1,
        dpi: int=500,
        save_path: str = None, **kwargs):

    # Plot comparison plots of b_prob vs DCI solutions
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Parameters for initial and updated plots
    legend_fsize = 14
    tick_fsize = 18
    if plot_version == 1:
        ylim = [-0.2, 5.05]
        in_opts = {
            "color": "b",
            "linestyle": "--",
            "linewidth": 4,
            "label": "Initial/Prior",
        }
        up_opts = {"color": "k", "linestyle": "-.", "linewidth": 4, "label": "Updated"}
        ps_opts = {"color": "g", "linestyle": ":", "linewidth": 4, "label": "Posterior"}
        ob_opts = {
            "color": "r",
            "linestyle": "-",
            "linewidth": 4,
            "label": "$N(0.25,0.1^2)$",
        }
        pr_opts = {
            "color": "b",
            "linestyle": "-.",
            "linewidth": 4,
            "label": "PF of Initial",
        }
        pf_opts = {
            "color": "k",
            "linestyle": "--",
            "linewidth": 4,
            "label": "PF of Updated",
        }
        psf_opts = {
            "color": "g",
            "linestyle": ":",
            "linewidth": 4,
            "label": "PF of Posterior",
        }

    else:
        ylim = [-0.2, 6.0]
        in_opts = {
            "color": "b",
            "linestyle": "--",
            "linewidth": 4,
            "label": "$\\pi_{in}(\\lambda)=\\pi_{prior}(\\lambda) = \\mathcal{U}([0,1])$",
        }
        up_opts = {
            "color": "k",
            "linestyle": "-.",
            "linewidth": 4,
            "label": "$\\pi_{up}(Q(\\lambda))$",
        }
        ps_opts = {
            "color": "g",
            "linestyle": ":",
            "linewidth": 4,
            "label": "$\\pi_{post}(Q(\\lambda))$",
        }
        ob_opts = {
            "color": "r",
            "linestyle": "-",
            "linewidth": 4,
            "label": "$\\pi_{ob}(Q(\\lambda))=\\pi_{like}(d|\\lambda)=N(0.25,0.1^2)$",
        }
        pr_opts = {
            "color": "b",
            "linestyle": "-.",
            "linewidth": 4,
            "label": "$\\pi_{pr}(Q(\\lambda))$",
        }
        pf_opts = {
            "color": "k",
            "linestyle": "--",
            "linewidth": 4,
            "label": "PF of $\\pi_{up}(Q(\\lambda))$",
        }
        psf_opts = {
            "color": "g",
            "linestyle": ":",
            "linewidth": 4,
            "label": "PF of $\\pi_{post}(Q(\\lambda))$",
        }

    if space=='param':
        # Plot figure to created axis - note this will solve the SIP problem
        d_prob.plot_param_space(ax=ax, in_opts=in_opts, up_opts=up_opts, win_opts=None)
        b_prob.plot_param_space(ax=ax, pr_opts=None, ps_opts=ps_opts)

        # Format figure
        _ = ax.set_xlim([-1, 1])
        _ = ax.set_ylim(ylim)
        _ = ax.tick_params(axis="x", labelsize=tick_fsize)
        _ = ax.tick_params(axis="y", labelsize=tick_fsize)
        _ = ax.set_xlabel("$\\Lambda$", fontsize=1.25 * tick_fsize)
        _ = ax.legend(fontsize=legend_fsize, loc="upper left")
    else:
        # b_prob - Plot data-likelihood and and push-forward of posterior in observable space D
        d_prob.plot_obs_space(ax=ax, pr_opts=pr_opts, pf_opts=pf_opts, ob_opts=ob_opts)
        b_prob.plot_obs_space(ax=ax, ll_opts=None, pf_opts=psf_opts)

        # Format figure
        _ = ax.set_xlim([-1, 1])
        _ = ax.set_ylim(ylim)
        _ = ax.tick_params(axis="x", labelsize=tick_fsize)
        _ = ax.tick_params(axis="y", labelsize=tick_fsize)
        _ = ax.set_xlabel("$\\mathcal{D}$", fontsize=1.25 * tick_fsize)
        _ = ax.legend(fontsize=legend_fsize, loc="upper left")

    return ax


def run_comparison_example(
    N_vals: List[int] = [1, 5, 10, 20],
    plot_version: int = 1,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = True,
    **kwargs,
):

    """
    Run Simple Example

    Entry-point function for running simple polynomial example comparing Bayesian
    and Data-Consistent solutions.

    Parameters
    ----------
    plot_version: int, default=1
        Version of plot labels to use. Set to 1 for english names in legends.
        Any other value will result in LaTex symbols for the legend.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    kwargs: dict, optional
        Additional keyword arguments will be passed to
    :func:`mud.examples.polynomial_1D` to override defaults for problem set-up.

    Returns
    -------
    res: Tuple[:class:`mud.base.DensityProblem`, :class:`mud.base.BayesProblem`, :class:`matplotlib.axes.Axes`]
        Tuple of ``(d_prob, b_prob, ax)`` containing the resulting Density and
        Bayesian problem classes in ``d_prob`` and ``b_prob``, resp., and the
        matplotlib axis to which the results were plotted.
    """

    for N in N_vals:
        # Number of samples to use, and assumed observed normal distribution parameters
        def_args = {
            "p": 5,
            "n_samples": int(1e3),
            "mu": 0.25,
            "sigma": 0.1,
            "domain": np.array([[-1, 1]]),
            "N": N,
        }
        def_args.update(kwargs)
        lam, q_lam, data = polynomial_1D(
            **def_args
        )  # p=p, N=N, domain=domain, mu=mu, sigma=sigma, num_data=1)

        # Set up and solve DCI problem
        d_prob = DensityProblem(lam, q_lam, domain=def_args["domain"])
        d_prob.set_observed(norm(loc=np.mean(data), scale=def_args["sigma"]))
        d_prob.estimate()

        # Set up and solve Bayesian problem
        bayes = BayesProblem(lam, q_lam, domain=def_args["domain"])
        bayes.set_likelihood(norm(loc=data, scale=def_args["sigma"]))
        bayes.estimate()

        ax = comparison_plot(d_prob, bayes, space="param", plot_version=plot_version)
        if N != 1:
            _ = ax.text(-0.75, 5, f"N = {N}", fontsize=22)
            _ = ax.set_ylim([-0.2, 28.0])
        _save_fig(
            f"bip-vs-sip-{N}.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )

        ax = comparison_plot(d_prob, bayes, space="obs", plot_version=plot_version)
        if N != 1:
            _ = ax.text(-0.75, 5, f"N = {N}", fontsize=22)
            _ = ax.set_ylim([-0.2, 28.0])
        _save_fig(
            f"bip-vs-sip-pf-{N}.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )


def run_contours(
    save_path: str = None, dpi: int = 500, close_fig: bool = True, **kwargs
):
    """ """

    # Build linear problem - Overwrite defaults with anything in **kwargs
    def_args = {
        "A": np.array([[1, 1]]),
        "b": np.array([[0]]),
        "y": np.array([[1]]),
        "mean_i": np.array([[0.25, 0.25]]).T,
        "cov_i": np.array([[1, -0.25], [-0.25, 0.5]]),
        "cov_o": np.array([[1]]),
    }
    def_args.update(kwargs)
    lin_prob = LinearGaussianProblem(**def_args)

    mud_sol, map_sol, ls_sol = (
        lin_prob.solve("mud"),
        lin_prob.solve("map"),
        lin_prob.solve("ls"),
    )

    # Plot data mismatch contours
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Plot contour map for data mismatch term
    lin_prob.plot_fun_contours(
        ax=ax, terms="data", levels=50, cmap=cm.viridis, alpha=1.0
    )

    # Plot solution contour for 2-1 map, going through least squares solution as the reference point.
    lin_prob.plot_contours(
        ax=ax,
        annotate=True,
        note_loc=[0.1, 0.9],
        label="Solution Contour",
        plot_opts={"color": "r"},
        annotate_opts={"fontsize": 20, "backgroundcolor": "w"},
    )

    # Plot solution contour for 2-1 map, going through least squares solution as the reference point.
    ax.axis("equal")

    _ = ax.set_xlim([0, 1])
    _ = ax.set_ylim([0, 1])

    _save_fig(
        "data_mismatch_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
    )

    # Plot Regularization
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lin_prob.plot_fun_contours(
        ax=ax, terms="reg", levels=50, cmap=cm.viridis, alpha=1.0
    )
    lin_prob.plot_sol(
        ax=ax,
        point="initial",
        label="Initial Mean",
        pt_opts={"color": "k", "s": 100, "marker": "o", "label": "MUD", "zorder": 10},
    )
    _ = ax.axis([0, 1, 0, 1])
    _save_fig("tikonov_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig)

    # Plot Modified Regularization
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lin_prob.plot_fun_contours(
        ax=ax, terms="reg_m", levels=50, cmap=cm.viridis, alpha=1.0
    )
    lin_prob.plot_sol(
        ax=ax,
        point="initial",
        label="Initial Mean",
        pt_opts={"color": "k", "s": 100, "marker": "o", "label": "MUD", "zorder": 10},
    )
    _ = ax.axis([0, 1, 0, 1])
    _save_fig(
        "consistent_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
    )

    # Plot contour map for functional for Bayesian Posteriror
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lin_prob.plot_fun_contours(ax=ax, terms="bayes", levels=50, cmap=cm.viridis)

    # Plot initial mean point
    lin_prob.plot_sol(
        ax=ax,
        point="initial",
        pt_opts={"color": "k", "s": 100, "marker": "o", "label": "MUD", "zorder": 20},
    )

    # Plot Least Squares Solution
    lin_prob.plot_sol(
        ax=ax,
        point="ls",
        label="Least Squares",
        note_loc=[0.49, 0.55],
        pt_opts={"color": "xkcd:blue", "s": 100, "marker": "d", "zorder": 10},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot MAP Solution
    lin_prob.plot_sol(
        ax=ax,
        point="map",
        label="MAP",
        pt_opts={
            "color": "tab:orange",
            "s": 100,
            "linewidths": 3,
            "marker": "x",
            "zorder": 10,
        },
        ln_opts=None,
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot solution contour for 2-1 map, going through least squares solution as the reference point.
    lin_prob.plot_contours(
        ax=ax,
        annotate=False,
        note_loc=[0.1, 0.9],
        label="Solution Contour",
        plot_opts={"color": "r"},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    _ = ax.axis([0, 1, 0, 1])
    _save_fig(
        "classical_solution.png", save_path=save_path, dpi=dpi, close_fig=close_fig
    )

    # Plot contour map for functional for Updated Density
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    lin_prob.plot_fun_contours(ax=ax, terms="dc", levels=50, cmap=cm.viridis)

    # Plot initial mean point
    lin_prob.plot_sol(
        ax=ax,
        point="initial",
        pt_opts={"color": "k", "s": 100, "marker": "o", "label": "MUD", "zorder": 20},
    )

    # Plot Least Squares Solution and line from initial to it
    lin_prob.plot_sol(
        ax=ax,
        point="ls",
        label="Least Squares",
        note_loc=[0.49, 0.55],
        pt_opts={"color": "k", "s": 100, "marker": "d", "zorder": 10},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot MUD Solution
    lin_prob.plot_sol(
        point="mud",
        ax=ax,
        label="MUD",
        pt_opts={"color": "k", "s": 100, "marker": "*", "zorder": 10},
        ln_opts={"color": "k", "marker": "*", "lw": 1, "zorder": 10},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot solution contour for 2-1 map, going through least squares solution as the reference point.
    lin_prob.plot_contours(
        ax=ax,
        annotate=False,
        note_loc=[0.1, 0.9],
        label="Solution Contour",
        plot_opts={"color": "r"},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    ax.axis("equal")
    _ = ax.axis([0, 1, 0, 1])
    _save_fig(
        "consistent_solution.png", save_path=save_path, dpi=dpi, close_fig=close_fig
    )


def run_comparison_contours(
    save_path: str = None, dpi: int = 500, close_fig: bool = True, **kwargs
):
    """
    Run Comparison Plot Example

    Produces Figure 5 in paper, comparing MUD, MAP, and least squares solution
    contours for the linear gaussian problem:


    """

    # Build linear problem - Overwrite defaults with anything in **kwargs
    def_args = {
        "A": np.array([[1, 1]]),
        "b": np.array([[0]]),
        "y": np.array([[1]]),
        "mean_i": np.array([[0.25, 0.25]]).T,
        "cov_i": np.array([[1, -0.5], [-0.5, 0.5]]),
        "cov_o": np.array([[0.5]]),
        "alpha": 1.0,
    }
    def_args.update(kwargs)
    lin_prob = LinearGaussianProblem(**def_args)

    mud_sol, map_sol, ls_sol = (
        lin_prob.solve("mud"),
        lin_prob.solve("map"),
        lin_prob.solve("ls"),
    )

    # Plot comparison contour plot of map vs mud solution
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    lin_prob.plot_fun_contours(ax=ax, terms="bayes", levels=50, cmap=cm.viridis)

    # Plot initial mean point
    lin_prob.plot_sol(
        ax=ax,
        point="initial",
        pt_opts={"color": "k", "s": 100, "marker": "o", "label": "MUD", "zorder": 10},
    )

    # Plot Least Squares Solution and line from initial to it
    lin_prob.plot_sol(
        ax=ax,
        point="ls",
        label="Least Squares",
        note_loc=[0.49, 0.55],
        pt_opts={"color": "k", "s": 100, "marker": "d", "zorder": 10},
    )

    # Plot MAP Solution
    lin_prob.plot_sol(
        ax=ax,
        point="map",
        label="MAP",
        pt_opts={
            "color": "tab:orange",
            "s": 100,
            "linewidth": 3,
            "marker": "x",
            "zorder": 10,
        },
        ln_opts=None,
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot MUD Solution
    lin_prob.plot_sol(
        point="mud",
        ax=ax,
        label="MUD",
        pt_opts={"color": "k", "s": 100, "linewidth": 3, "marker": "*", "zorder": 10},
        ln_opts={"color": "k", "marker": "*", "lw": 1, "zorder": 10},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot solution contour for 2-1 map, going through least squares solution as the reference point.
    lin_prob.plot_contours(
        ax=ax,
        annotate=False,
        note_loc=[0.1, 0.9],
        label="Solution Contour",
        plot_opts={"color": "r"},
        annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
    )

    # Plot "Reference" parameter
    ax.scatter([0.7], [0.3], color="k", s=100, marker="s", zorder=11)
    _ = ax.axis([0, 1, 0, 1])
    _save_fig(
        "map_compare_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
    )


def run_wme_covariance_example(
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = True,
    dim_input: int = 20,
    dim_output: int = 5,
    sigma: float = 1e-1,
    Ns: List[int] = [10, 100, 1000, 10000],
    seed: int = 21,
):

    if seed is not None:
        np.random.seed(seed)

    initial_cov = np.diag(np.sort(np.random.rand(dim_input))[::-1] + 0.5)
    lam_ref = np.random.randn(dim_input).reshape(-1, 1)

    # Create operator list of random linear problems
    operator_list, data_list, _ = random_linear_wme_problem(
        lam_ref,
        [0] * dim_output,  # noiseless data bc we want to simulate multiple trials
        num_qoi=dim_output,
        num_observations=[max(Ns)]
        * dim_output,  # want to iterate over increasing measurements
        dist="norm",
        repeated=True,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    index_values = np.arange(dim_input) + 1
    lines = ["solid", "dashed", "dashdot", "dotted"]

    noise_draw = np.random.randn(dim_output, max(Ns)) * sigma
    for j, N in enumerate(Ns):

        # Sub-select over increasing measurements and add noise
        _oper_list = operator_list  # [M[0:N, :] for M in operator_list]
        _d = np.array([y[0:N] for y in data_list]) + noise_draw[:, 0:N]
        _data_list = _d.tolist()

        # Build Linear WME problem
        linear_wme_prob = LinearWME(_oper_list, _data_list, sigma, cov_i=initial_cov)

        # Solve for updated covariance
        up_cov = linear_wme_prob.updated_cov()
        up_sdvals = sp.linalg.svdvals(up_cov)

        ax.scatter(
            index_values,
            up_sdvals,
            marker="o",
            s=200,
            facecolors="none",
            edgecolors="k",
        )

        ax.plot(
            index_values,
            up_sdvals,
            label=f"$N={N:1.0E}$",
            alpha=1,
            lw=3,
            ls=lines[j % len(lines)],
            c="k",
        )

    _ = ax.set_yscale("log")
    _ = ax.set_xticks(index_values)
    _ = ax.set_xticklabels(ax.get_xticks(), rotation=0)
    _ = ax.set_xlabel("Index")
    _ = ax.set_ylabel("Eigenvalue")
    _ = ax.legend(loc="lower left")

    _save_fig(
        "lin-meas-cov-sd-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )


def run_high_dim_linear(
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = True,
    dim_input: int = 100,
    dim_output: int = 100,
    seed: int = 21,
):

    lam_ref, randn_high_dim = random_linear_problem(
        dim_input=dim_input, dim_output=dim_output, seed=seed
    )
    A_ranks = rank_decomposition(randn_high_dim.A)

    c = np.linalg.norm(lam_ref)
    err = lambda xs: [np.linalg.norm(x - lam_ref) / c for x in xs]

    dim_errs = []
    rank_errs = []
    alpha_list = [1.0, 0.1, 0.001]
    for alpha in alpha_list:
        r_errs = []
        d_errs = []
        randn_high_dim.alpha = alpha
        for dim in range(1, dim_output + 1, 1):
            # Solve inverse problem for dimensional subset problem
            # Tuple is returned, with (mud, map, least-squares) solutions
            dim_solutions = randn_high_dim.solve(method="all", output_dim=dim)

            # Construct Rank k Linear Problem
            y_rank = A_ranks[dim - 1] @ lam_ref + randn_high_dim.b
            rank_prob = LinearGaussianProblem(
                A_ranks[dim - 1],
                randn_high_dim.b,
                y_rank,
                randn_high_dim.mean_i,
                randn_high_dim.cov_i,
                alpha=alpha,
            )
            rank_solutions = rank_prob.solve(method="all")

            # Compute errors
            d_errs.append(np.array(err(dim_solutions)))
            r_errs.append(np.array(err(rank_solutions)))

        dim_errs.append(np.array(d_errs))
        rank_errs.append(np.array(r_errs))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.arange(1, dim_output, 1)
    for idx, alpha in enumerate(alpha_list):
        # Plot convergence for MUD Solutions
        ax.plot(x, dim_errs[idx][:, 0][:-1], label="MUD", c="k", lw=10)

        # Plot convergence plot for MAP Solutions - Annotate for different alphas
        ax.plot(x, dim_errs[idx][:, 1][:-1], label="MAP", c="r", ls="--", lw=5)
        ax.annotate(
            f"$\\alpha$={alpha:1.2E}",
            (100, max(dim_errs[idx][:, 1][-1], 0.01)),
        )

        # Plot convergence for Least Squares Solutions
        ax.plot(
            x,
            dim_errs[idx][:, 2][:-1],
            label="LSQ",
            c="xkcd:light blue",
            ls="-",
            lw=5,
            zorder=10,
        )

    # Label plot
    _ = ax.set_title(
        "Convergence for Various $\\Sigma_{init} = \\alpha \\Sigma$",
    )
    _ = ax.set_ylim(0, 1.0)
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_xlabel("Dimension of Output Space")
    _ = ax.legend(["MUD", "MAP", "Least Squares"])

    _save_fig(
        "lin-dim-cov-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for idx, alpha in enumerate(alpha_list):
        # Plot convergence for MUD Solutions
        ax.plot(x, rank_errs[idx][:, 0][:-1], label="MUD", c="k", lw=10)

        # Plot convergence plot for MAP Solutions - Annotate for different alphas
        ax.plot(x, rank_errs[idx][:, 1][:-1], label="MAP", c="r", ls="--", lw=5)
        ax.annotate(
            f"$\\alpha$={alpha:1.2E}",
            (100, max(rank_errs[idx][:, 1][-1], 0.01)),
        )

        # Plot convergence for Least Squares Solutions
        ax.plot(
            x,
            rank_errs[idx][:, 2][:-1],
            label="LSQ",
            c="xkcd:light blue",
            ls="-",
            lw=5,
            zorder=10,
        )

    # Label plot
    _ = ax.set_title(
        "Convergence for Various $\\Sigma_{init} = \\alpha \\Sigma$",
    )
    _ = ax.set_ylim(0, 1.0)
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_xlabel("Rank(A)")
    _ = ax.legend(["MUD", "MAP", "Least Squares"])

    _save_fig(
        "lin-rank-cov-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )

# TODO: Add argparse and inputs to control which examples get run and
# where outputs are placed
if __name__ == "__main__":

    cwd = Path.cwd()
    fig_dir = cwd / 'figures'
    contour_dir = fig_dir / 'contours'
    comparison_dir = fig_dir / 'comparison'
    lin_dir = fig_dir / 'lin'
    ode_dir = fig_dir / 'ode'
    pde_dir = fig_dir / 'pde'
    data_dir = fig_dir / 'data'

    fig_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    contour_dir.mkdir(exist_ok=True)
    lin_dir.mkdir(exist_ok=True)
    ode_dir.mkdir(exist_ok=True)
    pde_dir.mkdir(exist_ok=True)

    run_comparison_example(save_path=str(comparison_dir))
    run_contours(save_path=str(contour_dir))
    run_comparison_contours(save_path=str(contour_dir))
    run_wme_covariance_example(save_path=str(lin_dir))
    run_high_dim_linear(save_path=str(lin_dir))
