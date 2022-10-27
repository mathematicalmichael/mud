"""
MUD vs MAP Comparison Example

Functions for running 1-dimensional polynomial inversion problem.
"""
import logging
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import norm  # type: ignore

from mud.base import BayesProblem, DensityProblem
from mud.examples.simple import polynomial_1D_data
from mud.plot import mud_plot_params, save_figure

plt.rcParams.update(mud_plot_params)

_logger = logging.getLogger(__name__)


def comparison_plot(
    d_prob: DensityProblem,
    b_prob: BayesProblem,
    space: str = "param",
    ax: plt.Axes = None,
    save_path: str = None,
    dpi: int = 500,
):
    """
    Generate plot comparing MUD vs MAP solution

    Parameters
    ----------
    d_prob : mud.base.DensityProblem
        DensityProblem object that has been solved already with
        d_prob.estimate() or another such method.
    b_prob : mud.base.BayesProblem
        BayesProblem object that has been solved already with
        b_prob.estimate() or another such method.
    space : str, default="param"
        What space to plot. Default is "param" to plot the parameter, or input,
        space, and thus the updated parameter distributions and associated
        MUD/MAP solutions. Any other value will plot the observable space, which
        includes the predicted distribution for the DensityProblem and the
        data-likelihood distribution for the BayesProblem.
    ax : matplotlib.pyplot.Axes, optional
        Existing matplotlib Axes object to plot onto. If none provided
        (default), then a figure is initialized.
    save_path : str, optional
        Path to save figure to.
    dpi : int
        If set to `save_path` is specified, then the resolution of the saved
        image to use.

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        Axes object that was plotted onto or created.
    """

    # Plot comparison plots of b_prob vs DCI solutions
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Parameters for initial and updated plots
    legend_fsize = 14
    tick_fsize = 18
    ylim_p = [-0.2, 5.5]
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

    if space == "param":
        d_prob.plot_param_space(
            ax=ax, in_opts=in_opts, up_opts=up_opts, win_opts=None, mud_opts=None
        )
        b_prob.plot_param_space(ax=ax, pr_opts=None, ps_opts=ps_opts, map_opts=None)

        # Format figure
        _ = ax.set_xlim([-1, 1])
        _ = ax.set_ylim(ylim_p)
        _ = ax.tick_params(axis="x", labelsize=tick_fsize)
        _ = ax.tick_params(axis="y", labelsize=tick_fsize)
        _ = ax.set_xlabel("$\\Lambda$", fontsize=1.25 * tick_fsize)
        _ = ax.legend(fontsize=legend_fsize, loc="upper left")
    else:
        d_prob.plot_obs_space(
            ax=ax,
            pr_opts=pr_opts,
            pf_opts=pf_opts,
            ob_opts=ob_opts,
            y_range=np.array([[-1, 1]]),
        )
        b_prob.plot_obs_space(ax=ax, ll_opts=None, pf_opts=psf_opts)  # ,
        # y_range=np.array([[0,1]]))

        # Format figure
        _ = ax.set_xlim([-1, 1])
        _ = ax.tick_params(axis="x", labelsize=tick_fsize)
        _ = ax.tick_params(axis="y", labelsize=tick_fsize)
        _ = ax.set_xlabel("$\\mathcal{D}$", fontsize=1.25 * tick_fsize)
        _ = ax.legend(fontsize=legend_fsize, loc="upper left")

    return ax


def run_comparison_example(
    p: int = 5,
    num_samples: int = 1000,
    mu: float = 0.25,
    sigma: float = 0.1,
    domain: List[int] = [-1, 1],
    N_vals: List[int] = [1, 5, 10, 20],
    latex_labels: bool = True,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = False,
):
    r"""
    Run MUD vs MAP Comparison Example

    Entry-point function for running simple polynomial example comparing
    Bayesian and Data-Consistent solutions. Inverse problem involves inverting
    the polynomial QoI map Q(lam) = lam^5.

    Parameters
    ----------
    p: int, default=5
        Power of polynomial in :ref:`QoI map<eq:q_poly>`.
    num_samples: int, default=100
        Number of :math:`\lambda` samples to generate from a uniform
        distribution over ``domain`` for solving inverse problem.
    mu: float, default=0.25
        True mean value of observed data.
    sigma: float, default=0.1
        Standard deviation of observed data.
    domain: :obj:`numpy.typing.ArrayLike`, default=[[-1, 1]]
        Domain to draw lambda samples from.
    N_vals: List[int], default=[1, 5, 10, 20]
        Values for N, the number of data-points to use to solve inverse
        problems, to use. Each N value will produce two separate plots.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool , default=False
        Set to True to close figure and only save it. Useful when running in
        notebook environments.

    Returns
    -------
    res: List[Tuple[:class:`mud.base.DensityProblem`, :class:`mud.base.BayesProblem`,
                    :class:`matplotlib.axes.Axes`]]
        List of Tuples of ``(d_prob, b_prob, ax)`` containing the resulting
        Density and Bayesian problem objects in ``d_prob`` and ``b_prob``,
        resp., and the matplotlib axis to which the results were plotted, for
        each N case run.
    """

    res = []
    for N in N_vals:
        lam, q_lam, data = polynomial_1D_data(
            p=p,
            N=N,
            domain=np.array([domain]),
            mu=mu,
            sigma=sigma,
            num_samples=num_samples,
        )

        d_prob = DensityProblem(lam, q_lam, domain=domain)
        d_prob.set_observed(norm(loc=np.mean(data), scale=sigma))
        d_prob.estimate()

        b_prob = BayesProblem(lam, q_lam, domain=domain)
        b_prob.set_likelihood(norm(loc=data, scale=sigma))
        b_prob.estimate()

        ax = comparison_plot(d_prob, b_prob, space="param")
        if N != 1:
            _ = ax.text(-0.75, 5, f"N = {N}", fontsize=22)
            _ = ax.set_ylim([-0.2, 28.0])
        else:
            ax.set_ylim([-0.2, 6])
        save_figure(
            f"bip-vs-sip-{N}.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )

        ax = comparison_plot(d_prob, b_prob, space="obs")
        if N != 1:
            _ = ax.text(-0.75, 5, f"N = {N}", fontsize=22)
            _ = ax.set_ylim([-0.2, 28.0])
        else:
            ax.set_ylim([-0.2, 4.5])
        save_figure(
            f"bip-vs-sip-pf-{N}.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )

        res.append([d_prob, b_prob, ax])

    return res
