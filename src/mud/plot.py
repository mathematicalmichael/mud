"""
MUD Plotting Module

Plotting utility functions for visualizing data-sets and distributions related
to algorithm implemented within the MUD library.

Functions
---------

"""

import numpy as np
from matplotlib import pyplot as plt  # type: ignore

from mud.base import BayesProblem, DensityProblem
from mud.util import null_space

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{bm}"

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"


def comparison_plot(
    d_prob: DensityProblem,
    b_prob: BayesProblem,
    space: str = "param",
    ax: plt.Axes = None,
    plot_version: int = 1,
    dpi: int = 500,
    save_path: str = None,
    **kwargs
):

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

    if space == "param":
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


def plotChain(mud_chain, ref_param, color="k", s=100):
    num_steps = len(mud_chain)
    current_point = mud_chain[0]
    plt.scatter(current_point[0], current_point[1], c="b", s=s)
    for i in range(0, num_steps):
        next_point = mud_chain[i]
        points = np.hstack([current_point, next_point])
        plt.plot(points[0, :], points[1, :], c=color)
        current_point = next_point

    plt.ylim([0, 1])
    plt.xlim([0, 1])
    #     plt.axis('off')
    plt.scatter(ref_param[0], ref_param[1], c="r", s=s)


def plot_contours(
    A, ref_param, subset=None, color="k", ls=":", lw=1, fs=20, w=1, s=100, **kwds
):
    if subset is None:
        subset = np.arange(A.shape[0])
    A = A[np.array(subset), :]
    numQoI = A.shape[0]
    AA = np.hstack([null_space(A[i, :].reshape(1, -1)) for i in range(numQoI)]).T
    for i, contour in enumerate(subset):
        xloc = [ref_param[0] - w * AA[i, 0], ref_param[1] + w * AA[i, 0]]
        yloc = [ref_param[0] - w * AA[i, 1], ref_param[1] + w * AA[i, 1]]
        plt.plot(xloc, yloc, c=color, ls=ls, lw=lw, **kwds)
        plt.annotate("%d" % (contour + 1), (xloc[0], yloc[0]), fontsize=fs)


def make_2d_normal_mesh(N=50, window=1):
    X = np.linspace(-window, window, N)
    Y = np.linspace(-window, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)


def make_2d_unit_mesh(N=50, window=1):
    X = np.linspace(0, window, N)
    Y = np.linspace(0, window, N)
    X, Y = np.meshgrid(X, Y)
    XX = np.vstack([X.ravel(), Y.ravel()]).T
    return (X, Y, XX)


def plot2d_pca(
    X_train: np.typing.ArrayLike,
    idxs: np.typing.ArrayLike = [0, 1],
    ax: plt.Axes = None,
    label: bool = True,
    **kwargs
):
    """
    Plot PCA Trained Data

    Plots 2D scatter plot of trained PCA data-set from `apply_pca` method. By
    default plots the first two components on 2-dimensional x-y grid, but can
    control which components get plotted with the `idxs` argument.

    Parameters
    ----------


    Returns
    -------


    Examples
    --------
    """

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], **kwargs)

    if label:
        ax.set_title(r"$\bm{Y_2 = XW_2}$")
        ax.set_xlabel("$y_1$")
        ax.set_ylabel("$y_2$")

    return scatter, ax


def plot_pca_vecs(pca, ax=None, fixed=True, lims=None, label=True):
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    # Plot first two principle component vectors.
    vec1 = pca.components_[0]
    vec2 = pca.components_[1]
    if fixed:
        vec1 = vec1 if vec1[0] > 0 else -1.0 * vec1
        vec1 = vec1 / np.max(vec1)
        vec2 = vec2 if vec2[0] < 0 else -1.0 * vec2
        vec2 = vec2 / np.max(vec2)

    plt.scatter(np.arange(len(vec1)), vec1, s=1, label="$\\boldsymbol{w}_0$")

    # ax2 = ax.twinx()
    plt.scatter(
        np.arange(len(vec2)), vec2, s=1, color="orange", label="$\\boldsymbol{w}_1$"
    )

    if lims:
        ax.set_ylim(lims[0])
        # ax2.set_ylim(lims[1])

    if label:
        ordinal = lambda n: "%d%s" % (
            n,
            "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )
        ax.set_title("$\\boldsymbol{w}_i$")
        ax.set_xlabel("index")
        ax.set_ylabel("$(\\boldsymbol{w}_0)_i$")
        # ax2.set_ylabel("$(\\boldsymbol{w}_1)_i$")

        h1, l1 = ax.get_legend_handles_labels()
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1+h2, l1+l2, loc='lower left', fontsize=14)
        ax.legend(h1, l1, loc="lower left")

    return ax


def plot_pca_sample_contours(samples, X_train, ax=None, i=0, s=100, label=True):
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    # Plot lambda spaced contoured by our trained PCA data set.
    plt.scatter(samples[:, 0], samples[:, 1], c=X_train[:, i], s=s)

    if label:
        if i == 0:
            ax.set_title("$t_{i,1} = x_{i,j}w_{j,1}$")
        else:
            ax.set_title("$t_{i,2} = x_{i,j}w_{j,2}$")
        ax.set_xlabel("$\\Lambda_1$")
        ax.set_ylabel("$\\Lambda_2$")

    return ax
