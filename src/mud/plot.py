"""
MUD Plotting Module

Plotting utility functions for visualizing data-sets and distributions related
to algorithm implemented within the MUD library.

Functions
---------

"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats.contingency import margins  # type: ignore

from mud.util import null_space

# Matplotlib plotting options
mud_plot_params = {
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.labelsize": 14,
    "axes.titlesize": 26,
    "legend.fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlepad": 1,
    "axes.labelpad": 1,
    "font.size": 16,
    "savefig.facecolor": "white",
    "text.usetex": True,
    "text.latex.preamble": " ".join(
        [r"\usepackage{bm}", r"\usepackage{amsfonts}", r"\usepackage{amsmath}"]
    ),
}
plt.rcParams.update(mud_plot_params)


def _check_latex():
    """check latex installation"""
    global mud_plot_params

    path = Path.cwd() / ".test_fig.png"
    plt.plot([0], [1], label=r"$a_\text{foo} = \lambda$")
    try:
        plt.legend()
        plt.savefig(str(path))
        path.unlink(missing_ok=True)
    except RuntimeError:
        print("NOT USING TEX")
        mud_plot_params["text.usetex"] = False
        mud_plot_params["text.latex.preamble"] = ""
        plt.rcParams.update(mud_plot_params)


def save_figure(fname: str, save_path: str = None, close_fig: bool = True, **kwargs):
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
    close_fig: bool, default=True
        Whether to close the figure after saving it.
    kwargs: dict, optional
        Arguments to pass to savefig()


    """
    global mud_plot_params

    if save_path is not None:
        fname = str(Path(save_path) / Path(fname))
        plt.savefig(fname, **kwargs)
    if close_fig:
        plt.close()


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


def plot_1D_vecs(vecs, markers=None, ax=None, label=True, **kwargs):
    """Plot components of 1D vectors"""
    # Plot density of trained data
    if ax is None:
        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 1, 1)

    for i, v in enumerate(vecs):
        if markers is not None:
            kwargs["marker"] = markers[i]
        plt.scatter(np.arange(len(v)), v, label=f"$p^{{({i+1})}}$", **kwargs)

    if label:
        ax.set_xlabel("i")
        ax.set_ylabel("$p^{{(l)}}_i$")
        ax.legend()

    return ax


# TODO: replace make_2d_normal_Grid with this
def build_nd_mesh_grid(domain, aff=100):
    """
    Build n-dimensional mesh grid

    Parameters
    ----------
    domain: List[List[int]]
        List of `[min, max]` ranges for each parameter to construct grid over.
    aff : int, default=100
        Number of points to use in each dimension.

    Returns
    -------
    (mesh, grid_points): Tuple[numpy.ndarray, numpy.ndarray]
        Tuple consistent of n-dimensional grid of points `mesh` and the list of
        coordinates $(x_1, x_2, ..., x_n)$ for each grid point along.

    Examples
    --------
    Building 1d ``mesh"

    >>> mesh_1d, pts_1d = build_nd_mesh_grid([[0,1]], aff=5)
    >>> mesh_1d
    [array([0.  , 0.25, 0.5 , 0.75])]
    >>> pts_1d
    array([[0.  , 0.25, 0.5 , 0.75]])

    """

    mesh = np.meshgrid(*[np.linspace(i, j, aff)[:-1] for i, j in domain])
    grid_points = np.vstack([x.ravel() for x in mesh])

    return (mesh, grid_points)


def plot_dist(dist, domain, ax=None, idx=0, source="kde", aff=100, **kwargs):
    """
    Plot a probability distribution over a given domain.

    """
    # Create plot if one isn't passed in
    _, ax = plt.subplots(1, 1) if ax is None else (None, ax)

    # Mesh is a list of grids over n-dim space the distribution is defined over
    mesh, grid = build_nd_mesh_grid(domain, aff=aff)
    plot_x = np.linspace(domain[idx, 0], domain[idx, 1], aff)[: aff - 1]

    # TODO: Add check that domain size matches dimension of dist
    if source == "pdf":
        # Compute observed distribution using stored pdf
        plot_y = margins(np.reshape(dist.pdf(grid.T).prod(axis=1), mesh[idx].shape))[
            idx
        ].reshape(-1)
    elif source == "kde":
        # Compute observed distribution using kernel density estimate
        plot_y = margins(np.reshape(dist(grid), mesh[idx].shape).T)[idx].reshape(-1)
    else:
        raise ValueError("Source must be one of pdf or kde.")

    # Scale distribution
    total = np.sum(plot_y * np.abs(plot_x[1] - plot_x[0]))
    plot_y = plot_y / total

    # Plot updated distribution over parameter space
    ax.plot(plot_x, plot_y, **kwargs)

    return ax


def plot_vert_line(ax, x_loc, ylim=None, **kwargs):
    """Plot a vertical line on an existing axis at `x_loc`"""
    ylims = list(ax.get_ylim())
    ylims[1] = ylim if ylim is not None else ylims[1]
    ax.plot([x_loc, x_loc], [ylims[0], ylims[1]], **kwargs)
    ax.set_ylim(ylims)


_check_latex()
