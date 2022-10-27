"""
MUD Linear Examples

Functions for examples for linear problems.
"""
import logging
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy as sp  # type: ignore
from matplotlib import cm  # type: ignore

from mud.base import IterativeLinearProblem, LinearGaussianProblem, LinearWMEProblem
from mud.plot import mud_plot_params, save_figure
from mud.util import rank_decomposition, std_from_equipment

# Matplotlib plotting options
plt.rcParams.update(mud_plot_params)

_logger = logging.getLogger(__name__)


def random_linear_wme_problem(
    reference_point,
    std_dev,
    num_qoi=1,
    num_observations=10,
    dist="normal",
    repeated=False,
):
    """
    Create a random linear WME problem

    Parameters
    ----------
    reference_point : np.ndarray
        Reference true parameter value.
    dist: str, default='normal'
        Distribution to draw random linear map from. 'normal' or 'uniform' supported
        at the moment.
    num_qoi : int, default = 1
        Number of QoI
    num_observations: int, default = 10
        Number of observation data points.
    std_dev: np.ndarray, optional
        Standard deviation of normal distribution from where observed data points are
        drawn from. If none specified, noise-less data is created.

    Returns
    -------


    """
    if isinstance(std_dev, (int, float)):
        std_dev = np.array([std_dev]) * num_qoi
    else:
        assert len(std_dev) == num_qoi

    if isinstance(num_observations, (int, float)):
        num_observations = [num_observations] * num_qoi
    else:
        assert len(num_observations) == num_qoi

    assert len(std_dev) == len(num_observations)

    dim_input = len(reference_point)
    operator_list = []
    data_list = []
    for n, s in zip(num_observations, std_dev):

        if dist == "normal":
            M = np.random.randn(num_qoi, dim_input)
        else:
            M = np.random.rand(num_qoi, dim_input)

        if repeated:  # just use first row
            M = M[0, :].reshape(1, dim_input)

        if isinstance(s, (int, float)):  # support for s per measurement
            s = np.array([s] * n)  # noqa: E221
        else:
            assert (
                len(s.ravel()) == n
            ), f"St. Dev / Data mismatch. data: {n}, s: {len(s.ravel())}"

        ref_input = np.array(list(reference_point)).reshape(-1, 1)
        ref_data = M @ ref_input  # noqa: E221
        noise = np.diag(s) @ np.random.randn(n, 1)
        if ref_data.shape[0] == 1:
            ref_data = float(ref_data)
        data = ref_data + noise

        operator_list.append(M)
        data_list.append(data.ravel())

    return operator_list, data_list, std_dev


def random_linear_problem(
    dim_input: int = 10,
    dim_output: int = 10,
    mean_i: np.typing.ArrayLike = None,
    cov_i: np.typing.ArrayLike = None,
    seed: int = None,
):
    """Construct a random linear Gaussian Problem"""

    if seed is not None:
        np.random.seed(seed)

    # Construct random inputs drawn from standard normal
    A = np.random.randn(dim_output, dim_input)
    b = np.random.randn(dim_output).reshape(-1, 1)
    lam_ref = np.random.randn(dim_input).reshape(-1, 1)
    y = A @ lam_ref + b

    # Initial guess at mean is just origin
    if mean_i is None:
        mean_i = np.zeros(dim_input).reshape(-1, 1)

    # Initial Covariance drawn from standard normal centered at 0.5
    if cov_i is None:
        cov_i = np.diag(np.sort(np.random.rand(dim_input))[::-1] + 0.5)

    lin_prob = LinearGaussianProblem(A, b, y, mean_i, cov_i)

    return lam_ref, lin_prob


def noisy_linear_data(M, reference_point, std, num_data=None):
    """
    Creates data produced by model assumed to be of the form:

    Q(lam) = M * lam + odj,i =Mj(λ†)+ξi, ξi ∼N(0,σj2), 1≤i≤Nj

    Parameters
    ----------

    Returns
    -------

    """
    dim_input = len(reference_point)

    if num_data is None:
        num_data = M.shape[0]
    assert (
        M.shape[1] == dim_input
    ), f"Operator/Reference dimension mismatch. op: {M.shape}, input dim: {dim_input}"
    assert (
        M.shape[0] == 1 or M.shape[0] == num_data
    ), f"Operator/Data dimension mismatch. op: {M.shape}, observations: {num_data}"
    if isinstance(std, (int, float)):  # support for std per measurement
        std = np.array([std] * num_data)  # noqa: E221
    else:
        assert (
            len(std.ravel()) == num_data
        ), f"St. Dev / Data mismatch. data: {num_data}, std: {len(std.ravel())}"

    ref_input = np.array(list(reference_point)).reshape(-1, 1)  # noqa: E221
    ref_data = M @ ref_input  # noqa: E221
    noise = np.diag(std) @ np.random.randn(num_data, 1)  # noqa: E221
    if ref_data.shape[0] == 1:
        ref_data = float(ref_data)
    data = ref_data + noise  # noqa: E221
    return data.ravel()


def rotation_map(qnum=10, tol=0.1, b=None, ref_param=None, seed=None):
    """
    Generate test data linear rotation map

    """
    if seed is not None:
        np.random.seed(24)

    vec = np.linspace(0, np.pi, qnum)
    A = np.array([[np.sin(theta), np.cos(theta)] for theta in vec])
    A = A.reshape(qnum, 2)
    b = np.zeros((qnum, 1)) if b is None else b
    ref_param = (
        np.array([[0.5, 0.5]]).reshape(-1, 1) if ref_param is None else ref_param
    )

    # Compute observed value
    y = A @ ref_param + b
    initial_mean = np.random.randn(2).reshape(-1, 1)
    initial_cov = np.eye(2) * std_from_equipment(tol)

    return (A, b, y, initial_mean, initial_cov, ref_param)


def rotation_map_trials(
    numQoI=10,
    method="ordered",
    num_trials=100,
    model_eval_budget=100,
    ax=None,
    color="r",
    label="Ordered QoI $(10\\times 10D)$",
    seed=None,
):
    """
    Run a set of trials for linear rotation map problems

    """

    # Initialize plot if axis object is not passed in
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Build Rotation Map. This will initialize seed of trial if specified
    A, b, y, initial_mean, initial_cov, ref_param = rotation_map(qnum=numQoI, seed=seed)

    # Calculate number of epochs per trial using budget and number of QoI
    num_epochs = model_eval_budget // numQoI

    errors = []
    for trial in range(num_trials):
        # Get a new random initial mean to start from per trial on same problem
        initial_mean = np.random.rand(2, 1)

        # Initialize number of epochs and idx choices to use on this trial
        epochs = num_epochs
        choice = np.arange(numQoI)

        # Modify epochs/choices based off of method
        if method == "ordered":
            # Ordered - GO through each row in order once per epoch, each trial
            epochs = num_epochs
        elif method == "shuffle":
            # Shuffled - Shuffle rows on each trial, in order once per epoch
            np.random.shuffle(choice)
            epochs = num_epochs
        elif method == "batch":
            # Batch - Perform only one epoch, but iterate in random
            #         order num_epochs times over each row of A
            choice = list(np.arange(numQoI)) * num_epochs
            np.random.shuffle(choice)
            epochs = 1
        elif method == "random":
            # Randoms - Perform only one epoch, but do num_epochs*rows
            #           random choices of rows of A, with replacement
            choice = np.random.choice(np.arange(numQoI), size=num_epochs * numQoI)
            epochs = 1

        # Initialize Iterative Linear Problem and solve using number of epochs
        prob = IterativeLinearProblem(
            A, b=b, y=y, initial_mean=initial_mean, cov=initial_cov, idx_order=choice
        )
        _ = prob.solve(num_epochs=epochs)

        # Plot errors with respect to reference parameter over each iteration
        prob.plot_chain_error(ref_param, alpha=0.1, ax=ax, color=color, fontsize=36)

        # Append to errors matrix to calculate mean error across trials
        errors.append(prob.get_errors(ref_param))

    # Compute mean errors at each iteration across all trials
    avg_errs = np.mean(np.array(errors), axis=0)

    # Plot mean errors
    ax.plot(avg_errs, color, lw=5, label=label)


def run_contours(
    plot_fig: List[str] = None,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = False,
    **kwargs,
):
    """
    Run Contours

    Produces contour plots of 2-D parameter space for 2-to-1 linear map
    found in Figures 3 and 5 of [ref]. These contour plots show the different
    regularization terms between Bayesian and Data-Consistent solutions that
    lead to a different optimization problem and therefore a different solution
    to the inverse problem.

    Parameters
    ----------
    plot_fig : str, default='all'
        Which figures to produce. Possible options are data_mismatch,
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool, default=False
        Set to True to close figure and only save it.
    kwargs: dict, optional
        kwargs to overwrite default arguments used to build linear problem.
        Possible values include and their expected types, and default values:
            - A : 2D array, default=[[1, 1]]
            - b - 1D array, default = [0]
            - y - 1D array, default =[1]
            - mean_i - 1D array, default = [0.25, 0.25]
            - cov_i - 2D array, default = [[1, -0.25], [-0.25, 0.5]]
            - cov_o - 1D array, default = [1]

    Returns
    ----------
    lin_prob : mud.base.LinearGaussianProblem
        LinearGaussianProblem object with solved linear inverse problem and
        associated data within.
    """
    plot_fig = ["all"] if plot_fig is None else plot_fig

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

    _ = (lin_prob.solve("mud"), lin_prob.solve("map"), lin_prob.solve("ls"))

    if "data_mismatch" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        lin_prob.plot_fun_contours(
            ax=ax, terms="data", levels=50, cmap=cm.viridis, alpha=1.0
        )
        lin_prob.plot_contours(
            ax=ax,
            annotate=True,
            note_loc=[0.1, 0.9],
            label="Solution Contour",
            plot_opts={"color": "r"},
            annotate_opts={"fontsize": 20, "backgroundcolor": "w"},
        )
        ax.axis("equal")
        _ = ax.set_xlim([0, 1])
        _ = ax.set_ylim([0, 1])
        save_figure(
            "data_mismatch_contour.png",
            save_path=save_path,
            dpi=dpi,
            close_fig=close_fig,
        )
    if "tikhonov" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        lin_prob.plot_fun_contours(
            ax=ax, terms="reg", levels=50, cmap=cm.viridis, alpha=1.0
        )
        lin_prob.plot_sol(
            ax=ax,
            point="initial",
            label="Initial Mean",
            pt_opts={
                "color": "k",
                "s": 100,
                "marker": "o",
                "label": "MUD",
                "zorder": 10,
            },
        )
        _ = ax.axis([0, 1, 0, 1])
        save_figure(
            "tikhonov_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )
    if "consistent" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        lin_prob.plot_fun_contours(
            ax=ax, terms="reg_m", levels=50, cmap=cm.viridis, alpha=1.0
        )
        lin_prob.plot_sol(
            ax=ax,
            point="initial",
            label="Initial Mean",
            pt_opts={
                "color": "k",
                "s": 100,
                "marker": "o",
                "label": "MUD",
                "zorder": 10,
            },
        )
        _ = ax.axis([0, 1, 0, 1])
        save_figure(
            "consistent_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )
    if "map" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        lin_prob.plot_fun_contours(ax=ax, terms="bayes", levels=50, cmap=cm.viridis)
        lin_prob.plot_fun_contours(
            ax=ax,
            terms="data",
            levels=25,
            cmap=cm.viridis,
            alpha=0.5,
            vmin=0,
            vmax=4,
        )
        lin_prob.plot_sol(
            ax=ax,
            point="initial",
            pt_opts={
                "color": "k",
                "s": 100,
                "marker": "o",
                "label": "MUD",
                "zorder": 20,
            },
        )
        lin_prob.plot_sol(
            ax=ax,
            point="ls",
            label="Least Squares",
            note_loc=[0.49, 0.55],
            pt_opts={"color": "xkcd:blue", "s": 100, "marker": "d", "zorder": 10},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
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
        lin_prob.plot_contours(
            ax=ax,
            annotate=False,
            note_loc=[0.1, 0.9],
            label="Solution Contour",
            plot_opts={"color": "r"},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
        _ = ax.axis([0, 1, 0, 1])
        save_figure(
            "classical_solution.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )
    if "mud" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        lin_prob.plot_fun_contours(ax=ax, terms="dc", levels=50, cmap=cm.viridis)
        lin_prob.plot_fun_contours(
            ax=ax,
            terms="data",
            levels=25,
            cmap=cm.viridis,
            alpha=0.5,
            vmin=0,
            vmax=4,
        )
        lin_prob.plot_sol(
            ax=ax,
            point="initial",
            pt_opts={
                "color": "k",
                "s": 100,
                "marker": "o",
                "label": "MUD",
                "zorder": 20,
            },
        )
        lin_prob.plot_sol(
            ax=ax,
            point="ls",
            label="Least Squares",
            note_loc=[0.49, 0.55],
            pt_opts={"color": "k", "s": 100, "marker": "d", "zorder": 10},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
        lin_prob.plot_sol(
            point="mud",
            ax=ax,
            label="MUD",
            pt_opts={"color": "k", "s": 100, "marker": "*", "zorder": 10},
            ln_opts={"color": "k", "marker": "*", "lw": 1, "zorder": 10},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
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
        save_figure(
            "consistent_solution.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )
    if "comparison" in plot_fig or "all" in plot_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        lin_prob.plot_fun_contours(ax=ax, terms="bayes", levels=50, cmap=cm.viridis)
        lin_prob.plot_fun_contours(
            ax=ax,
            terms="data",
            levels=25,
            cmap=cm.viridis,
            alpha=0.5,
            vmin=0,
            vmax=4,
        )
        lin_prob.plot_sol(
            ax=ax,
            point="initial",
            pt_opts={
                "color": "k",
                "s": 100,
                "marker": "o",
                "label": "MUD",
                "zorder": 10,
            },
        )
        lin_prob.plot_sol(
            ax=ax,
            point="ls",
            label="Least Squares",
            note_loc=[0.49, 0.55],
            pt_opts={"color": "k", "s": 100, "marker": "d", "zorder": 10},
        )
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
        lin_prob.plot_sol(
            point="mud",
            ax=ax,
            label="MUD",
            pt_opts={
                "color": "k",
                "s": 100,
                "linewidth": 3,
                "marker": "*",
                "zorder": 10,
            },
            ln_opts={"color": "k", "marker": "*", "lw": 1, "zorder": 10},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
        lin_prob.plot_contours(
            ax=ax,
            annotate=False,
            note_loc=[0.1, 0.9],
            label="Solution Contour",
            plot_opts={"color": "r"},
            annotate_opts={"fontsize": 14, "backgroundcolor": "w"},
        )
        pt = [0.7, 0.3]
        ax.scatter([pt[0]], [pt[1]], color="k", s=100, marker="s", zorder=11)
        nc = (pt[0] - 0.02, pt[1] + 0.02)
        ax.annotate("Truth", nc, fontsize=14, backgroundcolor="w")
        _ = ax.axis([0, 1, 0, 1])
        save_figure(
            "map_compare_contour.png", save_path=save_path, dpi=dpi, close_fig=close_fig
        )

    return lin_prob


def run_wme_covariance(
    dim_input: int = 20,
    dim_output: int = 5,
    sigma: float = 1e-1,
    Ns: List[int] = [10, 100, 1000, 10000],
    seed: int = None,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = False,
):
    """
    Weighted Mean Error Map Updated Covariance

    Reproduces figure 4 from [ref], showing the spectral properties of the
    updated covriance for a the Weighted Mean Error map on a randomly
    generated linear operator as more data from repeated measurements is used
    to constructthe QoI map.

    Parameters
    ----------
    dim_input: int, default=20
        Input dimension of linear map (number of rows in A).
    dim_output: int, default=5
        Output dimension of linear map (number of columns in A).
    sigma: float, default=1e-1
        N(0, sigma) error added to produce "measurements" from linear operator.
    Ns: List[str]. default = [10, 100, 1000, 10000]
        List of number of data points to collect in constructing Q_WME map to
        view how the spectral properties of the updated covariance change as
        more data is included in the Q_WME map.
    seed: int, default = 21
        To fix results for reproducibility. Set to None to randomize results.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool, default=False
        Set to True to close figure and only save it.

    Returns
    -------
    linear_wme_prob, ax: Tuple[mud.base.LinearWMEProblem, matplotlib.pyplot.Axes]
        Tuple containing solved linear WME problems for each Ns value, and
        axes containing the plot of the first 20 eigenvalues of the updated
        covariances for each Q_WME map.
    """

    if seed is not None:
        np.random.seed(seed)

    initial_cov = np.diag(np.sort(np.random.rand(dim_input))[::-1] + 0.5)
    lam_ref = np.random.randn(dim_input).reshape(-1, 1)

    # Create operator list of random linear problems
    operator_list, data_list, _ = random_linear_wme_problem(
        lam_ref,
        [0] * dim_output,  # noiseless data bc we want to simulate multiple trials
        num_qoi=dim_output,
        num_observations=[max(Ns)] * dim_output,  # iterate over increasing measurements
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

        # Build Linear WME problem solve, and plot
        linear_wme_prob = LinearWMEProblem(
            _oper_list, _data_list, sigma, cov_i=initial_cov
        )
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
    save_figure(
        "lin-meas-cov-sd-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )

    return linear_wme_prob, ax


def run_high_dim_linear(
    dim_input: int = 100,
    dim_output: int = 100,
    seed: int = 21,
    save_path: str = None,
    dpi: int = 500,
    close_fig: bool = True,
):
    """
    Run High Dimension Linear Example

    Reproduces Figure 6 from [ref], showing the relative error between the true
    parameter value and the MUD, MAP and least squares solutions to linear
    gaussian inversion problems for increasing dimension and rank of a randomly
    generated linear map A.

    Parameters
    ----------
    dim_input: int, default=20
        Input dimension of linear map (number of rows in A).
    dim_output: int, default=5
        Output dimension of linear map (number of columns in A).
    seed: int, default = 21
        To fix results for reproducibility. Set to None to randomize results.
    save_path: str, optional
        If provided, path to save the resulting figure to.
    dpi: int, default=500
        Resolution of images saved
    close_fig: bool, default=False
        Set to True to close figure and only save it.

    Returns
    ----------
    rank_errs, dim_errs : Tuple[np.array, np.array]
        Tuple containing the error between the true solution and each of the
        (mud, map, least_squares) solutions for increasing dimension and rank
        from 1 to dim_output. These arrays are used to produce the plots given.
    """

    lam_ref, randn_high_dim = random_linear_problem(
        dim_input=dim_input, dim_output=dim_output, seed=seed
    )
    A_ranks = rank_decomposition(randn_high_dim.A)

    c = np.linalg.norm(lam_ref)

    def err(xs):
        return [np.linalg.norm(x - lam_ref) / c for x in xs]

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
            # Returns tuple of (mud, map, ls) solutions.
            rank_solutions = rank_prob.solve(method="all")

            # Compute errors
            d_errs.append(np.array(err(dim_solutions)))
            r_errs.append(np.array(err(rank_solutions)))

        dim_errs.append(np.array(d_errs))
        rank_errs.append(np.array(r_errs))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    x = np.arange(1, dim_output + 1, 1)
    for idx, alpha in enumerate(alpha_list):
        # Plot convergence for MUD and LS Solutions once
        if idx == 0:
            ax.plot(x, dim_errs[idx][:, 0], label="MUD", c="k", lw=10)

            # Plot convergence for Least Squares Solutions
            ax.plot(
                x,
                dim_errs[idx][:, 2],
                label="LSQ",
                c="xkcd:light blue",
                ls="-",
                lw=5,
                zorder=10,
            )

        # Plot convergence plot for MAP Solutions - Annotate for different alphas
        ax.plot(x, dim_errs[idx][:, 1], label="MAP", c="r", ls="--", lw=5, zorder=10)
        ax.annotate(
            f"$\\alpha$={alpha:1.2E}",
            (100, max(dim_errs[idx][:, 1][-1], 0.01)),
        )

    # Label plot
    _ = ax.set_title(
        "Convergence for Various $\\Sigma_{init} = \\alpha \\Sigma$",
    )
    _ = ax.set_ylim(0, 1.0)
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_xlabel("Dimension of Output Space")
    _ = ax.legend(["MUD", "MAP", "Least Squares"])

    save_figure(
        "lin-dim-cov-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for idx, alpha in enumerate(alpha_list):
        if idx == 1:
            # Plot convergence for MUD Solutions
            ax.plot(x, rank_errs[idx][:, 0], label="MUD", c="k", lw=10)

            # Plot convergence for Least Squares Solutions
            ax.plot(
                x,
                rank_errs[idx][:, 2],
                label="LSQ",
                c="xkcd:light blue",
                ls="-",
                lw=5,
                zorder=10,
            )

        # Plot convergence plot for MAP Solutions - Annotate for different alphas
        ax.plot(x, rank_errs[idx][:, 1], label="MAP", c="r", ls="--", lw=5, zorder=10)
        ax.annotate(
            f"$\\alpha$={alpha:1.2E}",
            (100, max(rank_errs[idx][:, 1][-1], 0.01)),
        )

    # Label plot
    _ = ax.set_title(
        "Convergence for Various $\\Sigma_{init} = \\alpha \\Sigma$",
    )
    _ = ax.set_ylim(0, 1.0)
    _ = ax.set_ylabel("Relative Error")
    _ = ax.set_xlabel("Rank(A)")
    _ = ax.legend(["MUD", "MAP", "Least Squares"])

    save_figure(
        "lin-rank-cov-convergence.png",
        save_path=save_path,
        dpi=dpi,
        close_fig=close_fig,
    )

    return dim_errs, rank_errs
