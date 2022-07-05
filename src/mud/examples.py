import logging
from typing import Callable, Dict, List, Union

import numpy as np
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import distributions as ds  # type: ignore

from mud.base import (DensityProblem, IterativeLinearProblem,
                      LinearGaussianProblem)
from mud.funs import wme
from mud.pde import PDEProblem
from mud.util import std_from_equipment

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def polynomial_1D(
    p: int = 5,
    n_samples: int = int(1e3),
    domain: Union[np.ndarray, List[List[float]]] = [[-1, 1]],
    mu: float = 0.25,
    sigma: float = 0.1,
    N: int = 1,
):
    """
    Polynomial 1D QoI Map

    Generates test data for an inverse problem involving the polynomial QoI map

    .. math::
        Q_p(\\lambda) = \\lambda^p
        :name: eq:q_poly

    Where the uncertain parameter to be determined is :math:`\lambda`.
    ``n_samples`` samples from a uniform distribution over ``domain`` are
    generated using :func:`numpy.random.uniform` and pushed through the
    :ref:`forward model <eq:q_poly>`. ``N`` observed data points are
    generated from a normal distribution centered at ``mu`` with standard
    deviation ``sigma`` using :obj:`scipy.stats.norm`.

    Parameters
    ----------
    p: int, default=5
        Power of polynomial in :ref:`QoI map<eq:q_poly>`.
    num_samples: int, default=100
        Number of :math:`\lambda` samples to generate from a uniform
        distribution over ``domain`` for solving inverse problem.
    domain: :obj:`numpy.typing.ArrayLike`, default=[[-1, 1]]
        Domain to draw lambda samples from.
    mu: float, default=0.25
        True mean value of observed data.
    sigma: float, default=0.1
        Standard deviation of observed data.
    N: int, default=1
        Number of data points to generate from observed distribution. Note if 1,
        the default value, then the singular drawn value will always be ``mu``.

    Returns
    -------
    data: Tuple[:class:`numpy.ndarray`,]
        Tuple of ``(lam, q_lam, data)`` where ``lam`` is contains the
        :math:`\lambda` samples, ``q_lam`` the value of :math:`Q_p(\lambda)`,
        and ``data`` the observed data values from the
        :math:`\mathcal{N}(\mu, \sigma)` distribution.
    """

    # QoI Map - Polynomial x^p
    QoI = lambda x, y: x**y

    # Generate samples lam, QoI(lam), and simulated data
    domain = np.reshape(domain, (1, 2))
    lam = np.random.uniform(
        low=domain[0][0], high=domain[0][1], size=n_samples
    ).reshape(-1, 1)
    q_lam = QoI(lam, p).reshape(-1, 1)  # Evaluate lam^5 samples
    if N == 1:
        data = np.array([mu])
    else:
        data = norm.rvs(loc=mu, scale=sigma**2, size=N)

    return lam, q_lam, data


def rotation_map(qnum=10, tol=0.1, b=None, ref_param=None, seed=None):
    """
    Generate test data linear rotation map

    """
    if seed is not None:
        np.random.seed(seed)

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

    # Calcluate number of epochs per trial using budget and number of QoI
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

        # Append to erros matrix to calculate mean error accross trials
        errors.append(prob.get_errors(ref_param))

    # Compute mean errors at each iteration across all trials
    avg_errs = np.mean(np.array(errors), axis=0)

    # Plot mean errors
    ax.plot(avg_errs, color, lw=5, label=label)


def identity_uniform_1D_density_prob(
    num_samples=2000,
    num_obs=20,
    y_true=0.5,
    noise=0.05,
    weights=None,
    domain=[0, 1],
    wme_map=True,
    analytical_pred=True,
):
    """
    1D Density Problem using WME on identity map with uniform initial

    Sets up a Density Problem using a given domain (unit by default) and a
    uniform initial distribution under an identity map and the Weighted
    Mean Error map to . This function is used
    as a set-up for tejjsts to the DensityProblem class.

    `num_obs` observations
    are collected from an initial distribution and used as the true signal,
    with noise being added to each observation.
    Sets up an inverse problem using the unit domain and uniform distribution
    under an identity map. This is equivalent to studying a
    \"steady state\" signal over time, or taking repeated measurements
    of the same quantity to reduce variance in the uncertainty.
    """
    init_dist = ds.uniform(loc=domain[0], scale=domain[1] - domain[0])
    X = init_dist.rvs(size=(num_samples, 1))

    if wme_map:
        y_pred = np.repeat(X, num_obs, 1)
        # data is truth + noise
        y_observed = y_true * np.ones(num_obs) + noise * np.random.randn(num_obs)
        Y = wme(y_pred, y_observed, sd=noise)

        # Build Density problem of M(X) = WME(X,y_observed) over domain
        D = DensityProblem(X, Y, np.array([domain]), weights=weights)
        D.set_initial(init_dist)

        if analytical_pred:
            # analytical construction of predicted domain under identity map.
            y_domain = np.repeat(np.array([[0], [1]]), num_obs, 1)
            mn, mx = wme(y_domain, y_observed, sd=noise)
            loc, scale = mn, mx - mn
            D.set_predicted(ds.uniform(loc=loc, scale=scale))
    else:
        # Build Density problem of M(X) = X over domain
        D = DensityProblem(X, X, np.array([domain]), weights=weights)
        D.set_initial(init_dist)

        if analytical_pred:
            D.set_predicted(init_dist)

    return D


def exp_decay_1D(
    u_0: float = 0.75,
    time_range: Union[List[float], Tuple[float]] = [0, 4.0],
    domain: Union[List[float], Tuple[float]] = [0, 1],
    num_samples: int = 10000,
    lambda_true: float = 0.5,
    t_start: float = 0.0,
    sampling_freq: float = 100.0,
    std_dev: float = 0.05,
):

    u_t_lambda = lambda t, l: u_0 * np.exp(-np.outer(l, t))

    # Build initial samples
    initial = ds.uniform(loc=domain[0], scale=domain[1] - domain[0])

    exp_decay = PDEProblem()
    exp_decay.domain = domain
    exp_decay.times = np.arange(t_start, time_range[1], 1 / sampling_freq)
    exp_decay.sample_dist = "u"
    exp_decay.lam = initial.rvs(size=num_samples)
    exp_decay.data = u_t_lambda(exp_decay.times, exp_decay.lam)
    exp_decay.true_vals = u_t_lambda(exp_decay.times, lambda_true)[0]
    exp_decay.std_dev = std_dev

    return exp_decay


def exp_decay_2D(
    time_range=[0, 3.0],
    domain=np.array([[0.7, 0.8], [0.25, 0.75]]),
    num_samples=100,
    lambda_true=[0.75, 0.5],
    N=100,
    t_start=0.0,
    sampling_freq=10.0,
    std_dev=0.05,
):

    u_t_lambda = lambda t, l1, l2: (l1 * np.exp(-np.outer(t, l2))).T

    # Build initial samples
    num_params = domain.shape[0]
    mn = np.min(domain, axis=1)
    mx = np.max(domain, axis=1)
    initial = ds.uniform(loc=mn, scale=mx - mn)

    exp_decay = PDEProblem()
    exp_decay.domain = domain
    exp_decay.times = np.arange(t_start, time_range[1], 1 / sampling_freq)[0:N]
    exp_decay.sample_dist = "u"
    exp_decay.lam = initial.rvs(size=(num_samples, num_params))
    exp_decay.data = u_t_lambda(
        exp_decay.times, exp_decay.lam[:, 0], exp_decay.lam[:, 1]
    )
    exp_decay.true_vals = u_t_lambda(exp_decay.times, lambda_true)[0]
    exp_decay.std_dev = std_dev

    return exp_decay


def pde_scan_n_data_points(
    prob_constructor: Callable,
    prob_kwargs: Dict = {},
    method: str = "wme",
    pca_components: int = 1,
    N_trials: int = 10,
    N_vals: List[int] = list(range(5, 100, 5)),
):
    """ """
    pde_prob = prob_constructor(**prob_kwargs)
    if type(pde_prob) != PDEProblem:
        raise ValueError("Constructor did not return a PDEProblem object")
    pde_prob.validate()

    scan_res = np.zeros((len(N_vals), 4))
    for i, N in enumerate(N_vals):
        _logger.info(f"Solving pde problem using {N} data points.")
        times_idxs = np.arange(0, N)
        mud_ests = []
        r_vals = []
        for j in range(N_trials):
            # Take new set of noise measurments per trial
            pde_prob.measurements_from_reference()
            mud_prob = pde_prob.mud_problem(
                method=method, pca_components=pca_components
            )
            mud_ests.append(mud_prob.estimate())
            r_vals.append(mud_prob.exp_r())

        scan_res[i][0] = np.mean(mud_ests)
        scan_res[i][1] = np.var(mud_ests)
        scan_res[i][2] = np.mean(r_vals)
        scan_res[i][3] = np.var(r_vals)

    return scan_res


def noisy_linear_data(
    M: np.ndarray,
    reference_point: np.ndarray,
    std: Union[float, int, np.ndarray],
    num_obs: int = None,
):
    """
    Creates data produced by model assumed to be of the form:
    # FIXME: LaTeX
    Q(\lambda) = M\lambda + odj,i = Mj(λ^\\dagger) + ξi, ξi ∼ N(0, σ_j^2), 1 ≤ i ≤ Nj

    Parameters
    ----------

    Returns
    -------

    """
    dim_input = len(reference_point)

    if num_obs is None:
        num_obs = M.shape[0]
    assert (
        M.shape[1] == dim_input
    ), f"Operator/Reference dimension mismatch. op: {M.shape}, input dim: {dim_input}"
    assert (
        M.shape[0] == 1 or M.shape[0] == num_obs
    ), f"Operator/Data dimension mismatch. op: {M.shape}, observations: {num_obs}"
    if isinstance(std, (int, float)):  # support for std per measurement
        std = np.array([std] * num_obs)  # noqa: E221
    else:
        assert (
            len(std.ravel()) == num_obs
        ), f"St. Dev / Data mismatch. data: {num_obs}, std: {len(std.ravel())}"

    ref_input = np.array(list(reference_point)).reshape(-1, 1)  # noqa: E221
    ref_data = M @ ref_input  # noqa: E221
    noise = np.diag(std) @ np.random.randn(num_obs, 1)  # noqa: E221
    if ref_data.shape[0] == 1:
        ref_data = float(ref_data)
    data = ref_data + noise  # noqa: E221
    return data.ravel()


def random_linear_wme_problem(
    reference_point: float,
    std_dev: float,
    num_qoi: int = 1,
    num_observations: int = 10,
    dist: str = "normal",
    repeated: bool = False,
):
    """
    Create a random linear WME problem

    Parameters
    ----------
    reference_point : ndarray
        Reference true parameter value.
    dist: str, default='normal'
        Distribution to draw random linear map from. 'normal' or 'uniform' supported at the moment.
    num_qoi : int, default = 1
        Number of QoI
    num_observations: int, default = 10
        Number of observation data points.
    std_dev: ndarray, optional
        Standard deviation of normal distribution from where observed data points are drawn from.
        If none specified, noise-less data is created.

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
    mean_i: np.ndarray = None,
    cov_i: np.ndarray = None,
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

    # Initial Covariance drawn from standard normal centerred at 0.5
    if cov_i is None:
        cov_i = np.diag(np.sort(np.random.rand(dim_input))[::-1] + 0.5)

    lin_prob = LinearGaussianProblem(A, b, y, mean_i, cov_i)

    return lam_ref, lin_prob
