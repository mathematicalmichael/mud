"""
Simple Example
"""
import numpy as np
from scipy.stats import distributions as ds  # type: ignore
from scipy.stats import norm  # type: ignore

from mud.base import BayesProblem, DensityProblem, SpatioTemporalProblem


def polynomial_1D_data(
    p: int = 5,
    num_samples: int = int(1e3),
    domain: np.typing.ArrayLike = [[-1, 1]],
    init_dist=ds.uniform(loc=0, scale=1),
    mu: float = 0.25,
    sigma: float = 0.1,
    N: int = 1,
):
    r"""
    Polynomial 1D QoI Map

    Generates test data for an inverse problem involving the polynomial QoI map

    .. math::
        Q_p(\\lambda) = \\lambda^p
        :name: eq:q_poly

    Where the uncertain parameter to be determined is :math:`\lambda`.
    ``num_samples`` samples from a uniform distribution over ``domain`` are
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

    Examples
    --------
    Note when N=1, data point drawn is always equal to mean.

    >>> import numpy as np
    >>> from mud.examples.comparison import polynomial_1D_data
    >>> lam, q_lam, data = polynomial_1D_data(num_samples=10, N=1)
    >>> data[0]
    0.25
    >>> len(lam)
    10

    For higher values of N, values are drawn from N(mu, sigma) distribution.

    >>> lam, q_lam, data = polynomial_1D_data(N=10, mu=0.5, sigma=0.01)
    >>> len(data)
    10
    >>> np.mean(data) < 0.6
    True
    """

    # QoI Map - Polynomial x^p
    def QoI(x, y):
        return x**y

    # Generate samples lam, QoI(lam), and simulated data
    domain = np.reshape(domain, (1, 2))
    lam = init_dist.rvs(size=(num_samples, 1))
    q_lam = QoI(lam, p).reshape(-1, 1)  # Evaluate lam^5 samples
    if N == 1:
        data = np.array([mu])
    else:
        data = norm.rvs(loc=mu, scale=sigma**2, size=N)

    return lam, q_lam, data


def identity_1D_density_prob(
    num_samples=2000,
    num_obs=20,
    mu=0.5,
    sigma=0.05,
    weights=None,
    init_dist="uniform",
    normalize=False,
    domain=[0, 1],
    analytical_pred=True,
):
    """
    Identity 1D Density Problem

    Solving 1d identity map parameter estimation problem using the
    DensityProblem class and the mud point estimate.
    """
    if init_dist == "uniform":
        init_dist = ds.uniform(loc=domain[0], scale=domain[1] - domain[0])
    lam, q_lam, data = polynomial_1D_data(
        p=1, num_samples=num_samples, N=num_obs, init_dist=init_dist, mu=mu, sigma=sigma
    )
    D = DensityProblem(lam, q_lam, domain, weights=weights, normalize=normalize)
    D.set_initial(init_dist)
    D.set_observed(ds.norm(np.mean(data), sigma))
    if analytical_pred:
        D.set_predicted(init_dist)

    return D


def identity_1D_bayes_prob(
    num_samples=1000,
    num_obs=50,
    mu=0.5,
    sigma=0.05,
    init_dist=ds.uniform(loc=0, scale=1),
    domain=None,
):
    """
    Identity 1D Bayes Problem

    Solving 1d identity map parameter estimation problem using the
    BayesProlem class and the map point estimate.
    """
    lam, q_lam, data = polynomial_1D_data(
        p=1, num_samples=num_samples, N=num_obs, init_dist=init_dist, mu=mu, sigma=sigma
    )
    B = BayesProblem(lam, q_lam, domain=domain)
    B.set_likelihood(ds.norm(loc=data, scale=sigma))
    return B


def identity_1D_temporal_prob(
    num_samples=2000,
    num_obs=20,
    y_true=0.5,
    noise=0.05,
    weights=None,
    init_dist=ds.uniform(loc=0, scale=1),
    normalize=False,
    domain=None,
    wme_map=True,
    analytical_pred=True,
):
    """
    Identity 1D Temporal Problem

    Solving 1d identity map parameter estimation problem using the
    SpatioTemporalProblem class to construct DensityProblem class using the
    WME map to aggregate data.
    """
    lam, q_lam, data = polynomial_1D_data(
        p=1,
        num_samples=num_samples,
        N=num_obs,
        init_dist=init_dist,
        mu=y_true,
        sigma=noise,
    )
    data = {
        "lam": lam,
        "data": data,
        "true_vals": y_true,
        "measurements": None,
        "std_dev": noise,
        "sample_dist": "uniform",
        "domain": domain,
        "lam_ref": y_true,
        "times": np.arange(num_obs),
    }
    temporal_prob = SpatioTemporalProblem()
    temporal_prob.load(data)
    D = temporal_prob.mud_problem(method="wme")
    D.set_initial(init_dist)
    D.set_observed(ds.norm(0, 1))  # always N(0,1) for WME map
    return D
