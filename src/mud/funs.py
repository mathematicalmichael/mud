# -*- coding: utf-8 -*-
"""
Python console script for `mud`, installed with
`pip install .` or `python setup.py install`
"""

import argparse
import sys
import logging
import numpy as np

from mud import __version__
from mud.base import DensityProblem, BayesProblem
from scipy.stats import distributions as dists

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Demonstration of analytical MUD point"
    )
    parser.add_argument(
        "--version", action="version", version="mud {ver}".format(ver=__version__)
    )
    parser.add_argument(dest="n", help="Number of QoI", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("Using {} Quantities of Interest".format(args.n))
    _logger.info("Script end.")


def run():
    """Entry point for console_scripts"""
    main(sys.argv[1:])


############################################################


def wme(X, data, sd=None):
    if sd is None:
        sd = np.std(data)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    num_evals = X.shape[0]
    assert X.shape[1] == len(data)

    residuals = np.subtract(X, data)
    weighted_residuals = np.divide(residuals, sd)
    assert weighted_residuals.shape[0] == num_evals

    weighted_sum = np.sum(weighted_residuals, axis=1)
    return weighted_sum / np.sqrt(len(data))


def makeRi(A, initial_cov):
    predicted_cov = A @ initial_cov @ A.T
    if isinstance(predicted_cov, float):
        ipc = 1.0 / predicted_cov * np.eye(1)
    else:
        ipc = np.linalg.inv(predicted_cov)
    Ri = np.linalg.inv(initial_cov) - A.T @ ipc @ A
    return Ri


def check_args(A, b, y, mean, cov, data_cov):
    n_samples, dim_input = A.shape

    if data_cov is None:
        data_cov = np.eye(n_samples)
    if cov is None:
        cov = np.eye(dim_input)
    if mean is None:
        mean = np.zeros((dim_input, 1))
    if b is None:
        b = np.zeros((n_samples, 1))
    if y is None:
        y = np.zeros(n_samples)

    ravel = False
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        ravel = True

    if b.ndim == 1:
        b = b.reshape(-1, 1)

    if mean.ndim == 1:
        mean = mean.reshape(-1, 1)

    n_data, n_targets = y.shape

    if n_samples != n_data:
        raise ValueError(
            "Number of samples in X and y does not correspond:"
            " %d != %d" % (n_samples, n_data)
        )

    z = y - b - A @ mean

    return ravel, z, mean, cov, data_cov


def mud_sol(A, b, y=None, mean=None, cov=None, data_cov=None, return_pred=False):
    """
    For SWE problem, we are inverting N(0,1).
    This is the default value for `data_cov`.
    """
    ravel, z, mean, cov, _ = check_args(A, b, y, mean, cov, data_cov)
    inv_pred_cov = np.linalg.pinv(A @ cov @ A.T)
    update = cov @ A.T @ inv_pred_cov
    mud_point = mean + update @ z

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        mud_point = mud_point.ravel()

    if return_pred:
        return mud_point, update
    else:
        return mud_point


def updated_cov(X, init_cov=None, data_cov=None):
    """
    We start with the posterior covariance from ridge regression
    Our matrix R = init_cov^(-1) - X.T @ pred_cov^(-1) @ X
    replaces the init_cov from the posterior covariance equation.
    Simplifying, this is given as the following, which is not used
    due to issues of numerical stability (a lot of inverse operations).

    up_cov = (X.T @ np.linalg.inv(data_cov) @ X + R )^(-1)
    up_cov = np.linalg.inv(\
        X.T@(np.linalg.inv(data_cov) - inv_pred_cov)@X + \
        np.linalg.inv(init_cov) )

    We return the updated covariance using a form of it derived
    which applies Hua's identity in order to use Woodbury's identity.

    >>> updated_cov(np.eye(2))
    array([[1., 0.],
           [0., 1.]])
    >>> updated_cov(np.eye(2)*2)
    array([[0.25, 0.  ],
           [0.  , 0.25]])
    >>> updated_cov(np.eye(3)[:, :2]*2, data_cov=np.eye(3))
    array([[0.25, 0.  ],
           [0.  , 0.25]])
    >>> updated_cov(np.eye(3)[:, :2]*2, init_cov=np.eye(2))
    array([[0.25, 0.  ],
           [0.  , 0.25]])
    """
    if init_cov is None:
        init_cov = np.eye(X.shape[1])
    else:
        assert X.shape[1] == init_cov.shape[1]

    if data_cov is None:
        data_cov = np.eye(X.shape[0])
    else:
        assert X.shape[0] == data_cov.shape[1]

    pred_cov = X @ init_cov @ X.T
    inv_pred_cov = np.linalg.pinv(pred_cov)
    # pinv b/c inv unstable for rank-deficient A

    # Form derived via Hua's identity + Woodbury
    K = init_cov @ X.T @ inv_pred_cov
    up_cov = init_cov - K @ (pred_cov - data_cov) @ K.T

    return up_cov


def mud_sol_alt(A, b, y=None, mean=None, cov=None, data_cov=None, return_pred=False):
    """
    Doesn't use R directly, uses new equations.
    This presents the equation as a rank-k update
    to the error of the initial estimate.
    """
    ravel, z, mean, cov, data_cov = check_args(A, b, y, mean, cov, data_cov)
    up_cov = updated_cov(X=A, init_cov=cov, data_cov=data_cov)
    update = up_cov @ A.T @ np.linalg.inv(data_cov)
    mud_point = mean + update @ z

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        mud_point = mud_point.ravel()

    if return_pred:
        return mud_point, update
    else:
        return mud_point


def map_sol(A, b, y=None, mean=None, cov=None, data_cov=None, w=1, return_pred=False):
    ravel, z, mean, cov, data_cov = check_args(A, b, y, mean, cov, data_cov)
    inv = np.linalg.inv
    post_cov = inv(A.T @ inv(data_cov) @ A + w * inv(cov))
    update = post_cov @ A.T @ inv(data_cov)
    map_point = mean + update @ z

    if ravel:
        # When y was passed as a 1d-array, we flatten the coefficients.
        map_point = map_point.ravel()

    if return_pred:
        return map_point, update
    else:
        return map_point


def performEpoch(A, b, y, initial_mean, initial_cov, data_cov=None, idx=None):
    dim_out = A.shape[0]
    mud_chain = []

    _mean = initial_mean
    mud_chain.append(_mean)
    if idx is None:
        idx = range(dim_out)
    for i in idx:
        _A = A[i, :].reshape(1, -1)
        _b = b[i]
        _y = y[i]
        _mud_sol = mud_sol(_A, _b, _y, _mean, initial_cov, data_cov=None)
        mud_chain.append(_mud_sol)
        _mean = mud_chain[-1]
    return mud_chain


def iterate(A, b, y, initial_mean, initial_cov, data_cov=None, num_epochs=1, idx=None):
    chain = performEpoch(A, b, y, initial_mean, initial_cov, data_cov, idx)
    for _ in range(1, num_epochs):
        chain += performEpoch(A, b, y, chain[-1], initial_cov, data_cov, idx)

    return chain


def mud_problem(lam, qoi, qoi_true, domain, sd=0.05, num_obs=None, split=None):
    """
    Wrapper around mud problem, takes in raw qoi + synthetic data and
    performs WME transformation, instantiates solver object
    """
    if lam.ndim == 1:
        lam = lam.reshape(-1, 1)

    if qoi.ndim == 1:
        qoi = qoi.reshape(-1, 1)
    dim_output = qoi.shape[1]

    if num_obs is None:
        num_obs = dim_output
    elif num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    elif num_obs > dim_output:
        raise ValueError("num_obs must be <= dim(qoi)")

    # TODO: handle empty sd -> take it from the data.
    # TODO: swap for data + leave noise generation separate. no randomness in method.
    noise = np.random.randn(num_obs) * sd
    if split is None:
        # this is our data processing step.
        data = qoi_true[0:num_obs] + noise
        q = wme(qoi[:, 0:num_obs], data, sd).reshape(-1, 1)
    else:  # vector-valued QoI map. TODO: assert dimensions <= input_dim
        q = []
        for qoi_indices in split:
            _q = qoi_indices[qoi_indices < num_obs]
            _qoi = qoi[:, _q]
            _data = np.array(qoi_true)[_q] + noise[_q]
            _newqoi = wme(_qoi, _data, sd)
            q.append(_newqoi)
        q = np.vstack(q).T
    # this implements density-based solutions, mud point method
    d = DensityProblem(lam, q, domain)
    return d


def map_problem(lam, qoi, qoi_true, domain, sd=0.05, num_obs=None, log=False):
    """
    Wrapper around map problem, takes in raw qoi + synthetic data and
    instantiates solver object
    """
    if lam.ndim == 1:
        lam = lam.reshape(-1, 1)

    if qoi.ndim == 1:
        qoi = qoi.reshape(-1, 1)
    dim_output = qoi.shape[1]

    if num_obs is None:
        num_obs = dim_output
    elif num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    elif num_obs > dim_output:
        raise ValueError("num_obs must be <= dim(qoi)")

    # this is our data processing step.
    data = qoi_true[0:num_obs] + np.random.randn(num_obs) * sd
    #     likelihood = dists.norm(loc=qoi[:, :num_obs], scale=sd)
    likelihood = dists.norm(loc=data, scale=sd)

    # this implements bayesian likelihood solutions, map point method
    b = BayesProblem(lam, qoi[:, 0:num_obs], domain)
    b.set_likelihood(likelihood, log=log)
    return b


if __name__ == "__main__":
    run()
