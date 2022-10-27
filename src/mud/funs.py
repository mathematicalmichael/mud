# -*- coding: utf-8 -*-
"""
Python console script for `mud`, installed with
`pip install .` or `python setup.py install`
"""

import logging

import numpy as np
from scipy.stats import distributions as dists  # type: ignore

from mud.base import (
    BayesProblem,
    DensityProblem,
    IterativeLinearProblem,
    LinearGaussianProblem,
    SpatioTemporalProblem,
)

_logger = logging.getLogger(__name__)


def wme(predictions, data, sd=None):
    """
    Calculates Weighted Mean Error (WME) functional.

    Parameters
    ----------
    predictions: numpy.ndarray of shape (n_samples, n_features)
        Predicted values against which data is compared.
    data: list or numpy.ndarray of shape (n_features, 1)
        Collected (noisy) data
    sd: float, optional
        Standard deviation

    Returns
    -------
    numpy.ndarray of shape (n_samples, 1)

    """
    sd = np.std(data) if sd is None else sd
    predictions = predictions.reshape(1, -1) if predictions.ndim == 1 else predictions
    N = predictions.shape[1]
    if N != len(data):
        raise ValueError(f"Predictions and data dim mismatch {N} != {len(data)}")

    return np.sum((1 / (sd * np.sqrt(N))) * (predictions - data), axis=1)


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


def lin_prob(A, b, y=None, mean=None, cov=None, data_cov=None, alpha=None):
    """
    Linear Gaussian Problem Solver Entrypoint
    """
    lin_prob = LinearGaussianProblem(
        A,
        b=b,
        y=y,
        mean_i=mean,
        cov_i=cov,
        cov_o=data_cov,
        alpha=alpha,
    )
    return lin_prob


def mud_sol(A, b, y=None, mean=None, cov=None, data_cov=None):
    """
    For SWE problem, we are inverting N(0,1).
    This is the default value for `data_cov`.
    """
    lp = lin_prob(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov)
    mud_pt = lp.solve(method="mud")
    mud_pt = mud_pt if np.array(y).ndim > 1 else mud_pt.ravel()

    return mud_pt


def mud_sol_with_cov(A, b, y=None, mean=None, cov=None, data_cov=None):
    """
    Doesn't use R directly, uses new equations.
    This presents the equation as a rank-k update
    to the error of the initial estimate.
    """
    lp = lin_prob(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov)
    mud_pt = lp.solve(method="mud_alt")
    mud_pt = mud_pt if np.array(y).ndim > 1 else mud_pt.ravel()

    return mud_pt, lp.up_cov


def map_sol(A, b, y=None, mean=None, cov=None, data_cov=None, w=1):
    """MAP Linear Gaussian Problem Solve"""
    lp = lin_prob(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov, alpha=w)
    map_pt = lp.solve(method="map")
    map_pt = map_pt if np.array(y).ndim > 1 else map_pt.ravel()

    return map_pt


def map_sol_with_cov(A, b, y=None, mean=None, cov=None, data_cov=None, w=1):
    """MAP Linear Gaussian Problem Solve"""
    lp = lin_prob(A, b, y=y, mean=mean, cov=cov, data_cov=data_cov, alpha=w)
    map_pt = lp.solve(method="map")
    map_pt = map_pt if np.array(y).ndim > 1 else map_pt.ravel()

    return map_pt, lp.cov_p


def iter_lin_solve(
    A,
    b,
    y=None,
    mean=None,
    cov=None,
    data_cov=None,
    method="mud",
    num_epochs=1,
    idx_order=None,
):
    """
    Iterative Linear Gaussian Problem Solver Entrypoint
    """
    lin_prob = IterativeLinearProblem(
        A,
        b=b,
        y=y,
        mean_i=mean,
        cov_i=cov,
        cov_o=data_cov,
        alpha=1.0,
        idx_order=idx_order,
    )
    res = lin_prob.solve(num_epochs=num_epochs, method=method)

    return res


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


def data_prob(
    lam,
    qoi,
    qoi_true=None,
    measurements=None,
    std_dev=None,
    sample_dist="uniform",
    domain=None,
    lam_ref=None,
    times=None,
    sensors=None,
    idxs=None,
    method="wme",
    init_dist=dists.uniform(loc=0, scale=1),
):
    """
    Data-Constructed Map Solve

    Wrapper around SpatioTemporalProblem class to create and solve a MUD problem
    by first aggregating observed and siumalated data in a data-constructed
    qoi map.
    """
    data = {
        "lam": lam,
        "data": qoi,
        "true_vals": qoi_true,
        "measurements": measurements,
        "std_dev": std_dev,
        "sample_dist": sample_dist,
        "domain": domain,
        "lam_ref": lam_ref,
        "times": times,
        "sensors": sensors,
    }
    sp_prob = SpatioTemporalProblem()
    sp_prob.load(data)
    D = sp_prob.mud_problem(method=method)
    D.set_initial(init_dist)
    D.set_observed(dists.norm(0, 1))  # always N(0,1) for WME map
    D.estimate()

    return D


def map_problem(lam, qoi, qoi_true, domain, sd=0.05, num_obs=None, log=False):
    """
    Wrapper around map problem, takes in raw qoi + synthetic data and
    instantiates solver object
    """
    lam = lam.reshape(-1, 1) if lam.ndim == 1 else lam
    qoi = qoi.reshape(-1, 1) if qoi.ndim == 1 else qoi

    dim_output = qoi.shape[1]
    if num_obs is None:
        num_obs = dim_output
    elif num_obs < 1:
        raise ValueError("num_obs must be >= 1")
    elif num_obs > dim_output:
        raise ValueError("num_obs must be <= dim(qoi)")

    data = qoi_true[0:num_obs] + np.random.randn(num_obs) * sd
    likelihood = dists.norm(loc=data, scale=sd)
    b = BayesProblem(lam, qoi[:, 0:num_obs], domain)
    b.set_likelihood(likelihood, log=log)
    return b


def mud_problem(
    lam, qoi, qoi_true, domain, sd=0.05, num_obs=None, split=None, weights=None
):
    """
    Wrapper around mud problem, takes in raw qoi + synthetic data and
    performs WME transformation, instantiates solver object.
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
    d = DensityProblem(lam, q, domain, weights=weights)
    return d
