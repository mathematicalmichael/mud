# -*- coding: utf-8 -*-
"""
Python console script for `mud`, installed with
`pip install .` or `python setup.py install`
"""

import logging

import numpy as np
from scipy.stats import distributions as dists

from mud.base import (BayesProblem, IterativeLinearProblem,
                      LinearGaussianProblem, SpatioTemporalProblem)

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


def lin_solv(A, b, y=None, mean=None, cov=None, data_cov=None, method="mud"):
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
        alpha=1.0,
    )
    res = lin_prob.solve(method=method)

    return res


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


def iter_data_prob(
    lam,
    qoi,
    data,
    domain,
    sd=0.05,
    weights=None,
    num_it=1,
    pca_components=None,
    init_dist=dists.uniform(loc=0, scale=1),
):
    """
    Iterative MUD Problem.

    TODO: implement using SpatioTemporalProblem class
    """
    pass


#
#     if lam.ndim == 1:
#         lam = lam.reshape(-1, 1)
#
#     if qoi.ndim == 1:
#         qoi = qoi.reshape(-1, 1)
#     num_obs = qoi.shape[1]
#
#     # Split qoi values for each sample and observed data into equal size groups
#     qoi_splits = np.array_split(np.copy(qoi), num_it, axis=1)
#     data_splits = np.array_split(np.copy(data), num_it)
#
#     mud_res = []
#     pca_res = []
#     for i in range(num_it):
#         if pca_components:
#             # Compute residutals - Dividing by std deviation here?
#             res = (qoi_splits[i] - data_splits[i]) / sd
#
#             # Standarize and perform linear PCA
#             sc = StandardScaler()
#             pca = PCA(n_components=pca_components)
#             X_train = pca.fit_transform(sc.fit_transform(res))
#             pca_res.append((pca, X_train))
#
#             q = np.array([wme(v*qoi_splits[i],
#                 v*data_splits[i], sd) for v in pca.components_]).T
#         else:
#             # Select slice of data
#             q = wme(qoi_splits[i], data_splits[i], sd).reshape(-1, 1)
#
#         # Solve MUD Density problem, using weights from previous iteration.
#         d = DensityProblem(lam, q, domain, weights=weights)
#         _ = d.estimate()
#
#         # Add r ratio from this iteration to weight chain for next iteration.
#         weights = d._r if i==0 else np.vstack([weights, d._r])
#         mud_res.append(d)
#
#     return mud_res, pca_res
