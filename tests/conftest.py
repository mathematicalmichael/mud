# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for mud.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest
from mud.base import DensityProblem, BayesProblem
from mud.funs import wme
from scipy.stats import distributions as ds
import numpy as np


@pytest.fixture
def problem_generator_identity_1D():
    def identity_uniform_1D(
        num_samples=2000, num_obs=20, y_true=0.5, noise=0.05, weights=None
    ):
        """
        Sets up an inverse problem using the unit domain and uniform distribution
        under an identity map. This is equivalent to studying a
        \"steady state\" signal over time, or taking repeated measurements
        of the same quantity to reduce variance in the uncertainty.
        """
        dist = ds.uniform(loc=0, scale=1)
        X = dist.rvs(size=(num_samples, 1))
        y_pred = np.repeat(X, num_obs, 1)
        # data is truth + noise
        y_observed = y_true * np.ones(num_obs) + noise * np.random.randn(num_obs)
        Y = wme(y_pred, y_observed, sd=noise)
        # analytical construction of predicted domain under identity map.
        y_domain = np.repeat(np.array([[0], [1]]), num_obs, 1)
        mn, mx = wme(y_domain, y_observed, sd=noise)
        loc, scale = mn, mx - mn
        dist = ds.uniform(loc=loc, scale=scale)

        D = DensityProblem(X, Y, np.array([[0, 1]]), weights=weights)
        D.set_predicted(dist)
        return D

    return identity_uniform_1D


@pytest.fixture
def identity_problem_mud_1D(problem_generator_identity_1D):
    return problem_generator_identity_1D()


@pytest.fixture
def identity_problem_map_1D():
    X = np.random.rand(1000, 1)
    num_observations = 50
    y_pred = np.repeat(X, num_observations, 1)
    y_true = 0.5
    noise = 0.05
    y_observed = y_true * np.ones(num_observations) + noise * np.random.randn(
        num_observations
    )
    B = BayesProblem(X, y_pred, np.array([[0, 1]]))
    B.set_likelihood(ds.norm(loc=y_observed, scale=noise))
    return B


@pytest.fixture
def identity_problem_mud_1D_equal_weights(problem_generator_identity_1D):
    num_samples = 5000
    return problem_generator_identity_1D(
        num_samples=num_samples,
        weights=np.ones(num_samples),
    )


@pytest.fixture
def identity_problem_mud_1D_bias_weights(problem_generator_identity_1D):
    num_samples = 5000
    weights = np.ones(num_samples)
    D = problem_generator_identity_1D(
        num_samples=num_samples,
        weights=np.ones(num_samples),
    )
    weights[D.X[:, 0] < 0.2] = 0.1
    weights[D.X[:, 0] > 0.8] = 0.1
    D.set_weights(weights)
    return D
