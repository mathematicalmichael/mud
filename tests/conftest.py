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
def identity_1D_50_wme():
    X = np.random.rand(100, 1)
    num_observations = 50
    y_pred = np.repeat(X, num_observations, 1)
    y_true = 0.5
    noise = 0.05
    y_observed = y_true * np.ones(num_observations) + noise * np.random.randn(
        num_observations
    )
    Y = wme(y_pred, y_observed, sd=noise)
    return (X, Y)


@pytest.fixture
def identity_problem_mud_1D(identity_1D_50_wme):
    X, Y = identity_1D_50_wme
    return DensityProblem(X, Y, np.array([[0, 1], [0, 1]]))


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
    B = BayesProblem(X, y_pred, np.array([[0, 1], [0, 1]]))
    B.set_likelihood(ds.norm(loc=y_observed, scale=noise))
    return B


@pytest.fixture
def identity_problem_mud_1D_equal_weights(identity_1D_50_wme):
    X, Y = identity_1D_50_wme
    weights = np.ones(X.shape[0])
    return DensityProblem(X, Y, np.array([[0, 1], [0, 1]]), weights=weights)


@pytest.fixture
def identity_problem_mud_1D_bias_weights(identity_1D_50_wme):
    X, Y = identity_1D_50_wme
    weights = np.ones(X.shape[0])
    weights[X[:, 0] < 0.2] = 0.1
    weights[X[:, 0] > 0.8] = 0.1
    return DensityProblem(X, Y, np.array([[0, 1], [0, 1]]), weights=weights)
