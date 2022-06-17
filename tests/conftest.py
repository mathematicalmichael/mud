# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for mud.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import numpy as np
import pytest
from scipy.stats import distributions as ds

from mud.base import BayesProblem
from mud.examples import identity_uniform_1D_density_prob


@pytest.fixture
def dist_wo_weights():
    class Dist:
        @classmethod
        def pdf(self, x, **kwargs):
            return np.zeros(0)

    return Dist


@pytest.fixture
def problem_generator_identity_1D():
    return identity_uniform_1D_density_prob


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
