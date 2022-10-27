# -*- coding: utf-8 -*-
"""
    Conftest for mud tests

    Contains fixtures for common objects and data structures used in MUD tests.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from mud.examples.simple import identity_1D_bayes_prob, identity_1D_density_prob

def_test_dir = Path(__file__).parent / ".test_dir"


@pytest.fixture()
def test_dir():
    if def_test_dir.exists():
        shutil.rmtree(def_test_dir)
    def_test_dir.mkdir(exist_ok=True)
    test_dir_path = Path(def_test_dir).absolute()
    yield test_dir_path
    shutil.rmtree(test_dir_path, ignore_errors=True)


@pytest.fixture
def dist_wo_weights():
    class Dist:
        @classmethod
        def pdf(self, x, **kwargs):
            return np.zeros(0)

    return Dist


@pytest.fixture
def identity_problem_map_1D():
    return identity_1D_bayes_prob()


@pytest.fixture
def identity_problem_mud_1D():
    return identity_1D_density_prob()


@pytest.fixture
def identity_problem_mud_1D_equal_weights():
    num_samples = 5000
    return identity_1D_density_prob(
        num_samples=num_samples,
        weights=np.ones(num_samples),
    )


@pytest.fixture
def identity_problem_mud_1D_bias_weights():
    num_samples = 5000
    weights = np.ones(num_samples)
    D = identity_1D_density_prob(
        num_samples=num_samples,
        weights=np.ones(num_samples),
    )
    weights[D.X[:, 0] < 0.2] = 0.1
    weights[D.X[:, 0] > 0.8] = 0.1
    D.set_weights(weights)
    return D


@pytest.fixture
def identity_problem_mud_1D_domain():
    """
    Test setting weights, with and without normalization.
    """
    # Arrange
    num_samples = 5000
    D = identity_1D_density_prob(
        num_samples=num_samples,
        domain=[0, 1],
    )
    return D
