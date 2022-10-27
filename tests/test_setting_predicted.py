# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import distributions as ds  # type: ignore


def test_weights_in_predicted_with_no_distribution(
    identity_problem_mud_1D_equal_weights,
):
    """
    Mimicks existing usage in mud-examples.
    We want to be able to pass a `weights` keyword to the `set_predicted` method
    even if `weights` were passed during class initialization.

    This test checks that the weights are saved correctly and that no error is raised
    due to keyword handling.
    """
    # Arrange
    # weights were used for initialization
    # small sample size for speed
    D = identity_problem_mud_1D_equal_weights
    D.set_initial()  # domain has been set -> uniform as default
    # want to make sure we can set weights on predicted and ensure they are saved.
    weights = list(np.random.rand(D.n_samples))

    # Act
    # also checking that `bw_method` can be passed to `gaussian_kde`
    D.set_predicted(weights=weights, bw_method="scott")

    # Assert
    # ensure weights were set correctly, we don't care about any other results here.
    assert np.linalg.norm(weights - D._weights) == 0


def test_weights_in_predicted_with_wrong_distribution(
    identity_problem_mud_1D, dist_wo_weights
):
    """
    Ensures that if we pass weights to a distribution that does not require them,
    they are safely ignored but still saved.
    """
    # Arrange
    # weights were used for initialization
    # small sample size for speed
    D = identity_problem_mud_1D

    # want to make sure we can set weights on predicted and ensure they are saved.
    weights = np.random.rand(D.n_samples)

    # Act
    D.set_predicted(distribution=dist_wo_weights, weights=weights)

    # Assert
    # ensure weights were set correctly, we don't care about any other results here.
    assert np.linalg.norm(weights - D._weights) == 0
    # dummy distribution returns empty list as pdf evaluation.
    assert isinstance(D._pr, np.ndarray) and len(D._pr) == 0


def test_kwds_in_predicted_with_distribution(identity_problem_mud_1D):
    """
    Ensures that if we pass kwds to an unfrozen distribution that does requires them,
    they are passed to the pdf function.
    """
    # Arrange
    # small sample size for speed
    D = identity_problem_mud_1D

    # Act
    D.set_predicted(distribution=ds.uniform, loc=100, scale=2)

    # Assert
    assert D._pr is not None
    assert np.linalg.norm(D._pr) == 0  # no mutual support


def test_equal_weights_in_predicted_changes_nothing(
    identity_problem_mud_1D_equal_weights,
):
    """
    Ensures that the evaluation of predicted samples is equivalent when
    passing weight vectors to `gaussian_kde` which assign equal weights to all samples.
    """
    # Arrange
    # weights were used for initialization
    D = identity_problem_mud_1D_equal_weights
    D.set_initial()  # domain has been set -> uniform as default
    # want to make sure we can set weights on predicted and ensure they are saved.
    weights_ones = np.ones(D.n_samples)
    weights_normalized = weights_ones / D.n_samples

    # Act
    D.set_predicted(weights=weights_ones)
    predicted_ones = D._pr  # TODO: copy?
    D.set_predicted(weights=weights_normalized)
    predicted_normalized = D._pr

    # Assert
    # ensure weights do not impact evaluation of predicted density.
    assert np.linalg.norm(predicted_ones - predicted_normalized) < 1e-12
