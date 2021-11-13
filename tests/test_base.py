# -*- coding: utf-8 -*-

import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


def test_identity_mud_problem_1D(identity_problem_mud_1D):
    # Arrange
    D = identity_problem_mud_1D

    # Act
    mud_point = D.estimate()
    ratio = D._r

    # Assert
    assert np.round(mud_point, 1) == 0.5
    assert np.abs(np.mean(ratio) - 1) < 0.2


def test_we_can_set_weights_in_predicted(identity_problem_mud_1D_equal_weights):
    """
    Mimicks existing usage in mud-examples.
    We want to be able to pass a `weights` keyword to the `set_predicted` method
    even if `weights` were passed during class initialization.

    This test checks that the weights are saved correctly and that no error is raised
    due to keyword handling.
    """
    # Arrange
    # weights were used for initialization
    D = identity_problem_mud_1D_equal_weights
    D.set_initial()  # domain has been set -> uniform as default
    # want to make sure we can set weights on predicted and ensure they are saved.
    weights = np.random.rand(D._n_samples)

    # Act
    # also checking that `bw_method` can be passed to `gaussian_kde`
    D.set_predicted(weights=weights, bw_method="scott")

    # Assert
    # ensure weights were set correctly, we don't care about any other results here.
    assert np.linalg.norm(weights - D._weights) == 0


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
    weights_ones = np.ones(D._n_samples)
    weights_normalized = weights_ones / D._n_samples

    # Act
    D.set_predicted(weights=weights_ones)
    predicted_ones = D._pr  # TODO: copy?
    D.set_predicted(weights=weights_normalized)
    predicted_normalized = D._pr

    # Assert
    # ensure weights do not impact evaluation of predicted density.
    assert np.linalg.norm(predicted_ones - predicted_normalized) < 1e-14


def test_identity_mud_1D_with_equal_weights(identity_problem_mud_1D_equal_weights):
    # Arrange
    D = identity_problem_mud_1D_equal_weights

    # Act
    mud_point = D.estimate()
    ratio = D._r

    # Assert
    assert np.round(mud_point, 1) == 0.5
    assert np.abs(np.mean(ratio) - 1) < 0.2


def test_identity_mud_1D_with_biased_weights(identity_problem_mud_1D_bias_weights):
    # Arrange
    D = identity_problem_mud_1D_bias_weights

    # Act
    mud_point = D.estimate()
    updated_density = D._up
    ratio = D._r

    # Assert
    assert np.round(mud_point, 1) == 0.5
    assert np.sum(updated_density) > 0
    assert np.mean(ratio) > 0


def test_identity_map_problem_1D(identity_problem_map_1D):
    # Arrange
    D = identity_problem_map_1D

    # Act
    map_point = D.estimate()

    # Assert
    assert np.round(map_point, 1) == 0.5
