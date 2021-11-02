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
    updated_density = D._up
    ratio = D._r

    # Assert
    assert np.round(mud_point, 1) == 0.5
    assert np.sum(updated_density) > 0
    assert np.mean(ratio) > 0


def test_we_can_set_weights_in_predicted(identity_problem_mud_1D_equal_weights):
    """Mimicks existing usage in mud-examples"""
    # Arrange
    # weights were used for initialization
    D = identity_problem_mud_1D_equal_weights
    D.set_initial()  # domain has been set -> uniform as default
    # want to make sure we can set weights on predicted
    weights = np.random.rand(D._n_samples)
    D.set_predicted(weights=weights)

    # Act
    mud_point = D.estimate()
    updated_density = D._up
    ratio = D._r

    # Assert
    # ensure weights were set correctly
    assert np.linalg.norm(weights - D._weights) == 0
    assert np.round(mud_point, 1) == 0.5
    assert np.sum(updated_density) > 0
    assert np.mean(ratio) > 0


def test_identity_mud_1D_with_equal_weights(identity_problem_mud_1D_equal_weights):
    # Arrange
    D = identity_problem_mud_1D_equal_weights

    # Act
    mud_point = D.estimate()
    updated_density = D._up
    ratio = D._r

    # Assert
    assert np.round(mud_point, 1) == 0.5
    assert np.sum(updated_density) > 0
    assert np.mean(ratio) > 0


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
