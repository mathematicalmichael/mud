# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

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


def test_identity_mud_1D_with_normalized_weights(problem_generator_identity_1D):
    # Arrange
    D = problem_generator_identity_1D(
        num_samples=10, wme_map=True, analytical_pred=False
    )

    # Act
    D.set_weights(np.ones(10), normalize=True)

    # Assert
    assert all([x == 0.1 for x in D._weights])


def test_identity_mud_problem_1D_plot_params(identity_problem_mud_1D):
    # Arrange
    D = identity_problem_mud_1D
    fig, ax = plt.subplots(1, 2)

    # Act
    mud_point = D.estimate()
    D.plot_param_space(ax=ax[0])
    D.plot_param_space(ax=ax[1], in_opts=None, up_opts=None, win_opts=None)
    mud_alt = ax[0].lines[1].get_xdata()[np.argmax(ax[0].lines[1].get_ydata())]

    # Assert - Proper number of lines plotted and diff between plotted
    # and computed mud point is small
    assert len(ax[0].lines) == 3
    assert len(ax[1].lines) == 0
    assert np.abs(mud_alt - mud_point) < 0.1


def test_identity_mud_problem_1D_plot_obs(identity_problem_mud_1D):
    # Arrange
    D = identity_problem_mud_1D
    fig, ax = plt.subplots(1, 2)

    # Act
    _ = D.estimate()
    D.plot_obs_space(ax=ax[0])
    D.plot_obs_space(ax=ax[1], pr_opts=None, pf_opts=None, ob_opts=None)
    average_diff = np.average(
        np.abs(ax[0].lines[2].get_ydata() - ax[0].lines[0].get_ydata())
    )

    # Assert - Proper number of lines plotted and average diff between plots
    # observed and push-forward densities is small, since they should match.
    assert len(ax[0].lines) == 3
    assert len(ax[1].lines) == 0
    assert average_diff < 0.1
