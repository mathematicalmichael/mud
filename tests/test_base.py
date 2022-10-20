# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


def test_identity_1D_mud(identity_problem_mud_1D):
    # Arrange
    D = identity_problem_mud_1D

    # Act
    mud_point = D.estimate()
    ratio = D._r

    # Assert
    assert 1 == D.n_params
    assert 1 == D.n_features
    assert 2000 == D.n_samples
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

def test_identity_mud_2D_with_normalized_weights(identity_problem_mud_1D_equal_weights):
    # Arrange
    D = identity_problem_mud_1D_equal_weights

    # Act
    D.set_weights(np.ones(D.n_samples), normalize=True)

    # Assert
    assert (np.sum(D._weights) - 70.710) < 0.01


def test_identity_mud_problem_1D_plot_params(identity_problem_mud_1D_domain,
                                             identity_problem_mud_1D_equal_weights):
    # Arrange
    D_base = identity_problem_mud_1D_domain
    D_weights = identity_problem_mud_1D_equal_weights

    # Act
    mud_point_base = D_base.estimate()
    mud_point_weights = D_weights.estimate()
    ratio_base = D_base._r
    ratio_weights = D_weights._r
    ax1 = D_base.plot_param_space(true_val=[0.5])
    ax2 = D_weights.plot_param_space(win_opts={}, in_opts=None,
                                     up_opts=None, mud_opts=None)

    # Assert
    assert np.round(mud_point_base, 1) == 0.5
    assert np.abs(np.mean(ratio_base) - 1) < 0.5
    assert np.round(mud_point_weights, 1) == 0.5
    assert np.abs(np.mean(ratio_weights) - 1) < 0.2
    val = 1/np.linalg.norm(np.ones(10))
    assert len(ax1.get_lines()) == 4
    for l in ax1.get_lines():
        if 'init' in l.get_label():
            # uniform initial, all same values
            assert 1 == int(list(set(l.get_ydata()))[0])
        if 'update' in l.get_label():
            # peak of dist should be around the center of [0, 1] domain
            assert np.abs(0.5 - l.get_xdata()[np.argmax(l.get_ydata())]) < 0.1
        if 'MUD' in l.get_label():
            # Mud value should be verticle line (same x) around 0.5
            assert np.abs(l.get_xdata()[0] - 0.5) < 0.1
        if 'dagger' in l.get_label():
            assert np.abs(l.get_xdata()[0] - 0.5) < 0.0001
    assert len(ax2.get_lines()) == 1
    assert 'tilde' in ax2.get_lines()[0].get_label()
    assert np.abs(np.mean(ax2.get_lines()[0].get_ydata()) - 1) < 0.25


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


def test_identity_mud_problem_1D_plot_obs(identity_problem_mud_1D_domain):
    # Arrange
    D_base = identity_problem_mud_1D_domain

    # Act
    mud_point_base = D_base.estimate()
    ratio_base = D_base._r
    ax1 = D_base.plot_obs_space()
    ax2 = D_base.plot_obs_space(ob_opts=None, pr_opts=None, pf_opts=None)
    average_diff = np.average(
        np.abs(ax1.lines[2].get_ydata() - ax1.lines[0].get_ydata())
    )

    # Assert
    assert np.round(mud_point_base, 1) == 0.5
    assert np.abs(np.mean(ratio_base) - 1) < 0.2
    assert len(ax1.get_lines()) == 3
    assert average_diff < 0.1
    for l in ax1.get_lines():
        if 'obs' in l.get_label():
            # Observed distribution should be normal peaked around true val 0.5
            assert np.abs(0.5 - l.get_xdata()[np.argmax(l.get_ydata())]) < 0.1
        if 'pred' in l.get_label():
            # Predicted should be same as initial since identity map
            vals = list(set(l.get_ydata()))
            vals.sort()
            assert 0 == vals[0]
            assert 0.01 > (1 - vals[-1])
        if 'pf-pr' in l.get_label():
            assert l.get_xdata()[np.argmax(l.get_ydata())]
    assert len(ax2.get_lines()) == 0


def test_identity_map_problem_1D(identity_problem_map_1D):
    # Arrange
    D = identity_problem_map_1D

    # Act
    map_point = D.estimate()

    # Assert
    assert np.round(map_point, 1) == 0.5

