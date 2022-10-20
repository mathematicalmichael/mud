# -*- coding: utf-8 -*-
"""
Tests for mud examples

"""
import pytest
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from click.testing import CliRunner
from mud.cli import cli

__author__ = "Carlos del-Castillo-Negrete"
__copyright__ = "Carlos del-Castillo-Negrete"
__license__ = "mit"

def test_comparison_example():
  runner = CliRunner()
  result = runner.invoke(cli, ['examples', '-ns', 'comparison'])
  assert result.exit_code == 0

def test_contours_exampls():
  runner = CliRunner()
  result = runner.invoke(cli, ['examples', '-ns', 'contours'])
  assert result.exit_code == 0

def test_high_dim_linear():
  runner = CliRunner()
  result = runner.invoke(cli, ['examples', '-ns', 'high-dim-linear'])
  assert result.exit_code == 0

def test_wme_covariance():
  runner = CliRunner()
  result = runner.invoke(cli, ['examples', '-ns', 'wme-covariance'])
  assert result.exit_code == 0

def test_poisson_generate(test_dir):
  runner = CliRunner()
  result = runner.invoke(cli, ['examples', '--seed', '21',
                               'poisson-generate', '5', '2',
                               '--save_dir', test_dir])
  fname =  result.stdout[:-1].split('\n')[-1]

  with open(fname, 'rb') as fp:
      data = pickle.load(fp)

  assert result.exit_code == 0
  assert np.abs(data['true_vals'][0] + 0.48799728) < 0.001
  assert np.abs(data['true_vals'][1] - 0.00183782) < 0.001

def test_poisson_solve():
  runner = CliRunner()
  data = str(Path(__file__).parent / 'poisson_data')
  result = runner.invoke(cli, ['examples', '-ns', '--seed', '21', 'poisson-solve', data])
  assert result.exit_code == 0
  assert str(result.stdout) == "[-2.76754243 -1.6656349 ]\n"

def test_poisson_trials():
  runner = CliRunner()
  data = str(Path(__file__).parent / 'poisson_data')
  result = runner.invoke(cli, ['examples', '-ns', '--seed', '21',
                               'poisson-trials', data, '-n', '2'])
  assert result.exit_code == 0
  assert '0.018693404000' in str(result.stdout)
