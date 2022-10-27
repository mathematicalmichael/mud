# -*- coding: utf-8 -*-
"""
Tests for mud examples

"""
import pickle
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from mud.cli import cli


def test_comparison_example():
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "-ns", "comparison"])
    assert result.exit_code == 0


def test_contours_example():
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "-ns", "contours"])
    assert result.exit_code == 0


def test_high_dim_linear():
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "-ns", "high-dim-linear"])
    assert result.exit_code == 0


def test_wme_covariance():
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "-ns", "wme-covariance"])
    assert result.exit_code == 0


def test_poisson_generate(test_dir):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "examples",
            "--seed",
            "21",
            "--save-path",
            test_dir,
            "poisson-generate",
            "5",
            "2",
        ],
    )

    fname = result.stdout[:-1].split("\n")[-1]

    if "Unable to run fenics" not in result.stdout:
        with open(fname, "rb") as fp:
            data = pickle.load(fp)

        assert result.exit_code == 0
        assert np.abs(data["true_vals"][0] + 0.48799728) < 0.001
        assert np.abs(data["true_vals"][1] - 0.00183782) < 0.001


def test_poisson_solve():
    runner = CliRunner()
    data = str(Path(__file__).parent / "data" / "poisson_data")
    result = runner.invoke(
        cli, ["examples", "-ns", "--seed", "21", "poisson-solve", data]
    )
    assert result.exit_code == 0
    assert str(result.stdout) == "[-2.76754243 -1.6656349 ]\n"


def test_poisson_trials():
    runner = CliRunner()
    data = str(Path(__file__).parent / "data" / "poisson_data")
    result = runner.invoke(
        cli, ["examples", "-ns", "--seed", "21", "poisson-trials", data, "-n", "2"]
    )
    assert result.exit_code == 0
    assert "0.018693404000" in str(result.stdout)


def test_adcirc_solve():
    runner = CliRunner()
    data = str(Path(__file__).parent / "data" / "adcirc_data")
    result = runner.invoke(
        cli,
        [
            "examples",
            "-ns",
            "--seed",
            "21",
            "adcirc-solve",
            data,
            "-p",
            "all",
            "-t1",
            "2018-01-01T10:03:00.000000000",
            "-t2",
            "2018-01-01T12:33:00.000000000",
        ],
    )
    assert result.exit_code == 0
    assert "[0.05266253 0.00294599]" in str(result.stdout)


def test_mud_paper(test_dir):
    runner = CliRunner()
    result = runner.invoke(cli, ["examples", "--save-path", str(test_dir), "mud-paper"])
    assert result.exit_code == 0
