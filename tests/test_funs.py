# -*- coding: utf-8 -*-

import unittest

import numpy as np

import mud.funs as mdf


class TestIdentityInitialCovariance(unittest.TestCase):
    def setUp(self):
        self.A = np.random.randn(2, 2)
        self.id = np.eye(2)

    def test_solutions_with_orthogonal_map(self):
        # Arrange
        t = np.random.randn(2, 1)
        A = self.A
        b = np.random.randn(2, 1)
        c = self.id

        # Act
        y = A @ t + b
        sol_mud = mdf.mud_sol(A, b, y, cov=c)
        sol_mud_alt, updated_cov = mdf.mud_sol_with_cov(A, b, y, cov=c)
        sol_map = mdf.map_sol(A, b, y, cov=c)
        sol_map_alt, posterior_cov = mdf.map_sol_with_cov(A, b, y, cov=c)

        err_mud = sol_mud - t
        err_alt = sol_mud_alt - t
        err_map = sol_map - t

        # Assert
        assert np.linalg.norm(sol_map - sol_map_alt) < 1e-12
        assert np.linalg.norm(sol_mud - sol_mud_alt) < 1e-6
        assert np.linalg.norm(err_mud) < 1e-6
        assert np.linalg.norm(err_alt) < 1e-6
        assert np.linalg.norm(err_mud) < np.linalg.norm(err_map)

    def test_updated_cov_has_R_equal_zero_for_full_rank_A(self):
        up_cov = mdf.updated_cov(self.A, self.id, self.id)
        absolute_error = np.linalg.norm(up_cov - np.linalg.inv(self.A.T @ self.A))
        assert absolute_error / len(up_cov) < 1e-8


class TestWME(unittest.TestCase):
    def setUp(self):
        self.d = np.random.rand(10)
        self.A = np.tile(self.d, (1, 1))

    def test_wme(self):
        # all residuals are zero
        wme = mdf.wme(self.A, self.d, sd=1)
        assert len(wme) == self.A.shape[0]
        assert sum(wme) == 0

    def test_wme_with_no_sd(self):
        # all residuals are one. answer should be N/(sd*sqrt(N))
        wme = mdf.wme(1 + self.A, self.d)
        assert len(wme) == self.A.shape[0]
        assert np.allclose(wme[0], wme[-1])  # all samples should be equal
        ans = len(self.d) / (np.sqrt(len(self.d)) * np.std(self.d))
        assert abs(ans - wme[0]) < 1e-12


class TestWME_20(TestWME):
    def setUp(self):
        self.d = np.random.rand(20)
        self.A = np.tile(self.d, (100, 1))


# TODO: test wme works with data of shape (n_features, 1), (1, n_features), and list
