# -*- coding: utf-8 -*-

import unittest

import numpy as np

import mud.funs as mdf

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


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
        sol_mud = mdf.lin_solv(A, b, y, cov=c, method="mud")
        sol_alt = mdf.lin_solv(A, b, y, cov=c, method="mud_alt")
        sol_map = mdf.lin_solv(A, b, y, cov=c, method="map")

        err_mud = sol_mud - t
        err_alt = sol_alt - t
        err_map = sol_map - t

        # Assert
        assert np.linalg.norm(err_mud) < 1e-6
        assert np.linalg.norm(err_alt) < 1e-6
        assert np.linalg.norm(err_mud) < np.linalg.norm(err_map)


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

