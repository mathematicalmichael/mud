# -*- coding: utf-8 -*-

import unittest
import mud.funs as mdf
import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


class TestIdentityInitialCovariance(unittest.TestCase):

    def setUp(self):
        self.A = np.random.randn(2, 2)
        self.id = np.eye(2)

    def test_that_R_inverse_is_zero(self):
        assert np.linalg.norm(mdf.makeRi(self.A, self.id)) < 1E-8

    def test_solutions_with_orthogonal_map(self):
        # Arrange
        t = np.random.randn(2, 1)
        A = np.random.randn(2, 2)
        b = np.random.randn(2, 1)
        c = self.id

        # Act
        y = A @ t + b
        sol_mud = mdf.mud_sol(A, b, y, cov=c)
        sol_alt = mdf.mud_sol_alt(A, b, y, cov=c)
        # sol_map = mdf.map_sol(A, b, y, cov=100*c)

        err_mud = sol_mud - t
        err_alt = sol_alt - t
        # err_map = sol_map - t

        # Assert
        assert np.linalg.norm(err_mud) < 1E-8
        assert np.linalg.norm(err_alt) < 1E-8
        # assert np.linalg.norm(err_map) < 1E-6

    def test_updated_cov_has_R_equal_zero_for_full_rank_A(self):
        up_cov = mdf.updated_cov(self.A, self.id, self.id)
        assert np.linalg.norm(up_cov - np.linalg.inv(self.A.T @ self.A)) < 1E-6
