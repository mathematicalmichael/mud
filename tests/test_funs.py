# -*- coding: utf-8 -*-

import unittest
import mud.funs as mdf
import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


class TestIdentityInitialCovariance(unittest.TestCase):

    def setUp(self):
        self.A = np.eye(2)
        self.C = np.eye(2)

    def test_that_R_inverse_is_zero(self):
        assert np.linalg.norm(mdf.makeRi(self.A, self.C)) < 1E-14

    def test_solutions_with_orthogonal_map(self):
        # Arrange
        t = np.random.randn(2,1)
        A = np.random.randn(2,2)
        b = np.random.randn(2,1)
        # m = np.zeros(2,1)
        c = self.C

        # Act
        y = A@t + b
        sol = mdf.mud_sol(A, b, y, cov=c)
        err = sol - t
        # Assert
        assert np.linalg.norm(err) < 1E-8
