# -*- coding: utf-8 -*-

import unittest
import mud.norm as mdn
import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


class TestNorm(unittest.TestCase):

    def test_identity_induced_norm(self):
        # Assign
        X = np.random.rand(2,1)
        mat = np.eye(2)

        # Act
        result = mdn.mynorm(X, mat)
        check = np.linalg.norm(X)**2

        # Assert
        assert isinstance(result,float)
        self.assertAlmostEqual(result, check, 12)

    def test_scaled_identity_induced_norm(self):
        # iterate over a few scaling factors
        for n in range(2, 7):
            # Assign
            X = np.random.rand(2,1)
            mat = np.diag([n, n]) # scale norm by 1/n

            # Act
            result = mdn.mynorm(X, mat)*n
            check = np.linalg.norm(X)**2

            # Assert
            self.assertAlmostEqual(result, check, 12)

class TestFunctionals(unittest.TestCase):
    def setUp(self):
        self.operator = np.random.rand(1,2)
        self.inputs = np.random.rand(1,2)
        self.data = np.array([0])
        self.imean = np.array([0, 0])
        self.icov = 1
        self.omean = np.array([0])
        self.ocov = 1

    def test_full_functional_with_random_input(self):
        result = mdn.full_functional(self.operator,
                                     self.inputs,
                                     self.data,
                                     self.imean,
                                     self.icov,
                                     self.omean,
                                     self.ocov
                                    )
        assert result > 0

    def test_full_functional_with_zero_input(self):
        result = mdn.full_functional(self.operator,
                                     self.inputs*0,
                                     self.data,
                                     self.imean,
                                     self.icov,
                                     self.omean,
                                     self.ocov
                                    )
        assert result == 0