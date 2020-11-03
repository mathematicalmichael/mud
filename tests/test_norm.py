# -*- coding: utf-8 -*-

import unittest
import mud.norm as mdn
import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


class TestNorm(unittest.TestCase):

    def test_identity_induced_norm_on_vector(self):
        # Arrange
        X = np.random.rand(2,1) # single vector
        mat = np.eye(2)

        # Act
        result = mdn.mynorm(X, mat)
        check = np.linalg.norm(X, axis=0)**2

        # Assert
        assert isinstance(result, np.ndarray)
        self.assertAlmostEqual(result[0], check[0], 12)

    def test_scaled_identity_induced_norm(self):
        # iterate over a few scaling factors
        for n in range(2, 7):
            # Arrange
            X = np.random.rand(2,1)
            mat = np.diag([n, n]) # scale norm by 1/n

            # Act
            result = mdn.mynorm(X, mat)*n
            check = np.linalg.norm(X)**2

            # Assert
            self.assertAlmostEqual(result[0], check, 12)

class TestFunctionals_2to1(unittest.TestCase):
    def setUp(self):
        self.idim = 2
        self.operator = np.random.rand(1, self.idim)
        self.inputs = np.random.rand(1,self.idim)
        self.data = np.array([0])
        self.imean = np.array([0]*self.idim)
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

    def test_types_of_covariance_arguments_data(self):
        for oc in [1, np.array([[1]])]:
            result = mdn.norm_data(self.operator,
                                        self.inputs,
                                        self.data,
                                        self.omean,
                                        oc
                                        )
            assert result > 0
    
    def test_types_of_covariance_arguments_input(self):
        c = np.random.rand(self.idim, self.idim)
        for ic in [1, c@c.T]:
            result = mdn.norm_input(self.inputs,
                                    self.imean,
                                    ic
                                    )
            assert result > 0
            
    def test_types_of_covariance_arguments_predicted(self):
        c = np.random.rand(self.idim, self.idim)
        for ic in [1, c@c.T]:
            result = mdn.norm_predicted(self.operator,
                                        self.inputs,
                                        self.imean,
                                        ic
                                        )
            assert result > 0