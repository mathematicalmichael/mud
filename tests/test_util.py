# -*- coding: utf-8 -*-

import unittest
import mud.util as mdu

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


class TestUtil(unittest.TestCase):
    def test_probability_high_for_tolerance(self):
        # Arrange
        tolerances = [0.01, 0.05, 0.1, 0.25]
        result = []

        # Act
        for tol in tolerances:
            result.append(mdu.std_from_equipment(tolerance=tol, probability=0.9999))

        # Assert
        for i, tol in enumerate(tolerances):
            assert 3 * result[i] < tol

    def test_probability_med_for_tolerance(self):
        # Arrange
        tolerances = [0.01, 0.05, 0.1, 0.25]
        result = []

        # Act
        for tol in tolerances:
            result.append(mdu.std_from_equipment(tolerance=tol, probability=0.999))

        # Assert
        for i, tol in enumerate(tolerances):
            assert 3 * result[i] < tol
