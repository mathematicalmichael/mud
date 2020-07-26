# -*- coding: utf-8 -*-

import pytest
import mud.funs as mdf
import numpy as np

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


def test_fun():
    assert np.linalg.norm(mdf.makeRi(np.eye(2), np.eye(2))) < 1E-14
    #with pytest.raises(AssertionError):

