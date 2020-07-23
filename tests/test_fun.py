# -*- coding: utf-8 -*-

import pytest
import mud.fun as mdf

__author__ = "Mathematical Michael"
__copyright__ = "Mathematical Michael"
__license__ = "mit"


def test_fun():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
