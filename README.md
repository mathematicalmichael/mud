[![PyPI version](https://badge.fury.io/py/mud.svg)](https://badge.fury.io/py/mud)
![unit tests](https://github.com/mathematicalmichael/mud/actions/workflows/main.yml/badge.svg)
![publish](https://github.com/mathematicalmichael/mud/actions/workflows/publish-pypi.yml/badge.svg)
[![codecov](https://codecov.io/gh/mathematicalmichael/mud/branch/master/graph/badge.svg?token=HT880PYHPG)](https://codecov.io/gh/mathematicalmichael/mud)

# MUD

Analytical solutions and some associated utility functions for computing maximal updated density points for Data-Consistent Inversion.

## Description

Maximal Updated Density Points are the values which maximize an updated density, analogous to how a MAP (Maximum A-Posteriori) point maximizes a posterior density from Bayesian inversion.
Updated densities differ from posteriors in that they are the solution to a different problem which seeks to match the push-forward of the updated density to a specified observed distribution.

More about the differences here...

What does this package include?


## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
