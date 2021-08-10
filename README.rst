.. raw:: html

        <p align="left">
        <a href="https://pypi.org/project/mud/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mud"></a>
        <a href="https://github.com/mathematicalmichael/mud/actions"><img alt="Test Actions Status" src="https://github.com/mathematicalmichael/mud/actions/workflows/main.yml/badge.svg"></a>
        <a href="https://github.com/mathematicalmichael/mud/actions"><img alt="Build Actions Status" src="https://github.com/mathematicalmichael/mud/actions/workflows/build.yml/badge.svg"></a>
        <a href="https://github.com/mathematicalmichael/mud/actions"><img alt="Publish Actions Status" src="https://github.com/mathematicalmichael/mud/actions/workflows/publish.yml/badge.svg"></a>
        <a href="https://mud.readthedocs.io/en/stable/?badge=stable"><img alt="Documentation Status" src="https://readthedocs.org/projects/mud/badge/?version=stable"></a>
        <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
        <a href="https://coveralls.io/github/mathematicalmichael/mud?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/mathematicalmichael/mud/badge.svg?branch=main"></a>
        <a href="https://codecov.io/gh/mathematicalmichael/mud"><img alt="Coverage Status" src="https://codecov.io/gh/mathematicalmichael/mud/branch/main/graph/badge.svg?token=HT880PYHPG"></a>
        <a href="https://pepy.tech/project/mud"><img alt="Total Downloads" src="https://static.pepy.tech/personalized-badge/mud?period=total&units=abbreviation&left_color=gray&right_color=blue&left_text=downloads"></a>
        </p>

.. badge-header


MUD
***

Analytical solutions and some associated utility functions for computing Maximal Updated Density (MUD) parameter estimates for Data-Consistent Inversion.


Description
===========

Maximal Updated Density Points are the values which maximize an updated density, analogous to how a MAP (Maximum A-Posteriori) point maximizes a posterior density from Bayesian inversion.
Updated densities differ from posteriors in that they are the solution to a different problem which seeks to match the push-forward of the updated density to a specified observed distribution.
