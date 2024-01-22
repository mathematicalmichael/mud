.. image:: https://img.shields.io/pypi/v/mud
    :alt: PyPI
    :target: https://pypi.org/project/mud/

.. image:: https://github.com/mathematicalmichael/mud/actions/workflows/main.yml/badge.svg
    :alt: Test Actions Status
    :target: https://github.com/mathematicalmichael/mud/actions

.. image:: https://github.com/mathematicalmichael/mud/actions/workflows/build.yml/badge.svg
    :alt: Build Actions Status
    :target: https://github.com/mathematicalmichael/mud/actions

.. image:: https://github.com/mathematicalmichael/mud/actions/workflows/publish.yml/badge.svg
    :alt: Publish Actions Status
    :target: https://github.com/mathematicalmichael/mud/actions

.. image:: https://readthedocs.org/projects/mud/badge/?version=stable
    :alt: Documentation Status
    :target: https://mud.readthedocs.io/en/stable/?badge=stable

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code Style: black
    :target: https://github.com/psf/black

.. image:: https://coveralls.io/repos/github/mathematicalmichael/mud/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://coveralls.io/github/mathematicalmichael/mud?branch=main

.. image:: https://static.pepy.tech/personalized-badge/mud?period=total&units=abbreviation&left_color=gray&right_color=blue&left_text=downloads
    :alt: Total Downloads
    :target: https://pepy.tech/project/mud


.. badge-header

MUD
***

Analytical solutions and some associated utility functions for computing Maximal Updated Density (MUD) parameter estimates for Data-Consistent Inversion.


Description
===========

Maximal Updated Density Points are the values which maximize an updated density, analogous to how a MAP (Maximum A-Posteriori) point maximizes a posterior density from Bayesian inversion.
Updated densities differ from posteriors in that they are the solution to a different problem which seeks to match the push-forward of the updated density to a specified observed distribution.
