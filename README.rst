MUD
===

.. include:: ./badges.rst

Analytical solutions and some associated utility functions for computing maximal updated density points for Data-Consistent Inversion.

Description
===========

Maximal Updated Density Points are the values which maximize an updated density, analogous to how a MAP (Maximum A-Posteriori) point maximizes a posterior density from Bayesian inversion.
Updated densities differ from posteriors in that they are the solution to a different problem which seeks to match the push-forward of the updated density to a specified observed distribution.
