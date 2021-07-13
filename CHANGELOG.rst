=========
Changelog
=========

Versions 0.0.x
==============
- Setting up initial repository, configuring CI/CD
- Migration of code from CU-Denver-UQ/mud-paper repo
- Revisions of architecture, moving modules around
- Rapid iteration, not sticking to semantic versioning
- Possible breaking versions between patches (some functions moved to `mud-examples`)
- Defines basic functionality, classes, helpful functions


Version 0.0.25
==============
- Updated packaging to comply with PEP 517/518 using `pyscaffold `v4.0.2`
- Removes pyerf in favor of erfinv from `scipy.special` (available since `v0.2`)
- Renames `testing` to `dev` for optional dependency installation
- Adds `black` as a `dev` dependency
- Run `black` + `flake8` on whole project
- clean up `setup.cfg` file
- adds file for readthedocs


Version 0.0.26
==============
- Read the Docs set up, documentation infrastructure.


Version 0.0.27
==============
- Adding docstrings
- Removing `plot` module. `mud-examples` already has it.
- Fixing CHANGELOG typos with version numbers.
- Update README
- Update project description + metadata in `setup.cfg`
- `sphinx_copybutton` extension added

Version 0.1
===========

- Basic functionality and repo complete with information
- Beginning of adherence to semantic versioning rules
- i.e., breaking changes in major revision, contract changes in minor, bugfixes/features in patch.
