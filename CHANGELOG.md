# Changelog

All notable changes to this project will be documented in this file.

## [0.7.2] - Unreleased

- Added methods ``__copy__()`` and ``__deepcopy__()`` to a few mutable objects:
  ``expression.Monomial``, ``expression.Expression[R|C]``,
  ``loperator.HilbertSpace`` and ``loperator.SpacePartition``.

## [0.7.1] - 2021-12-17

- Enable support for Python 3.10.
- New methods ``loperator.SpacePartition.subspace_basis()`` and
  ``loperator.SpacePartition.subspace_bases()``.
- Minor documentation updates.

## [0.7.0] - 2021-10-09

### Added

- New function ``loperator.make_matrix()`` with three overloads. These wrapped
  C++ functions construct and return a matrix representation (2D NumPy array)
  of a linear operator acting (1) in a full Hilbert space, (2) in its subspace
  spanned by a given list of basis vectors or (3) between two subspaces of the
  Hilbert space.
- New classes ``loperator.NFermionSectorView(R|C)``. These objects are
  ``LOperator(R|C)``-compatible views that represent state vectors defined in an
  N-fermion subspace of a full Hilbert space. In addition to the view classes,
  there are two new utility functions ``loperator.n_fermion_sector_size()`` and
  ``loperator.n_fermion_sector_basis_states()``.
- New classes ``loperator.NFermionMultiSectorView(R|C)``. These objects are
  ``LOperator(R|C)``-compatible views that represent state vectors defined in an
  N-fermion multisector. A multisector is a set of all basis states, which have
  ``N_1`` particles within a subset of fermionic modes ``{S_1}``, ``N_2``
  particles within another subset ``{S_2}`` and so on. There can be any number
  of individual pairs ``({S_i}, N_i)`` (sectors contributing to the multisector)
  as long as all subsets ``{S_i}`` are disjoint. In addition to the view
  classes, there are two new utility functions
  ``loperator.n_fermion_multisector_size()`` and
  ``loperator.n_fermion_multisector_basis_states()``.
- New method ``loperator.HilbertSpace.has_algebra()``.
- New method ``loperator.SpacePartition.find_connections()``.

## [0.6.1] - 2021-03-30

### Added

- New method ``expression.Indices.__getitem__()``.
- New method ``loperator.HilbertSpace.index()``.
- CI: Tagged versions are now built into Docker images and pushed to
  [Docker Hub](https://hub.docker.com/repository/docker/ikrivenko/pycommute).
