# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- New function ``loperator.make_matrix()`` with three overloads. These wrapped
  C++ functions construct and return a matrix representation (2D NumPy array)
  of a linear operator acting (1) in a full Hilbert space, (2) in its subspace
  spanned by a given list of basis vectors or (3) between two subspaces of the
  Hilbert space.
- New method ``loperator.HilbertSpace.has_algebra()``.
- New method ``loperator.SpacePartition.find_connections()``.

## [0.6.1] - 2021-03-30

### Added

- New method ``expression.Indices.__getitem__()``.
- New method ``loperator.HilbertSpace.index()``.
- CI: Tagged versions are now built into Docker images and pushed to
  [Docker Hub](https://hub.docker.com/repository/docker/ikrivenko/pycommute).
