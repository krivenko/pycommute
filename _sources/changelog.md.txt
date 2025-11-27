# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - Unreleased

- Enable support for Python 3.11, 3.12, 3.13 and 3.14.
- Remove support for Python 3.6 and 3.7.
- It is now possible to define new algebras by extending the base class
  ``expression.Generator``. This functionality has been documented in the API
  reference, and a new example has been added.
- New auxiliary type ``expression.LinearFunctionGen`` that is used to define
  commutation relations and simplification rules for user-defined algebras.
- The property ``expression.Generator.algebra_id`` has been turned into a
  method.
- Added methods ``__copy__()`` and ``__deepcopy__()`` to a few mutable objects:
  ``expression.Monomial``, ``expression.Expression[R|C]``,
  ``loperator.HilbertSpace`` and ``loperator.SpacePartition``.
- New read-only attribute ``loperator.ElementarySpace.dim`` also inherited by
  ``loperator.ESpace(Fermion|Boson|Spin)``.
- Dimension of ``loperator.ESpaceBoson`` had to be a power of 2. Now this
  restriction is lifted. Arguments of its constructor and of
  ``loperator.make_space_boson()`` have been adjusted accordingly: They now
  expect the dimension argument ``dim`` instead of its binary logarithm
  ``n_bits``.
- Constructors of ``loperator.HilbertSpace`` are changed to accept the argument
  ``dim_boson`` instead of its binary logarithm ``bits_per_boson``.
- New read-only attribute ``loperator.HilbertSpace.is_sparse`` that is ``True``
  if some of the constituent elementary spaces have non-power-of-two dimensions.
- New read-only attribute ``loperator.HilbertSpace.vec_size`` that is equal to
  the minimal size of a state vector container compatible with this Hilbert
  space.
- Semantics of the existing attribute ``loperator.HilbertSpace.dim`` has been
  changed. Now it is equal to the exact dimension of the Hilbert space, which is
  smaller than ``loperator.HilbertSpace.vec_size`` if the Hilbert space is
  sparse.
- New method ``loperator.HilbertSpace.es_dim(es)`` that returns dimension of a
  constituent elementary space.
- New method ``loperator.HilbertSpace.foreach_elementary_space(f)`` that applies
  a given function to each constituent elementary space.
- New classes ``CompressedStateView(R|C)``. These objects are
  ``LOperator(R|C)``-compatible views that perform basis state index translation
  from a (possibly) sparse Hilbert space of dimension ``dim`` to the continuous
  range ``[0; dim-1]``.
- Methods ``loperator.SpacePartition.merge_subspaces()`` and
  ``loperator.SpacePartition.find_connections()`` no longer accept the ``hs``
  argument and instead use the ``loperator.HilbertSpace`` object provided upon
  construction.

## [0.7.1] - 2021-12-17

- Enable support for Python 3.10.
- New methods ``loperator.SpacePartition.subspace_basis()`` and
  ``loperator.SpacePartition.subspace_bases()``.
- Minor documentation updates.

## [0.7.0] - 2021-10-09

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

- New method ``expression.Indices.__getitem__()``.
- New method ``loperator.HilbertSpace.index()``.
- CI: Tagged versions are now built into Docker images and pushed to
  [Docker Hub](https://hub.docker.com/repository/docker/ikrivenko/pycommute).
