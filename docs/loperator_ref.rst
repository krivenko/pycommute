.. _loperator_ref:

``pycommute.loperator``
=======================

.. automodule:: pycommute.loperator

.. pybind11class:: pycommute.expression.Indices
.. pybind11class:: pycommute.loperator.ElementarySpace
.. pybind11class:: pycommute.loperator.ESpaceFermion
.. pybind11class:: pycommute.loperator.ESpaceBoson
.. pybind11class:: pycommute.loperator.ESpaceSpin
.. pybind11class:: pycommute.loperator.HilbertSpace
.. pybind11class:: pycommute.loperator.LOperatorR
.. pybind11class:: pycommute.loperator.LOperatorC
.. pybind11class:: pycommute.loperator.SpacePartition
.. pybind11class:: pycommute.loperator.MappedBasisViewR
.. pybind11class:: pycommute.loperator.MappedBasisViewC
.. pybind11class:: pycommute.loperator.BasisMapper
.. pybind11class:: pycommute.loperator.NFermionSectorViewR
.. pybind11class:: pycommute.loperator.NFermionSectorViewC
.. pybind11class:: pycommute.loperator.NFermionMultiSectorViewR
.. pybind11class:: pycommute.loperator.NFermionMultiSectorViewC

Elementary spaces
-----------------

.. autoclass:: ElementarySpace
  :members:
  :special-members:
  :exclude-members: __init__

.. autoclass:: pycommute.loperator.ESpaceFermion
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.loperator.make_space_fermion

.. autoclass:: pycommute.loperator.ESpaceBoson
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.loperator.make_space_boson

.. autoclass:: pycommute.loperator.ESpaceSpin
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.loperator.make_space_spin

Full Hilbert spaces
-------------------

.. autoclass:: pycommute.loperator.HilbertSpace
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.loperator.foreach

Linear operators
----------------

.. autoclass:: pycommute.loperator.LOperatorR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.LOperatorC
  :members:
  :special-members:

Hilbert space partition
-----------------------

.. autoclass:: pycommute.loperator.SpacePartition
  :members:
  :special-members:

.. automethod:: pycommute.loperator.make_space_partition

Mapped views of state vectors
-----------------------------

.. autoclass:: pycommute.loperator.MappedBasisViewR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.MappedBasisViewC
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.BasisMapper
  :members:
  :special-members:

N-fermion sector views of state vectors
---------------------------------------

.. automethod:: pycommute.loperator.n_fermion_sector_size

.. automethod:: pycommute.loperator.n_fermion_sector_basis_states

.. autoclass:: pycommute.loperator.NFermionSectorViewR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.NFermionSectorViewC
  :members:
  :special-members:

.. automethod:: pycommute.loperator.n_fermion_multisector_size

.. automethod:: pycommute.loperator.n_fermion_multisector_basis_states

.. autoclass:: pycommute.loperator.NFermionMultiSectorViewR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.NFermionMultiSectorViewC
  :members:
  :special-members:

Matrix representation of linear operators
-----------------------------------------

.. automethod:: pycommute.loperator.make_matrix
