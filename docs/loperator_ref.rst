.. _loperator_ref:

``pycommute.loperator``
=======================

.. automodule:: pycommute.loperator

.. pybind11class:: typing.SupportsInt
.. pybind11class:: typing.SupportsFloat
.. pybind11class:: numpy.float64
.. pybind11class:: numpy.complex128
.. pybind11class:: numpy.typing.ArrayLike
.. pybind11class:: pycommute.expression.Indices
.. pybind11class:: pycommute.expression.ExpressionR
.. pybind11class:: pycommute.expression.ExpressionC
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
.. pybind11class:: pycommute.loperator.CompressedStateViewR
.. pybind11class:: pycommute.loperator.CompressedStateViewC
.. pybind11class:: pycommute.loperator.NFermionSectorViewR
.. pybind11class:: pycommute.loperator.NFermionSectorViewC
.. pybind11class:: pycommute.loperator.NFermionMultiSectorViewR
.. pybind11class:: pycommute.loperator.NFermionMultiSectorViewC

Elementary spaces
-----------------

The following classes wrap *libcommute*'s
:ref:`base class for elementary spaces <elementary_spaces>` as well as its
extensions for the special cases of fermions, bosons and spins.

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

:py:class:`HilbertSpace` is a wrapper of *libcommute*'s
:ref:`Hilbert space <hilbert_space>` class.

.. autoclass:: pycommute.loperator.HilbertSpace
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.loperator.foreach

Linear operators
----------------

:py:class:`LOperatorR` and :py:class:`LOperatorC` represent *libcommute*'s
:ref:`linear operator <loperator>` objects specialized for the real and
complex scalar types respectively.

.. autoclass:: pycommute.loperator.LOperatorR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.LOperatorC
  :members:
  :special-members:

Hilbert space partition
-----------------------

:py:class:`SpacePartition` is a wrapper around *libcommute*'s utility class
:ref:`space_partition <space_partition>`.

.. autoclass:: pycommute.loperator.SpacePartition
  :members:
  :special-members:

.. automethod:: pycommute.loperator.make_space_partition

Mapped views of state vectors
-----------------------------

:py:class:`MappedBasisViewR` and :py:class:`MappedBasisViewC` are wrappers
around :ref:`mapped basis views <mapped_basis_view>` of real and complex state
vectors respectively.

.. autoclass:: pycommute.loperator.MappedBasisViewR
  :members:
  :special-members:
  :exclude-members: __init__

.. autoclass:: pycommute.loperator.MappedBasisViewC
  :members:
  :special-members:
  :exclude-members: __init__

.. autoclass:: pycommute.loperator.BasisMapper
  :members:
  :special-members:

Views of compressed state vectors
---------------------------------

:py:class:`CompressedStateViewR` and :py:class:`CompressedStateViewC` are
wrappers around
:ref:`views of real and complex compressed state vectors
<compressed_state_view>`
respectively.

.. autoclass:: pycommute.loperator.CompressedStateViewR
  :members:
  :special-members:

.. autoclass:: pycommute.loperator.CompressedStateViewC
  :members:
  :special-members:

N-fermion sector views of state vectors
---------------------------------------

The following types and functions wrap *libcommute*'s functionality related
to the :math:`N`-fermion :ref:`(multi)sector views <n_fermion_sector_view>`
of state vectors.

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
