.. _expression_ref:

``pycommute.expression``
========================

.. automodule:: pycommute.expression

.. pybind11class:: pycommute.expression.Indices
.. pybind11class:: pycommute.expression.SpinComponent
.. pybind11class:: pycommute.expression.GeneratorFermion
.. pybind11class:: pycommute.expression.GeneratorBoson
.. pybind11class:: pycommute.expression.GeneratorSpin
.. pybind11class:: pycommute.expression.Generator
.. pybind11class:: pycommute.expression.Monomial
.. pybind11class:: pycommute.expression.ExpressionR
.. pybind11class:: pycommute.expression.ExpressionC

Index sequence
--------------

.. note::

  :py:class:`Indices` objects are not the same thing as Python tuples because
  they follow a different ordering rule. Unlike the Python tuples, two
  index sequences ``I1`` and ``I2`` always compare as ``I1 < I2`` if
  ``len(I1) < len(I2)``.

.. autoclass:: Indices
  :members:
  :special-members:

Algebra ID constants
--------------------

Fermionic algebra ID

.. autoattribute:: pycommute.expression.FERMION

Bosonic algebra ID

.. autoattribute:: pycommute.expression.BOSON

Spin algebra ID

.. autoattribute:: pycommute.expression.SPIN

Algebra generator objects
-------------------------

.. autoclass:: Generator
  :members:
  :special-members:
  :exclude-members: __init__

.. autoclass:: pycommute.expression.GeneratorFermion
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.expression.make_fermion

.. autoclass:: pycommute.expression.GeneratorBoson
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.expression.make_boson

.. autoclass:: pycommute.expression.SpinComponent

.. autoclass:: GeneratorSpin
  :show-inheritance:
  :members:
  :special-members:

.. automethod:: pycommute.expression.make_spin

Monomial object
---------------

.. autoclass:: pycommute.expression.Monomial
  :members:
  :special-members:

Expression objects
------------------

.. autoclass:: pycommute.expression.ExpressionR
  :members:
  :special-members:

.. autoclass:: pycommute.expression.ExpressionC
  :members:
  :special-members:

.. automethod:: pycommute.expression.conj

.. automethod:: pycommute.expression.transform

Expression factory functions
----------------------------

.. automethod:: pycommute.expression.c_dag
.. automethod:: pycommute.expression.c
.. automethod:: pycommute.expression.n

.. automethod:: pycommute.expression.a_dag
.. automethod:: pycommute.expression.a

.. automethod:: pycommute.expression.S_p
.. automethod:: pycommute.expression.S_m
.. automethod:: pycommute.expression.S_x
.. automethod:: pycommute.expression.S_y
.. automethod:: pycommute.expression.S_z

.. automethod:: pycommute.expression.make_complex

Hermitian conjugate placeholder
-------------------------------

.. autoclass:: pycommute.expression.HC
.. autoattribute:: pycommute.expression.hc
