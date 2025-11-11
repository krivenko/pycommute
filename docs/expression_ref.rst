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

:py:class:`Indices` is a wrapper around *libcommute*'s
:ref:`dynamically-typed indices <dyn_indices>`.

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

The following classes wrap *libcommute*'s
:ref:`base class for algebra generators <generator>` as well as its extensions
for :ref:`fermionic <generator_fermion>`, :ref:`bosonic <generator_boson>` and
:ref:`spin <generator_spin>` algebras.

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

:py:class:`Monomial` is a wrapper around *libcommute*'s
:ref:`monomial <monomial>` object.

.. autoclass:: pycommute.expression.Monomial
  :members:
  :special-members:

Expression objects
------------------

:py:class:`ExpressionR` and :py:class:`ExpressionC` represent *libcommute*'s
:ref:`polynomial expression <expression>` objects specialized for the real and
complex scalar types respectively.

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

These are wrapped *libcommute*'s :ref:`factory functions <factories_dyn>` for
polynomial expressions with :ref:`dynamically typed indices <dyn_indices>`.

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

:py:class:`HC` is the Python analog of *libcommute*'s
:ref:`Hermitian conjugate placeholder <hc>`.

.. autoclass:: pycommute.expression.HC
.. autoattribute:: pycommute.expression.hc
