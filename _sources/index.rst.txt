pycommute
=========

Python bindings for the `libcommute <https://github.com/krivenko/libcommute>`_
quantum operator algebra DSL and exact diagonalization toolkit for C++.

These bindings inherit a subset of functionality supported by the C++ template
library: They expose polynomial expression and linear operator types with
real and complex coefficients built out of fermionic, bosonic and spin
operators [#]_. To learn more about basic concepts and features of this library,
please, refer to
`libcommute's documentation <https://krivenko.github.io/libcommute/>`_.
As a bonus, *pycommute* features a handful of Python functions to easily
construct Hamiltonians of some models widely used in the theory of quantum
many-body systems, quantum optics and the theory of spin lattices.

For now, there are a few :ref:`usage examples <examples>` and an API reference
for the :ref:`pycommute.expression <expression_ref>`
:ref:`pycommute.loperator <loperator_ref>`, and
:ref:`pycommute.models <models_ref>` modules.

Contents
========

.. toctree::
    :name: mastertoc
    :maxdepth: 3

    installation
    examples
    reference
    changelog
    genindex
    search

.. [#] Support for user-defined operator algebras in Python code may be added in
       a later release.
