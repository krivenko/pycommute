pycommute
=========

Python bindings for the `libcommute <https://github.com/krivenko/libcommute>`_
quantum operator algebra DSL and exact diagonalization toolkit for C++.

These bindings expose the polynomial expression and linear operator types with
real and complex coefficients built out of fermionic, bosonic and spin
operators [#]_. To learn more about basic concepts and features of this library,
please, refer to
`libcommute's documentation <https://krivenko.github.io/libcommute/>`_.

For now, there are a few :ref:`usage examples <examples>` and an API reference
for the :ref:`pycommute.expression <expression_ref>` and
:ref:`pycommute.loperator <loperator_ref>` modules.

Contents
========

.. toctree::
    :name: mastertoc
    :maxdepth: 3

    installation
    examples
    reference
    genindex
    search

.. [#] Support for user-defined operator algebras in Python code may be added in
       a later release.
