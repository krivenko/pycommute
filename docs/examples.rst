.. _examples:

Usage examples
==============

Holstein model
--------------

This example shows how to construct Hamiltonian of the
`Holstein model <https://arxiv.org/abs/cond-mat/9812252>`_ on a
square lattice and to check that the total number of electrons is a conserved
quantity of the model.

The Hamiltonian considered here is a sum of three terms.

* An electronic tight-binding model on a square lattice with only
  nearest-neighbour hopping allowed,

  .. math::

    \hat H_\text{e} = -t \sum_\sigma \sum_{\langle i,j\rangle}
                      c^\dagger_{i,\sigma} c_{j,\sigma}.

* A harmonic oscillator at each lattice site (a localized phonon),

  .. math::

    \hat H_\text{ph} = \omega_0 \sum_i a^\dagger_i a_i.

* A coupling between the electrons and the phonons.

  .. math::

    \hat H_\text{e-ph} = g \sum_\sigma \sum_i n_{i,\sigma}(a^\dagger_i + a_i).


.. literalinclude:: holstein.py
  :language: python
  :lines: 11-
  :linenos:


Spin-1/2 Heisenberg chain
-------------------------

.. TODO:: Heisenberg chain example
