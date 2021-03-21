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

Instead of writing the sums over lattice sites explicitly, we call library
functions :func:`tight_binding() <pycommute.models.tight_binding>`,
:func:`dispersion() <pycommute.models.dispersion>` and
:func:`holstein_int() <pycommute.models.holstein_int>`.
We also make use of the `NetworkX <https://networkx.org/>`_ package
to easily generate the adjacency matrix of the periodic square lattice.

.. literalinclude:: holstein.py
  :language: python
  :lines: 11-
  :linenos:


Spin-1/2 Heisenberg chain
-------------------------

The spin-1/2 Heisenberg chain is a textbook example of an integrable quantum
system. Its Hamiltonian

.. math::

  \hat H = g \sum_i \mathbf{S}_i \cdot \mathbf{S}_{i+1}

conserves three projections of the total spin

.. math::

  \mathbf{S} = \sum_i \mathbf{S}_i

as well as a series of higher order charges :math:`Q_n`. Existence of these
charges can be derived from the transfer matrix theory. Explicit expressions
for :math:`Q_n` were obtained in [GM94]_. The following script constructs
Hamiltonian of the Heisenberg chain with periodic boundary conditions
(see :func:`heisenberg() <pycommute.models.heisenberg>`)
and checks that :math:`[\hat H, \mathbf{S}] = 0`, :math:`[\hat H, Q_n] = 0` and
:math:`[Q_n, Q_m] = 0` for :math:`m,n = 3,4,5`.

.. literalinclude:: heisenberg_chain.py
  :language: python
  :lines: 11-
  :linenos:

.. [GM94] "Quantum Integrals of Motion for the Heisenberg Spin Chain",
  M. P. Grabowski and P. Mathieu,
  Mod. Phys. Lett. A, Vol. 09, No. 24, pp. 2197-2206 (1994),
  https://doi.org/10.1142/S0217732394002057
