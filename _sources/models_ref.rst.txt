.. _models_ref:

``pycommute.models``
====================

.. automodule:: pycommute.models

Functions in this module return
:py:class:`real <pycommute.expression.ExpressionR>` or
:py:class:`complex <pycommute.expression.ExpressionC>` expression objects
corresponding to some Hamiltonians widely used in the theory of quantum
many-body systems, quantum optics and statistical mechanics of interacting
spins.

By default, these functions use integer indices ``0, 1, ...`` to label
fermionic, bosonic and spin operators corresponding to individual basis states
(lattice sites, orbitals) in the output expressions. They also accept
custom lists of indices, which can come in two forms,

- A list of integer and/or string indices, e.g. ``[0, "a", 2, 5, "xyz"]``;
- A list of multi-indices, e.g. ``[('a', 1), ('b', 2), ('c', 'd')]``.

It is also allowed to mix indices and multi-indices in the same list.

Quantum many-body theory
------------------------

.. autofunction:: pycommute.models.tight_binding
.. autofunction:: pycommute.models.dispersion
.. autofunction:: pycommute.models.pairing
.. autofunction:: pycommute.models.zeeman
.. autofunction:: pycommute.models.hubbard_int
.. autofunction:: pycommute.models.bose_hubbard_int
.. autofunction:: pycommute.models.extended_hubbard_int
.. autofunction:: pycommute.models.t_j_int
.. autofunction:: pycommute.models.kondo_int
.. autofunction:: pycommute.models.holstein_int
.. autofunction:: pycommute.models.quartic_int
.. autofunction:: pycommute.models.kanamori_int
.. autofunction:: pycommute.models.slater_int

Spin lattice models
-------------------

.. autofunction:: pycommute.models.ising
.. autofunction:: pycommute.models.heisenberg
.. autofunction:: pycommute.models.anisotropic_heisenberg
.. autofunction:: pycommute.models.biquadratic_spin_int
.. autofunction:: pycommute.models.dzyaloshinskii_moriya

Quantum optics and quantum dissipation
--------------------------------------

.. autofunction:: pycommute.models.spin_boson
.. autofunction:: pycommute.models.rabi
.. autofunction:: pycommute.models.jaynes_cummings
