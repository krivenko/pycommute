#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2024 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Holstein model on a square lattice with periodic boundary conditions.
#

from itertools import product
import numpy as np

from pycommute.expression import ExpressionR, conj, FERMION, BOSON
from pycommute.expression import n
from pycommute.models import tight_binding, dispersion, holstein_int

from networkx.generators.lattice import grid_2d_graph
from networkx.linalg.graphmatrix import adjacency_matrix

#
# Let us define Hamiltonian of an electronic tight-binding model
# on a square lattice.
#

# Number of lattice sites in each direction
# (the total number of sites is N*N).
N = 10

# Electron hopping constant - energy parameter of the TB model.
t = 2.0

# Use NetworkX to construct the periodic square lattice (graph)
lat = grid_2d_graph(N, N, periodic=True)

# Create lists of indices for electronic spin-up and spin-down operators.
# lat.nodes() returns a list of N^2 pairs of site indices (x, y).
indices_up = [(x, y, "up") for x, y in lat.nodes()]
indices_dn = [(x, y, "down") for x, y in lat.nodes()]

# A sum of tight-binding Hamiltonians for both spins.
# The hopping matrix passed to tight_binding() is proportional to the
# adjacency matrix of the lattice.
hopping_matrix = -t * adjacency_matrix(lat).todense()
H_e = tight_binding(hopping_matrix, indices=indices_up) \
    + tight_binding(hopping_matrix, indices=indices_dn)

#
# Hamiltonian of phonons localized at lattice sites.
#

# Frequency of the localized phonon.
w0 = 0.5

# A lists of indices for bosonic operators, simply the (x, y) pairs
indices_phonon = lat.nodes()

# All N^2 phonons have the same frequency
phonon_freqs = w0 * np.ones(N ** 2)
H_ph = dispersion(phonon_freqs, indices=indices_phonon, statistics=BOSON)

#
# Hamiltonian of electron-phonon coupling.
#

# Electron-phonon coupling constant.
g = 0.1

H_e_ph = holstein_int(g * np.ones(N ** 2),
                      indices_up=indices_up,
                      indices_dn=indices_dn,
                      indices_boson=indices_phonon)

# Complete Holstein Hamiltonian.
H_H = H_e + H_ph + H_e_ph

# Print H_H. There will be quite a lot of terms for the 100-site lattice!
print("H_H =", H_H)

# Check hermiticity of H_H.
print(r"H_H - H_H^\dagger =", H_H - conj(H_H))

# Check that H_H commutes with the total number of electrons N_e.
N_e = ExpressionR()
for spin in ("up", "down"):
    for x, y in product(range(N), range(N)):
        N_e += n(x, y, spin)

print("[H_H, N_e] =", H_H * N_e - N_e * H_H)

#
# Iteration interface
#

# Iterate over monomial-coefficient pairs in polynomial expression H_H
for monomial, coeff in H_H:
    print("Coefficient:", coeff)
    # Iterate over algebra generators (creation/annihilation operators)
    # in the monomial
    for generator in monomial:
        # Detect what algebra this generator belongs to
        if generator.algebra_id == FERMION:
            print("\tFermionic", end=' ')
        elif generator.algebra_id == BOSON:
            print("\tBosonic", end=' ')
        # Creation or annihilation operator?
        if generator.dagger:
            print("creation operator", end=' ')
        else:
            print("annihilation operator", end=' ')
        # Extract indices carried by the generator
        print("with indices", list(generator.indices))
        # N.B. generator.indices is an instance of a special tuple-like type
        # `pycommute.expression.Indices`. Types of its elements are restricted
        # to integer and string, and its comparison operators behave differently
        # from those of the Python tuple. Otherwise, it supports `len()`,
        # indexed element access and iteration protocol.
