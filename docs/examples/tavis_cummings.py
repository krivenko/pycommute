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
# Diagonalization of a two-qubit Tavis-Cummings model.
#

import numpy as np

# Utility function used to construct the generalized Jaynes-Cummings model.
from pycommute.models import jaynes_cummings

# Hilbert spaces.
from pycommute.loperator import HilbertSpace, make_space_boson, make_space_spin

# Real-valued linear operator object.
from pycommute.loperator import LOperatorR

# Build the matrix form of a linear operator.
from pycommute.loperator import make_matrix

# Transition frequencies of qubits.
eps = np.array([1.0, 1.0])

# Oscillator frequency.
# (In a more general case of M oscillators, omega would be a length-M array).
omega = np.array([0.8])

# Qubit-oscillator coupling constants as a 2x1 array.
g = np.array([[0.5], [0.6]])

# Create the Tavis-Cummings Hamiltonian.
H = jaynes_cummings(eps, omega, g)

# Construct state space of our problem as a direct product of two
# two-dimensional Hilbert spaces (qubits) and one truncated bosonic Hilbert
# space.
# make_space_boson(4) returns the truncated bosonic space with allowed
# occupation numbers N = 0, 1, ..., (2^4-1).
hs = HilbertSpace([
    make_space_spin(1 / 2, 0),  # Qubit 1: spin-1/2, index 0
    make_space_spin(1 / 2, 1),  # Qubit 2: spin-1/2, index 1
    make_space_boson(4, 0)      # Oscillator, index 0
])

# Construct a linear operator corresponding to 'H' and acting in the Hilbert
# space 'hs'.
H_op = LOperatorR(H, hs)

#
# Prepare a matrix representation of 'H_op'
#

# Method I (manual).
H_mat1 = np.zeros((hs.dim, hs.dim))
for i in range(hs.dim):
    # A column vector psi = {0, 0, ..., 1, ..., 0}
    psi = np.zeros(hs.dim)
    psi[i] = 1.0
    # Act with H_op on psi and store the result the i-th column of H_mat
    H_mat1[:, i] = H_op * psi

# Method II (automatic and faster).
H_mat2 = make_matrix(H_op, hs)
print("Max difference between H_mat and H_mat2:",
      np.max(np.abs(H_mat1 - H_mat2)))

# Use NumPy to compute eigenvalues of H_mat
E = np.linalg.eigvals(H_mat1)

print("Energies:", np.sort(E))
