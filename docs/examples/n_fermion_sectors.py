#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2024 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Fermi-Hubbard model on a large square lattice and diagonalization of
# its moderately sized sectors.
#

import numpy as np

# Utility functions used to construct the Fermi-Hubbard Hamiltonian.
from pycommute.models import tight_binding, dispersion, hubbard_int

# Hilbert space.
from pycommute.loperator import HilbertSpace

# Real-valued linear operator object.
from pycommute.loperator import LOperatorR
# Real-valued N-fermion (multi)sector views and related functions
from pycommute.loperator import (
    NFermionSectorViewR,
    NFermionMultiSectorViewR,
    n_fermion_sector_size,
    n_fermion_multisector_size
)

from networkx.generators.lattice import grid_2d_graph
from networkx.linalg.graphmatrix import adjacency_matrix

#
# Let us define Hamiltonian of a tight-binding model on a square lattice.
#

# Number of lattice sites in each direction
# (the total number of sites is Nx * Ny).
Nx = 4
Ny = 4

# Hopping constant
t = 0.5
# Chemical potential
mu = 1.0
# Coulomb repulsion
U = 2.0

# Use NetworkX to construct the periodic square lattice (graph)
lat = grid_2d_graph(Nx, Ny, periodic=True)

# Create lists of indices for spin-up and spin-down operators.
# lat.nodes() returns a list of Nx * Ny pairs of site indices (x, y).
indices_up = [(x, y, "up") for x, y in lat.nodes()]
indices_dn = [(x, y, "down") for x, y in lat.nodes()]

# A sum of tight-binding Hamiltonians for both spins.
# The hopping matrix passed to tight_binding() is proportional to the
# adjacency matrix of the lattice.
hopping_matrix = -t * adjacency_matrix(lat).todense()
H = tight_binding(hopping_matrix, indices=indices_up) \
    + tight_binding(hopping_matrix, indices=indices_dn)

# Add the chemical potential terms
H += dispersion(-mu * np.ones(len(indices_up)), indices=indices_up)
H += dispersion(-mu * np.ones(len(indices_dn)), indices=indices_dn)

# Add the Hubbard interaction term
H += hubbard_int(U * np.ones(len(indices_up)),
                 indices_up=indices_up,
                 indices_dn=indices_dn)

# Analyze structure of H and construct a suitable Hilbert space.
hs = HilbertSpace(H)
print("Full Hilbert space dimension:", hs.dim)

# Construct a linear operator corresponding to H and acting in the Hilbert
# space 'hs'.
H_op = LOperatorR(H, hs)

#
# Diagonalize the N = 2 sector of the model using 'NFermionSectorViewR'
#

N = 2

sector_size = n_fermion_sector_size(hs, N)
print("Size of the N = 2 sector:", sector_size)

# Prepare a matrix representation of H_op within the N = 2 sector.
H_mat = np.zeros((sector_size, sector_size))
for i in range(sector_size):
    # A column vector psi = {0, 0, ..., 1, ..., 0}
    psi = np.zeros(sector_size)
    psi[i] = 1.0
    # This vector will receive the result of H_op * psi
    phi = np.zeros(sector_size)

    # Since both psi and phi are 1D NumPy arrays with size of the chosen
    # sector, H_op cannot act on them directly as it expects vectors of
    # size hs.dim.
    # Instead, we are going to use real-valued 2-fermion sector views of
    # psi and phi.
    psi_view = NFermionSectorViewR(psi, hs, N)
    phi_view = NFermionSectorViewR(phi, hs, N)

    # Now, H_op can act on the mapped views.
    H_op(psi_view, phi_view)

    # Store H_op * psi in the i-th column of H_mat.
    H_mat[:, i] = phi

# Use NumPy to compute eigenvalues of H_mat.
E = np.linalg.eigvals(H_mat).real
print("10 lowest eigenvalues of the N = 2 sector:", np.sort(E)[:10])

#
# Diagonalize the N_up = 1, N_down = 1 multisector of the model using
# 'NFermionMultiSectorViewR'
#

N_up = 1
N_dn = 1

# Define sectors contributing to the multisector
sector_up = (indices_up, N_up)
sector_dn = (indices_dn, N_dn)

sectors = [sector_up, sector_dn]

multisector_size = n_fermion_multisector_size(hs, sectors)
print("Size of the N_up = 1, N_down = 1 multisector:", multisector_size)

# Prepare a matrix representation of H_op within the multisector.
H_mat = np.zeros((multisector_size, multisector_size))
for i in range(multisector_size):
    # A column vector psi = {0, 0, ..., 1, ..., 0}
    psi = np.zeros(multisector_size)
    psi[i] = 1.0
    # This vector will receive the result of H_op * psi
    phi = np.zeros(multisector_size)

    # Since both psi and phi are 1D NumPy arrays with size of the chosen
    # sector, H_op cannot act on them directly as it expects vectors of
    # size hs.dim.
    # Instead, we are going to use real-valued multisector views of psi and phi.
    psi_view = NFermionMultiSectorViewR(psi, hs, sectors)
    phi_view = NFermionMultiSectorViewR(phi, hs, sectors)

    # Now, H_op can act on the mapped views.
    H_op(psi_view, phi_view)

    # Store H_op * psi in the i-th column of H_mat.
    H_mat[:, i] = phi

# Use NumPy to compute eigenvalues of H_mat.
E = np.linalg.eigvals(H_mat).real
print("10 lowest eigenvalues of the multisector:", np.sort(E)[:10])
