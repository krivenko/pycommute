#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# Sector-wise diagonalization of the Slater interaction Hamiltonian.
#

import numpy as np

# Utility function used to construct the Slater interaction Hamiltonian.
from pycommute.models import slater_int

# Hilbert space.
from pycommute.loperator import HilbertSpace

# Real-valued linear operator object.
from pycommute.loperator import LOperatorR
# Automatic space partition and basis state mapper
from pycommute.loperator import make_space_partition, foreach, BasisMapper

# Values of Slater radial integrals F_0, F_2, F_4
F = np.array([4.0, 1.0, 0.2])

# Slater Hamiltonian.
H = slater_int(F)

# Analyze structure of H and construct a suitable Hilbert space.
hs = HilbertSpace(H)

# Construct a linear operator corresponding to H and acting in the Hilbert
# space hs.
H_op = LOperatorR(H, hs)

# The first value returned by make_space_partition() is a SpacePartition object.
# It represents a partition of the full Hilbert space into sectors,
# i.e. subspaces invariant under action of H_op.
#
# The second returned value is a dictionary {(i, j): value} of all non-vanishing
# matrix elements of H_op. By construction, all matrix elements of H_op between
# different sectors vanish.
sp, matrix_elements = make_space_partition(H_op, hs)

# Print out information about the revealed sectors.
print("Dimension of full Hilbert space:", sp.dim)
print("Total number of sectors:", sp.n_subspaces)
for i in range(sp.dim):
    print("Many-body basis state %d belongs to sector %d" % (i, sp[i]))

# Use foreach() to compile lists of basis states spanning each sector.
sectors = [[] for _ in range(sp.n_subspaces)]
foreach(sp,
        lambda basis_state, sector: sectors[sector].append(basis_state))

# Diagonalize H_op within each sector
for n, sector in enumerate(sectors):
    sector_dim = len(sector)
    print(f"Diagonalizing sector {n} of H_op, sector size is {sector_dim}")

    # A BasisMapper object translates indices of basis states from the full
    # Hilbert space to a given sector.
    basis_mapper = BasisMapper(sector)

    # Prepare a matrix representation of H_op within current sector.
    H_mat = np.zeros((sector_dim, sector_dim))
    for i in range(sector_dim):
        # A column vector psi = {0, 0, ..., 1, ..., 0}
        psi = np.zeros(sector_dim)
        psi[i] = 1.0
        # This vector will receive the result of H_op * psi
        phi = np.zeros(sector_dim)

        # Since both psi and phi are 1D NumPy arrays with size of the chosen
        # sector, H_op cannot act on them directly as it expects vectors of
        # size hs.dim.
        # Instead, we are going to use our basis mapper object to construct
        # special views of psi and phi.
        psi_view = basis_mapper(psi)
        phi_view = basis_mapper(phi)

        # Now, H_op can act on the mapped views.
        H_op(psi_view, phi_view)

        # Store H_op * psi in the i-th column of H_mat.
        H_mat[:, i] = phi

    # Use NumPy to compute eigenvalues of H_mat.
    E = np.linalg.eigvals(H_mat)
    print("Energies:", np.sort(E))