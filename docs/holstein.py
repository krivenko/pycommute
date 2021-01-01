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
# Holstein model on a square lattice.
#

from itertools import product
from pycommute.expression import ExpressionR, conj
from pycommute.expression import c_dag, c, a_dag, a, n

#
# Let us define Hamiltonian of an electronic tight-binding model
# on a square lattice.
#

# Number of lattice sites in each direction
# (the total number of sites is N*N).
N = 10

# Electron hopping constant - energy parameter of the TB model.
t = 2.0

# TB Hamiltonian as an expression with real coefficients.
H_e = ExpressionR()

# Iterate over spin projections.
for spin in ("up", "down"):
    # Iterate over all lattice sites with coordinates i = (ix, iy).
    for ix, iy in product(range(N), range(N)):
        # Iterate over all lattice sites with coordinates j = (jx, jy).
        for jx, jy in product(range(N), range(N)):
            # Skip all pairs of lattice sites i and j that are not
            # nearest-neighbors. The modulus operation accounts for
            # periodic boundary conditions on the lattice.
            if (abs(ix - jx) % N == 1 and iy == jy) or \
               (ix == jx and abs(iy - jy) % N == 1):
                # Add a hopping term.
                # Functions c_dag() and c() that return fermionic
                # creation/annihilation operators.
                H_e += -t * c_dag(ix, iy, spin) * c(jx, jy, spin)

# Frequency of the localized phonon.
w0 = 0.5

#
# Hamiltonian of phonons localized at lattice sites.
# We want this object to have the same type as H_e.
#
H_ph = ExpressionR()

# Iterate over all lattice sites.
for ix, iy in product(range(N), range(N)):
    # Energy of the localized phonon at site (ix, iy).
    # Functions a_dag() and a() that return bosonic creation/annihilation
    # operators.
    H_ph += w0 * a_dag(ix, iy) * a(ix, iy)

# Electron-phonon coupling constant.
g = 0.1

#
# Hamiltonian of electron-phonon coupling.
#
H_e_ph = ExpressionR()

# Iterate over spin projections.
for spin in ("up", "down"):
    # Iterate over all lattice sites.
    for ix, iy in product(range(N), range(N)):
        # Electron-phonon coupling at site (ix, iy).
        # Function n() returns the fermionic number operator n = c_dag * c.
        H_e_ph += g * n(ix, iy, spin) * (a_dag(ix, iy) + a(ix, iy))

# Complete Holstein Hamiltonian.
H_H = H_e + H_ph + H_e_ph

# Print H_H. There will be quite a lot of terms for the 100-site lattice!
print("H_H =", H_H)

# Check hermiticity of H_H.
print("H_H - H_H^\\dagger =", H_H - conj(H_H))

# Check that H_H commutes with the total number of electrons N_e.
N_e = ExpressionR()
for spin in ("up", "down"):
    for ix, iy in product(range(N), range(N)):
        N_e += n(ix, iy, spin)

print("[H_H, N_e] =", H_H * N_e - N_e * H_H)
