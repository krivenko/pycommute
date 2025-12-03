#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2025 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# ## Kanamori interaction Hamiltonian
# ### and its expression in terms of integrals of motion N, S^2 and L^2.
#
#   "Strong Correlations from Hund's Coupling",
#   A. Georges, L. de' Medici and J. Mravlje,
#   Annu. Rev. Condens. Matter Phys. 2013. 4:137–78,
#   https://doi.org/10.1146/annurev-conmatphys-020911-125045
#

from itertools import product

# Polynomial expression with real and complex coefficients
from pycommute.expression import ExpressionR, ExpressionC

# Factory functions returning expressions for fermionic creation, annihilation
# and occupation number operators.
from pycommute.expression import c_dag, c, n

# Orbital degeneracy of the shell (t_{2g} triplet)
n_orbs = 3
# Coulomb integrals U and J
U = 4.0
J = 0.2

# Kanamori Hamiltonian (Eq. (2))
H_K = ExpressionR()

# Intraorbital density-density interaction terms
for m in range(n_orbs):
    H_K += U * n(m, "up") * n(m, "down")

# Interorbital density-density interaction terms (different spins)
for m1, m2 in product(range(n_orbs), range(n_orbs)):
    if m1 == m2:
        continue
    H_K += (U - 2 * J) * n(m1, "up") * n(m2, "down")

# Interorbital density-density interaction terms (same spin)
for m1, m2 in product(range(n_orbs), range(n_orbs)):
    if m1 >= m2:
        continue
    H_K += (U - 3 * J) * n(m1, "up") * n(m2, "up")
    H_K += (U - 3 * J) * n(m1, "down") * n(m2, "down")

# Spin-flip terms
for m1, m2 in product(range(n_orbs), range(n_orbs)):
    if m1 == m2:
        continue
    H_K += \
        -J * c_dag(m1, "up") * c(m1, "down") * c_dag(m2, "down") * c(m2, "up")

# Pair-hopping terms
for m1, m2 in product(range(n_orbs), range(n_orbs)):
    if m1 == m2:
        continue
    H_K += J * c_dag(m1, "up") * c_dag(m1, "down") * c(m2, "down") * c(m2, "up")

# Print the Hamiltonian
print("H_K =", H_K)

#
# ### Integrals of motion N, S^2 and L^2 (Eq. (4))
#

# Total number of particles.
N = ExpressionR()
for m in range(n_orbs):
    N += n(m, "up") + n(m, "down")

# Total spin operators S_x, S_y, S_z.
Sx = ExpressionC()
Sy = ExpressionC()
Sz = ExpressionC()
for m in range(n_orbs):
    Sx += 0.5 * (c_dag(m, "up") * c(m, "down") + c_dag(m, "down") * c(m, "up"))
    Sy += 0.5j * (c_dag(m, "down") * c(m, "up") - c_dag(m, "up") * c(m, "down"))
    Sz += 0.5 * (n(m, "up") - n(m, "down"))

# Operator S^2 = S_x S_x + S_y S_y + S_z S_z.
S2 = Sx * Sx + Sy * Sy + Sz * Sz


# Levi-Civita symbol \epsilon_{ijk}.
def eps(i, j, k):
    return (j - i) * (k - j) * (k - i) / 2


# Orbital isospin generators L_x, L_y, L_z.
Lx = ExpressionC()
Ly = ExpressionC()
Lz = ExpressionC()

for spin in ("up", "down"):
    for m1, m2 in product(range(n_orbs), range(n_orbs)):
        Lx += 1j * eps(0, m1, m2) * c_dag(m1, spin) * c(m2, spin)
        Ly += 1j * eps(1, m1, m2) * c_dag(m1, spin) * c(m2, spin)
        Lz += 1j * eps(2, m1, m2) * c_dag(m1, spin) * c(m2, spin)

# Operator L^2 = L_x L_x + L_y L_y + L_z L_z.
L2 = Lx * Lx + Ly * Ly + Lz * Lz

#
# ### Hamiltonian as a function of N, S^2 and L^2 (Eq. (7))
#

H_t2g = (U - 3 * J) / 2 * N * (N - 1.0) - 2 * J * S2 - J / 2 * L2 \
    + (5.0 / 2) * J * N

# Must be zero
print("H_K - H_t2g =", H_K - H_t2g)
