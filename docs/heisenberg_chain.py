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
# Periodic spin-1/2 Heisenberg chain and its integrals of motion
#
# Expressions for the integrals of motion are taken from
#
#   "Quantum Integrals of Motion for the Heisenberg Spin Chain",
#   M. P. Grabowski and P. Mathieu
#   Mod. Phys. Lett. A, Vol. 09, No. 24, pp. 2197-2206 (1994),
#   https://doi.org/10.1142/S0217732394002057
#

from numpy import array, dot, cross
from pycommute.expression import ExpressionR
from pycommute.expression import S_x, S_y, S_z

# Number of spins in the chain
N = 20
# Heisenberg exchange constant
g = 2

# List of 3-component spin vectors {S_0, S_1, ..., S_{N-1}}
S = [array([S_x(i), S_y(i), S_z(i)]) for i in range(N)]

# Hamiltonian of the spin-1/2 Heisenberg chain.
# Index shift modulo N ensures periodic boundary conditions.
H = sum(g * dot(S[i], S[(i+1)%N]) for i in range(N))

# Total spin of the chain.
S_tot = array(sum(S))

# All three components of S commute with the Hamiltonian.
print("[H, S_x] =", (H * S_tot[0] - S_tot[0] * H))
print("[H, S_y] =", (H * S_tot[1] - S_tot[1] * H))
print("[H, S_z] =", (H * S_tot[2] - S_tot[2] * H))

# Higher charge Q_3 (1st line of Eq. (10)).
Q3 = sum(dot(cross(S[i], S[(i+1)%N]), S[(i+2)%N]) for i in range(N))
print("[H, Q3] =", (H * Q3 - Q3 * H))

# Higher charge Q_4 (2nd line of Eq. (10)).
Q4 = sum(4.0 * dot(cross(cross(S[i], S[(i+1)%N]), S[(i+2)%N]), S[(i+3)%N])
         for i in range(N))
Q4 += sum(dot(S[i], S[(i+2)%N]) for i in range(N))
print("[H, Q4] =", (H * Q4 - Q4 * H))

# Higher charge Q_5 (3rd line of Eq. (10)).
Q5 = sum(4.0 * dot(
             cross(cross(cross(S[i], S[(i+1)%N]), S[(i+2)%N]), S[(i+3)%N]),
             S[(i+4)%N]
          ) for i in range(N))
Q5 += sum(dot(cross(S[i], S[(i+2)%N]), S[(i+3)%N]) for i in range(N))
Q5 += sum(dot(cross(S[i], S[(i+1)%N]), S[(i+3)%N]) for i in range(N))
print("[H, Q5] =", (H * Q5 - Q5 * H))

# Check that the higher charges pairwise commute.
print("[Q3, Q4] =", (Q3 * Q4 - Q4 * Q3))
print("[Q3, Q5] =", (Q3 * Q5 - Q5 * Q3))
print("[Q4, Q5] =", (Q4 * Q5 - Q5 * Q4))
