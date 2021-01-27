#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from itertools import product

from pycommute.expression import *
from pycommute.loperator import *

import numpy as np

class TestSpacePartition(TestCase):
    """Automatic Hilbert space partition"""

    @classmethod
    def setUpClass(cls):
        # Parameters of the 3 orbital Hubbard-Kanamori atom
        cls.n_orbs = 3
        cls.mu = 0.7
        cls.U = 3.0
        cls.J = 0.3

        orbs = range(cls.n_orbs)

        # Hamiltonian
        cls.H = ExpressionR()

        # Chemical potential
        cls.H += sum(-cls.mu * (n("up", o) + n("dn", o)) for o in orbs)

        # Intraorbital interactions
        cls.H += sum(cls.U * n("up", o) * n("dn", o) for o in orbs)

        # Interorbital interactions, different spins
        for o1, o2 in product(orbs, orbs):
            if o1 == o2: continue
            cls.H += (cls.U - 2 * cls.J) * n("up", o1) * n("dn", o2)

        # Interorbital interactions, equal spins
        for o1, o2 in product(orbs, orbs):
            if o2 >= o1: continue
            cls.H += (cls.U - 3 * cls.J) * n("up", o1) * n("up", o2)
            cls.H += (cls.U - 3 * cls.J) * n("dn", o1) * n("dn", o2)

        # Spin flips and pair hoppings
        for o1, o2 in product(orbs, orbs):
            if o1 == o2: continue
            cls.H += -cls.J * c_dag("up", o1) * c_dag("dn", o1) \
                            * c("up", o2) * c("dn", o2)
            cls.H += -cls.J * c_dag("up", o1) * c_dag("dn", o2) \
                            * c("up", o2) * c("dn", o1)
        # Hilbert space
        cls.hs = HilbertSpace(cls.H)
        # Linear operator form of the Hamiltonian
        cls.Hop = LOperatorR(cls.H, cls.hs)

    def extract_indices(self):
        orbs = range(self.n_orbs)
        bit_range_begin = lambda ind: self.hs.bit_range(ESpaceFermion(ind))[0]
        return [1 << bit_range_begin(Indices("dn", o)) for o in orbs] + \
               [1 << bit_range_begin(Indices("up", o)) for o in orbs]

    def test_space_partition(self):
        sp, matrix_elements = make_space_partition(self.Hop, self.hs, False)
        self.assertEqual(matrix_elements, {})

        self.assertEqual(sp.dim, 64)
        self.assertEqual(sp.n_subspaces, 44)

        # Calculated classification of states
        # Sets are used to neglect order of subspaces and of states within
        # a subspace
        v_cl = [set() for _ in range(sp.n_subspaces)]
        foreach(sp, lambda i, subspace: v_cl[subspace].add(i))
        cl = set(map(frozenset, v_cl))

        d0, d1, d2, u0, u1, u2 = self.extract_indices()

        # Expected classification of states
        ref_cl = set([
        # N=0
        frozenset([0]),
        #/N=1
        frozenset([d0]),
        frozenset([d1]),
        frozenset([d2]),
        frozenset([u0]),
        frozenset([u1]),
        frozenset([u2]),
        # N=2, same spin
        frozenset([d0 + d1]),
        frozenset([d0 + d2]),
        frozenset([d1 + d2]),
        frozenset([u0 + u1]),
        frozenset([u0 + u2]),
        frozenset([u1 + u2]),
        # N=2, pair hopping
        frozenset([d0 + u0, d1 + u1, d2 + u2]),
        # N=2, spin flip
        frozenset([d0 + u1, d1 + u0]),
        frozenset([d0 + u2, d2 + u0]),
        frozenset([d1 + u2, d2 + u1]),
        # N=3
        frozenset([d0 + d1 + d2]),
        frozenset([u0 + u1 + u2]),
        frozenset([d0 + d1 + u0, d1 + d2 + u2]),
        frozenset([d0 + d2 + u0, d1 + d2 + u1]),
        frozenset([d0 + d1 + u1, d0 + d2 + u2]),
        frozenset([d0 + u0 + u1, d2 + u1 + u2]),
        frozenset([d1 + u0 + u1, d2 + u0 + u2]),
        frozenset([d0 + u0 + u2, d1 + u1 + u2]),
        frozenset([d1 + d2 + u0, d0 + d2 + u1, d0 + d1 + u2]),
        frozenset([d2 + u0 + u1, d0 + u1 + u2, d1 + u0 + u2]),
        # N=4, 2 holes with the same spin
        frozenset([d2 + u0 + u1 + u2]),
        frozenset([d1 + u0 + u1 + u2]),
        frozenset([d0 + u0 + u1 + u2]),
        frozenset([d0 + d1 + d2 + u2]),
        frozenset([d0 + d1 + d2 + u1]),
        frozenset([d0 + d1 + d2 + u0]),
        # N=4, pair hopping
        frozenset([d1 + d2 + u1 + u2, d0 + d2 + u0 + u2, d0 + d1 + u0 + u1]),
        # N=4, spin flip
        frozenset([d1 + d2 + u0 + u2, d0 + d2 + u1 + u2]),
        frozenset([d1 + d2 + u0 + u1, d0 + d1 + u1 + u2]),
        frozenset([d0 + d2 + u0 + u1, d0 + d1 + u0 + u2]),
        # N=5
        frozenset([d1 + d2 + u0 + u1 + u2]),
        frozenset([d0 + d2 + u0 + u1 + u2]),
        frozenset([d0 + d1 + u0 + u1 + u2]),
        frozenset([d0 + d1 + d2 + u1 + u2]),
        frozenset([d0 + d1 + d2 + u0 + u2]),
        frozenset([d0 + d1 + d2 + u0 + u1]),
        # N=6
        frozenset([d0 + d1 + d2 + u0 + u1 + u2])
        ])

        self.assertEqual(cl, ref_cl)

    def test_matrix_elements(self):
        sp, matrix_elements = make_space_partition(self.Hop, self.hs, True)

        M = lambda i, j, val: (i, j, round(val, 9))

        v_melem = [set() for _ in range(sp.n_subspaces)]
        for i, j in matrix_elements:
            val = matrix_elements[(i, j)]
            v_melem[sp[i]].add(M(i, j, val))
        melem = set(map(frozenset, v_melem))

        d0, d1, d2, u0, u1, u2 = self.extract_indices()
        fs = lambda *args: frozenset([*args])
        mu, U, J = self.mu, self.U, self.J

        # Expected matrix elements
        ref_melem = set([
        # N=0
        fs(),
        # N=1
        fs(M(d0, d0, -mu)), fs(M(d1, d1, -mu)), fs(M(d2, d2, -mu)),
        fs(M(u0, u0, -mu)), fs(M(u1, u1, -mu)), fs(M(u2, u2, -mu)),
        # N=2, same spin
        fs(M(d0 + d1, d0 + d1, -2 * mu + U - 3 * J)),
        fs(M(d0 + d2, d0 + d2, -2 * mu + U - 3 * J)),
        fs(M(d1 + d2, d1 + d2, -2 * mu + U - 3 * J)),
        fs(M(u0 + u1, u0 + u1, -2 * mu + U - 3 * J)),
        fs(M(u0 + u2, u0 + u2, -2 * mu + U - 3 * J)),
        fs(M(u1 + u2, u1 + u2, -2 * mu + U - 3 * J)),
        # N=2, pair hopping
        fs(M(d0 + u0, d0 + u0, -2 * mu + U),
        M(d1 + u1, d1 + u1, -2 * mu + U),
        M(d2 + u2, d2 + u2, -2 * mu + U),
        M(d0 + u0, d1 + u1, J), M(d0 + u0, d2 + u2, J), M(d1 + u1, d2 + u2, J),
        M(d1 + u1, d0 + u0, J), M(d2 + u2, d0 + u0, J), M(d2 + u2, d1 + u1, J)),
        # N=2, spin flip
        fs(M(d0 + u1, d0 + u1, -2 * mu + U - 2 * J),
        M(d1 + u0, d1 + u0, -2 * mu + U - 2 * J),
        M(d0 + u1, d1 + u0, J), M(d1 + u0, d0 + u1, J)),
        fs(M(d0 + u2, d0 + u2, -2 * mu + U - 2 * J),
        M(d2 + u0, d2 + u0, -2 * mu + U - 2 * J),
        M(d0 + u2, d2 + u0, J), M(d2 + u0, d0 + u2, J)),
        fs(M(d1 + u2, d1 + u2, -2 * mu + U - 2 * J),
        M(d2 + u1, d2 + u1, -2 * mu + U - 2 * J),
        M(d1 + u2, d2 + u1, J), M(d2 + u1, d1 + u2, J)),
        # N=3
        fs(M(d0 + d1 + d2, d0 + d1 + d2, -3 * mu + 3 * U - 9 * J)),
        fs(M(u0 + u1 + u2, u0 + u1 + u2, -3 * mu + 3 * U - 9 * J)),
        fs(M(d0 + d1 + u0, d0 + d1 + u0, -3 * mu + 3 * U - 5 * J),
        M(d1 + d2 + u2, d1 + d2 + u2, -3 * mu + 3 * U - 5 * J),
        M(d0 + d1 + u0, d1 + d2 + u2, -J), M(d1 + d2 + u2, d0 + d1 + u0, -J)),
        fs(M(d0 + d2 + u0, d0 + d2 + u0, -3 * mu + 3 * U - 5 * J),
        M(d1 + d2 + u1, d1 + d2 + u1, -3 * mu + 3 * U - 5 * J),
        M(d0 + d2 + u0, d1 + d2 + u1, J), M(d1 + d2 + u1, d0 + d2 + u0, J)),
        fs(M(d0 + d1 + u1, d0 + d1 + u1, -3 * mu + 3 * U - 5 * J),
        M(d0 + d2 + u2, d0 + d2 + u2, -3 * mu + 3 * U - 5 * J),
        M(d0 + d1 + u1, d0 + d2 + u2, J), M(d0 + d2 + u2, d0 + d1 + u1, J)),
        fs(M(d0 + u0 + u1, d0 + u0 + u1, -3 * mu + 3 * U - 5 * J),
        M(d2 + u1 + u2, d2 + u1 + u2, -3 * mu + 3 * U - 5 * J),
        M(d0 + u0 + u1, d2 + u1 + u2, -J), M(d2 + u1 + u2, d0 + u0 + u1, -J)),
        fs(M(d1 + u0 + u1, d1 + u0 + u1, -3 * mu + 3 * U - 5 * J),
        M(d2 + u0 + u2, d2 + u0 + u2, -3 * mu + 3 * U - 5 * J),
        M(d1 + u0 + u1, d2 + u0 + u2, J), M(d2 + u0 + u2, d1 + u0 + u1, J)),
        fs(M(d0 + u0 + u2, d0 + u0 + u2, -3 * mu + 3 * U - 5 * J),
        M(d1 + u1 + u2, d1 + u1 + u2, -3 * mu + 3 * U - 5 * J),
        M(d0 + u0 + u2, d1 + u1 + u2, J), M(d1 + u1 + u2, d0 + u0 + u2, J)),
        fs(M(d1 + d2 + u0, d1 + d2 + u0, -3 * mu + 3 * U - 7 * J),
        M(d0 + d2 + u1, d0 + d2 + u1, -3 * mu + 3 * U - 7 * J),
        M(d0 + d1 + u2, d0 + d1 + u2, -3 * mu + 3 * U - 7 * J),
        M(d1 + d2 + u0, d0 + d2 + u1, J),
        M(d0 + d2 + u1, d1 + d2 + u0, J), M(d1 + d2 + u0, d0 + d1 + u2, -J),
        M(d0 + d1 + u2, d1 + d2 + u0, -J), M(d0 + d2 + u1, d0 + d1 + u2, J),
        M(d0 + d1 + u2, d0 + d2 + u1, J)),
        fs(M(d2 + u0 + u1, d2 + u0 + u1, -3 * mu + 3 * U - 7 * J),
        M(d0 + u1 + u2, d0 + u1 + u2, -3 * mu + 3 * U - 7 * J),
        M(d1 + u0 + u2, d1 + u0 + u2, -3 * mu + 3 * U - 7 * J),
        M(d2 + u0 + u1, d0 + u1 + u2, -J),
        M(d0 + u1 + u2, d2 + u0 + u1, -J), M(d2 + u0 + u1, d1 + u0 + u2, J),
        M(d1 + u0 + u2, d2 + u0 + u1, J), M(d0 + u1 + u2, d1 + u0 + u2, J),
        M(d1 + u0 + u2, d0 + u1 + u2, J)),
        # N=4, 2 holes with the same spin
        fs(M(d2 + u0 + u1 + u2, d2 + u0 + u1 + u2, -4 * mu + 6 * U - 13 * J)),
        fs(M(d1 + u0 + u1 + u2, d1 + u0 + u1 + u2, -4 * mu + 6 * U - 13 * J)),
        fs(M(d0 + u0 + u1 + u2, d0 + u0 + u1 + u2, -4 * mu + 6 * U - 13 * J)),
        fs(M(d0 + d1 + d2 + u0, d0 + d1 + d2 + u0, -4 * mu + 6 * U - 13 * J)),
        fs(M(d0 + d1 + d2 + u1, d0 + d1 + d2 + u1, -4 * mu + 6 * U - 13 * J)),
        fs(M(d0 + d1 + d2 + u2, d0 + d1 + d2 + u2, -4 * mu + 6 * U - 13 * J)),
        # N=4, pair hopping
        fs(M(d1 + d2 + u1 + u2, d1 + d2 + u1 + u2, -4 * mu + 6 * U - 10 * J),
        M(d0 + d2 + u0 + u2, d0 + d2 + u0 + u2, -4 * mu + 6 * U - 10 * J),
        M(d0 + d1 + u0 + u1, d0 + d1 + u0 + u1, -4 * mu + 6 * U - 10 * J),
        M(d1 + d2 + u1 + u2, d0 + d2 + u0 + u2, J),
        M(d0 + d2 + u0 + u2, d1 + d2 + u1 + u2, J),
        M(d1 + d2 + u1 + u2, d0 + d1 + u0 + u1, J),
        M(d0 + d1 + u0 + u1, d1 + d2 + u1 + u2, J),
        M(d0 + d2 + u0 + u2, d0 + d1 + u0 + u1, J),
        M(d0 + d1 + u0 + u1, d0 + d2 + u0 + u2, J)),
        # N=4, spin flip
        fs(M(d1 + d2 + u0 + u2, d1 + d2 + u0 + u2, -4 * mu + 6 * U - 12 * J),
        M(d0 + d2 + u1 + u2, d0 + d2 + u1 + u2, -4 * mu + 6 * U - 12 * J),
        M(d1 + d2 + u0 + u2, d0 + d2 + u1 + u2, J),
        M(d0 + d2 + u1 + u2, d1 + d2 + u0 + u2, J)),
        fs(M(d1 + d2 + u0 + u1, d1 + d2 + u0 + u1, -4 * mu + 6 * U - 12 * J),
        M(d0 + d1 + u1 + u2, d0 + d1 + u1 + u2, -4 * mu + 6 * U - 12 * J),
        M(d1 + d2 + u0 + u1, d0 + d1 + u1 + u2, J),
        M(d0 + d1 + u1 + u2, d1 + d2 + u0 + u1, J)),
        fs(M(d0 + d2 + u0 + u1, d0 + d2 + u0 + u1, -4 * mu + 6 * U - 12 * J),
        M(d0 + d1 + u0 + u2, d0 + d1 + u0 + u2, -4 * mu + 6 * U - 12 * J),
        M(d0 + d2 + u0 + u1, d0 + d1 + u0 + u2, J),
        M(d0 + d1 + u0 + u2, d0 + d2 + u0 + u1, J)),
        # N=5
        fs(M(d1 + d2 + u0 + u1 + u2,
        d1 + d2 + u0 + u1 + u2,
        -5 * mu + 10 * U - 20 * J)),
        fs(M(d0 + d2 + u0 + u1 + u2,
        d0 + d2 + u0 + u1 + u2,
        -5 * mu + 10 * U - 20 * J)),
        fs(M(d0 + d1 + u0 + u1 + u2,
        d0 + d1 + u0 + u1 + u2,
        -5 * mu + 10 * U - 20 * J)),
        fs(M(d0 + d1 + d2 + u1 + u2,
        d0 + d1 + d2 + u1 + u2,
        -5 * mu + 10 * U - 20 * J)),
        fs(M(d0 + d1 + d2 + u0 + u2,
        d0 + d1 + d2 + u0 + u2,
        -5 * mu + 10 * U - 20 * J)),
        fs(M(d0 + d1 + d2 + u0 + u1,
        d0 + d1 + d2 + u0 + u1,
        -5 * mu + 10 * U - 20 * J)),
        # N=6
        fs(M(d0 + d1 + d2 + u0 + u1 + u2,
        d0 + d1 + d2 + u0 + u1 + u2,
        -6 * mu + 15 * U - 30 * J))
        ])

        self.assertEqual(melem, ref_melem)

    def test_merge_subspaces(self):
        sp, melem = make_space_partition(self.Hop, self.hs, False)

        Cd, C, all_ops = [], [], []
        for spin in ("dn", "up"):
            for o in range(self.n_orbs):
                Cd.append(LOperatorR(c_dag(spin, 0), self.hs))
                C.append(LOperatorR(c(spin, 0), self.hs))

                all_ops.append(Cd[-1])
                all_ops.append(C[-1])

                sp.merge_subspaces(Cd[-1], C[-1], self.hs)

        # Calculated classification of states
        v_cl = [set() for _ in range(sp.n_subspaces)]
        foreach(sp, lambda i, subspace: v_cl[subspace].add(i))
        cl = set(map(frozenset, v_cl))

        in_state = np.zeros((sp.dim,), dtype = float)

        for op in all_ops:
            for i_sp in cl:
                f_sp = set()
                for i in i_sp:
                    in_state[i] = 1.0
                    out_state = op * in_state
                    for f, a in enumerate(out_state):
                        if abs(a) > 1e-10: f_sp.add(f)
                    in_state[i] = 0.0

                # op maps i_sp onto zero
                if len(f_sp) == 0: continue

                # Check if op maps i_sp to only one subspace
                self.assertEqual(
                  sum(int(f_sp.issubset(f_sp_ref)) for f_sp_ref in cl),
                  1
                )
