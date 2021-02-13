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

from pycommute.expression import (Indices, c, c_dag, a_dag)
from pycommute.loperator import (
    ESpaceFermion,
    HilbertSpace,
    LOperatorR, LOperatorC,
    BasisMapper
)

import numpy as np
from numpy.testing import assert_equal


class TestMappedBasisView(TestCase):
    """Basis-mapped view of a state vector"""

    def assert_equal_dicts_up_to_value_permutation(self, d1, d2):
        self.assertEqual(d1.keys(), d2.keys())
        self.assertEqual(set(d1.values()), set(d2.values()))

    @classmethod
    def setUpClass(cls):
        # Finite system of 4 fermions: 2 orbitals, two spin projections

        # Hamiltonian: spin flips
        cls.Hex = \
            2 * c_dag("up", 1) * c("up", 2) * c_dag("dn", 2) * c("dn", 1) \
            + 2 * c_dag("up", 2) * c("up", 1) * c_dag("dn", 1) * c("dn", 2)
        cls.Hp = \
            2 * c_dag("up", 1) * c("up", 2) * c_dag("dn", 1) * c("dn", 2) \
            + 2 * c_dag("up", 2) * c("up", 1) * c_dag("dn", 2) * c("dn", 1)

        cls.hs = HilbertSpace(cls.Hex + cls.Hp)

        # 3 = 1 + 2 -> |dn>_1 |dn>_2
        # 5 = 1 + 4 -> |dn>_1 |up>_1
        # 6 = 2 + 4 -> |dn>_2 |up>_1
        # 9 = 1 + 8 -> |dn>_1 |up>_2
        # 10 = 2 + 8 -> |dn>_2 |up>_2
        # 12 = 4 + 8 -> |up>_1 |up>_2

        # Map all basis states with 2 electrons so that their indices
        # are contiguous
        cls.mapping = {i: j for j, i in enumerate([3, 5, 6, 9, 10, 12])}
        cls.mapper = BasisMapper([3, 5, 6, 9, 10, 12])

        cls.st = np.array([0, 1, 2, 3, 4, 5], dtype=float)

    def test_map(self):
        self.assertEqual(self.hs.dim, 16)

        def bit_range(ind):
            return self.hs.bit_range(ESpaceFermion(ind))

        self.assertEqual(bit_range(Indices("dn", 1)), (0, 0))
        self.assertEqual(bit_range(Indices("dn", 2)), (1, 1))
        self.assertEqual(bit_range(Indices("up", 1)), (2, 2))
        self.assertEqual(bit_range(Indices("up", 2)), (3, 3))

        self.assertEqual(len(self.mapper), 6)

    def test_loperator(self):
        out = np.zeros((6,), dtype=float)
        dst = self.mapper(out)

        # Spin flips
        Hop = LOperatorR(self.Hex, self.hs)

        in1 = np.array([1, 1, 1, 1, 1, 1], dtype=float)
        src = self.mapper(in1)
        Hop(src, dst)
        assert_equal(out, np.array([0, 0, 2, 2, 0, 0], dtype=float))

        in2 = np.array([1, 1, 1, -1, 1, 1], dtype=float)
        src = self.mapper(in2)
        Hop(src, dst)
        assert_equal(out, np.array([0, 0, -2, 2, 0, 0], dtype=float))

        # Pair hops
        Hop = LOperatorR(self.Hp, self.hs)

        src = self.mapper(in1)
        Hop(src, dst)
        assert_equal(out, np.array([0, 2, 0, 0, 2, 0], dtype=float))

        in2 = np.array([1, 1, 1, 1, -1, 1], dtype=float)
        src = self.mapper(in2)
        Hop(src, dst)
        assert_equal(out, np.array([0, -2, 0, 0, 2, 0], dtype=float))

    def test_basis_state_indices(self):
        basis_indices = [3, 5, 6, 9, 10, 12]
        mapper = BasisMapper(basis_indices)
        self.assertEqual(len(mapper), 6)
        self.assertEqual(mapper.map, self.mapping)

        inv_mapping = {j: i for i, j in self.mapping.items()}
        self.assertEqual(mapper.inverse_map, inv_mapping)

    def test_O_vac(self):
        P = c_dag("dn", 1) * c_dag("dn", 2) + \
            c_dag("dn", 1) * c_dag("up", 1) + \
            c_dag("dn", 2) * c_dag("up", 1) + \
            c_dag("dn", 1) * c_dag("up", 2) + \
            c_dag("dn", 2) * c_dag("up", 2) + \
            c_dag("up", 1) * c_dag("up", 2)

        mapper = BasisMapper(LOperatorR(P, self.hs), self.hs)
        self.assertEqual(len(mapper), 6)
        self.assert_equal_dicts_up_to_value_permutation(mapper.map,
                                                        self.mapping)

        mapper = BasisMapper(LOperatorC(1j * P, self.hs), self.hs)
        self.assertEqual(len(mapper), 6)
        self.assert_equal_dicts_up_to_value_permutation(mapper.map,
                                                        self.mapping)

    def test_compositions(self):
        mapper = BasisMapper([], self.hs, 0)
        self.assertEqual(len(mapper), 1)
        self.assertEqual(mapper.map[0], 0)

        O_list = [LOperatorR(c_dag("dn", 1), self.hs),
                  LOperatorR(c_dag("dn", 2), self.hs),
                  LOperatorR(c_dag("up", 1), self.hs),
                  LOperatorR(c_dag("up", 2), self.hs)]

        mapper_N0 = BasisMapper(O_list, self.hs, 0)
        self.assertEqual(len(mapper_N0), 1)
        self.assertEqual(mapper_N0.map[0], 0)

        mapper_N2 = BasisMapper(O_list, self.hs, 2)
        self.assert_equal_dicts_up_to_value_permutation(mapper_N2.map,
                                                        self.mapping)

        O_list_complex = [LOperatorC(1j * c_dag("dn", 1), self.hs),
                          LOperatorC(1j * c_dag("dn", 2), self.hs),
                          LOperatorC(1j * c_dag("up", 1), self.hs),
                          LOperatorC(1j * c_dag("up", 2), self.hs)]
        mapper_N2 = BasisMapper(O_list_complex, self.hs, 2)
        self.assert_equal_dicts_up_to_value_permutation(mapper_N2.map,
                                                        self.mapping)

    def test_compositions_bosons(self):
        hs = HilbertSpace(a_dag(1) + a_dag(2) + a_dag(3) + a_dag(4), 4)

        O_list = [LOperatorR(a_dag(1), hs),
                  LOperatorR(a_dag(2), hs),
                  LOperatorR(a_dag(3), hs),
                  LOperatorR(a_dag(4), hs)]

        map_size_ref = [1, 4, 10, 20, 35, 56, 84, 120, 165, 220]
        for N in range(10):
            mapper = BasisMapper(O_list, hs, N)
            self.assertEqual(len(mapper), map_size_ref[N])
