#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2025 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from copy import copy, deepcopy

from pycommute.expression import (
    ExpressionR,
    FERMION, BOSON, SPIN,
    n, a, a_dag, S_p, S_m, S_z
)
from pycommute.loperator import (
    HilbertSpace,
    make_space_fermion, make_space_boson, make_space_spin,
    foreach
)


class TestHilbertSpace(TestCase):

    @classmethod
    def setUpClass(cls):
        # Fermionic elementary spaces
        cls.es_f_dn = make_space_fermion("dn", 0)
        cls.es_f_up = make_space_fermion("up", 0)
        cls.fermion_es = [cls.es_f_dn, cls.es_f_up]
        # Bosonic elementary spaces
        cls.es_b_x = make_space_boson(13, "x", 0)
        cls.es_b_y = make_space_boson(8, "y", 0)
        cls.boson_es = [cls.es_b_x, cls.es_b_y]
        # Spin-1/2 elementary spaces
        cls.es_s_i = make_space_spin(0.5, "i", 0)
        cls.es_s_j = make_space_spin(0.5, "j", 0)
        cls.spin_es = [cls.es_s_i, cls.es_s_j]
        # Spin-1 elementary spaces
        cls.es_s1_i = make_space_spin(1.0, "i", 0)
        cls.es_s1_j = make_space_spin(1.0, "j", 0)
        cls.spin1_es = [cls.es_s1_i, cls.es_s1_j]
        # Spin-3/2 elementary spaces
        cls.es_s32_i = make_space_spin(3 / 2, "i", 0)
        cls.es_s32_j = make_space_spin(3 / 2, "j", 0)
        cls.spin32_es = [cls.es_s32_i, cls.es_s32_j]

    def test_equality(self):
        hs_empty = HilbertSpace()
        self.assertTrue(hs_empty == hs_empty)
        self.assertFalse(hs_empty != hs_empty)

        hs = HilbertSpace([
            self.es_s32_i, self.es_s32_j,
            self.es_s1_i, self.es_s1_j,
            self.es_s_i, self.es_s_j,
            self.es_b_x, self.es_b_y,
            self.es_f_dn, self.es_f_up
        ])
        self.assertTrue(hs == hs)
        self.assertFalse(hs != hs)

        self.assertFalse(hs == hs_empty)
        self.assertTrue(hs != hs_empty)

    def test_constructors(self):
        hs_empty = HilbertSpace()
        self.assertEqual(len(hs_empty), 0)
        self.assertEqual(hs_empty.total_n_bits, 0)
        self.assertEqual(hs_empty.dim, 1)
        self.assertEqual(hs_empty.vec_size, 1)
        self.assertFalse(hs_empty.is_sparse)
        es_list = []
        hs_empty.foreach_elementary_space(lambda es: es_list.append(es))
        self.assertEqual(es_list, [])

        hs1 = HilbertSpace([
            self.es_s32_i, self.es_s32_j,
            self.es_s1_i, self.es_s1_j,
            self.es_s_i, self.es_s_j,
            self.es_b_x, self.es_b_y,
            self.es_f_dn, self.es_f_up
        ])
        self.assertEqual(len(hs1), 10)
        self.assertEqual(hs1.total_n_bits, 19)
        self.assertEqual(hs1.dim, 239616)
        self.assertEqual(hs1.vec_size, 524288)
        self.assertTrue(hs1.is_sparse)

        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space already exists$"):
            HilbertSpace([self.es_s32_i,
                          self.es_s_i,
                          self.es_s_i,
                          self.es_f_dn])

        self.assertEqual(HilbertSpace(self.fermion_es),
                         HilbertSpace([self.es_f_dn, self.es_f_up]))
        self.assertEqual(HilbertSpace(self.boson_es),
                         HilbertSpace([self.es_b_x, self.es_b_y]))
        self.assertEqual(HilbertSpace(self.spin_es),
                         HilbertSpace([self.es_s_i, self.es_s_j]))
        self.assertEqual(HilbertSpace(self.spin1_es),
                         HilbertSpace([self.es_s1_i, self.es_s1_j]))
        self.assertEqual(HilbertSpace(self.spin32_es),
                         HilbertSpace([self.es_s32_i, self.es_s32_j]))

    def test_copy(self):
        hs = HilbertSpace([
            self.es_s32_i, self.es_s32_j,
            self.es_s1_i, self.es_s1_j,
            self.es_s_i, self.es_s_j,
            self.es_b_x, self.es_b_y,
            self.es_f_dn, self.es_f_up
        ])
        hsc = copy(hs)
        self.assertEqual(hsc, hs)
        self.assertNotEqual(id(hsc), id(hs))
        hsdc = deepcopy(hs)
        self.assertEqual(hsdc, hs)
        self.assertNotEqual(id(hsdc), id(hs))

    def test_attributes(self):
        hs = HilbertSpace([
            self.es_s32_i,
            self.es_s32_j,
            self.es_s1_j,
            self.es_s_i, self.es_s_j,
            self.es_b_x,
            self.es_f_dn, self.es_f_up
        ])
        self.assertEqual(len(hs), 8)
        self.assertEqual(hs.total_n_bits, 14)
        self.assertEqual(hs.dim, 9984)
        self.assertEqual(hs.vec_size, 16384)
        self.assertTrue(hs.is_sparse)
        self.assertTrue(hs.has_algebra(FERMION))
        self.assertTrue(hs.has_algebra(BOSON))
        self.assertTrue(hs.has_algebra(SPIN))
        self.assertEqual(hs.algebra_bit_range(FERMION), (0, 1))
        self.assertEqual(hs.algebra_bit_range(BOSON), (2, 5))
        self.assertEqual(hs.algebra_bit_range(SPIN), (6, 13))

        es_list = []
        es_ref = [self.es_f_dn,
                  self.es_f_up,
                  self.es_b_x,
                  self.es_s_i,
                  self.es_s_j,
                  self.es_s1_j,
                  self.es_s32_i,
                  self.es_s32_j]
        hs.foreach_elementary_space(lambda es: es_list.append(es))
        self.assertEqual(es_list, es_ref)

        self.assertTrue(self.es_f_dn in hs)
        self.assertEqual(hs.index(self.es_f_dn), 0)
        self.assertEqual(hs.es_dim(self.es_f_dn), 2)
        self.assertEqual(hs.bit_range(self.es_f_dn), (0, 0))
        self.assertEqual(hs.basis_state_index(self.es_f_dn, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_f_dn, 1), 1)
        self.assertTrue(self.es_f_up in hs)
        self.assertEqual(hs.index(self.es_f_up), 1)
        self.assertEqual(hs.es_dim(self.es_f_up), 2)
        self.assertEqual(hs.bit_range(self.es_f_up), (1, 1))
        self.assertEqual(hs.basis_state_index(self.es_f_up, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_f_up, 1), 2)
        self.assertTrue(self.es_b_x in hs)
        self.assertEqual(hs.index(self.es_b_x), 2)
        self.assertEqual(hs.es_dim(self.es_b_x), 13)
        self.assertEqual(hs.bit_range(self.es_b_x), (2, 5))
        self.assertEqual(hs.basis_state_index(self.es_b_x, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_b_x, 1), 4)
        self.assertEqual(hs.basis_state_index(self.es_b_x, 5), 20)
        self.assertFalse(self.es_b_y in hs)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.index(self.es_b_y)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.es_dim(self.es_b_y)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.bit_range(self.es_b_y)
        self.assertTrue(self.es_s_i in hs)
        self.assertEqual(hs.index(self.es_s_i), 3)
        self.assertEqual(hs.es_dim(self.es_s_i), 2)
        self.assertEqual(hs.bit_range(self.es_s_i), (6, 6))
        self.assertEqual(hs.basis_state_index(self.es_s_i, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s_i, 1), 64)
        self.assertTrue(self.es_s_j in hs)
        self.assertEqual(hs.index(self.es_s_j), 4)
        self.assertEqual(hs.es_dim(self.es_s_j), 2)
        self.assertEqual(hs.bit_range(self.es_s_j), (7, 7))
        self.assertFalse(self.es_s1_i in hs)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.index(self.es_s1_i)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.es_dim(self.es_s1_i)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.bit_range(self.es_s1_i)
        self.assertTrue(self.es_s1_j in hs)
        self.assertEqual(hs.index(self.es_s1_j), 5)
        self.assertEqual(hs.es_dim(self.es_s1_j), 3)
        self.assertEqual(hs.bit_range(self.es_s1_j), (8, 9))
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 1), 256)
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 2), 512)
        self.assertTrue(self.es_s32_i in hs)
        self.assertEqual(hs.index(self.es_s32_i), 6)
        self.assertEqual(hs.es_dim(self.es_s32_i), 4)
        self.assertEqual(hs.bit_range(self.es_s32_i), (10, 11))
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 1), 1024)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 2), 2048)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 3), 3072)
        self.assertTrue(self.es_s32_j in hs)
        self.assertEqual(hs.index(self.es_s32_j), 7)
        self.assertEqual(hs.es_dim(self.es_s32_j), 4)
        self.assertEqual(hs.bit_range(self.es_s32_j), (12, 13))
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 1), 4096)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 2), 8192)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 3), 12288)

    def test_add(self):
        hs = HilbertSpace([self.es_s32_i])

        def check_hs(es,
                     index,
                     size,
                     dim,
                     total_n_bits,
                     b, e,
                     fermion, boson, spin):
            self.assertEqual(len(hs), size)
            self.assertEqual(hs.total_n_bits, total_n_bits)
            self.assertEqual(hs.dim, dim)
            self.assertEqual(hs.vec_size, 2 ** total_n_bits)
            self.assertEqual(hs.is_sparse, hs.dim != 2 ** total_n_bits)
            self.assertTrue(es in hs)
            self.assertEqual(hs.index(es), index)
            self.assertEqual(hs.bit_range(es), (b, e))

            if fermion is not None:
                self.assertTrue(hs.has_algebra(FERMION))
                self.assertEqual(hs.algebra_bit_range(FERMION), fermion)
            else:
                self.assertFalse(hs.has_algebra(FERMION))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % FERMION
                ):
                    hs.algebra_bit_range(FERMION)

            if boson is not None:
                self.assertTrue(hs.has_algebra(BOSON))
                self.assertEqual(hs.algebra_bit_range(BOSON), boson)
            else:
                self.assertFalse(hs.has_algebra(BOSON))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % BOSON
                ):
                    hs.algebra_bit_range(BOSON)

            if spin is not None:
                self.assertTrue(hs.has_algebra(SPIN))
                self.assertEqual(hs.algebra_bit_range(SPIN), spin)
            else:
                self.assertFalse(hs.has_algebra(SPIN))
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % SPIN
                ):
                    hs.algebra_bit_range(SPIN)

        check_hs(self.es_s32_i, 0, 1, 4, 2, 0, 1, None, None, (0, 1))
        hs.add(self.es_s32_j)
        check_hs(self.es_s32_j, 1, 2, 16, 4, 2, 3, None, None, (0, 3))
        hs.add(self.es_s1_j)
        check_hs(self.es_s1_j, 0, 3, 48, 6, 0, 1, None, None, (0, 5))
        hs.add(self.es_s_i)
        check_hs(self.es_s_i, 0, 4, 96, 7, 0, 0, None, None, (0, 6))
        hs.add(self.es_s_j)
        check_hs(self.es_s_j, 1, 5, 192, 8, 1, 1, None, None, (0, 7))
        hs.add(self.es_b_x)
        check_hs(self.es_b_x, 0, 6, 2496, 12, 0, 3, None, (0, 3), (4, 11))
        hs.add(self.es_f_dn)
        check_hs(self.es_f_dn, 0, 7, 4992, 13, 0, 0, (0, 0), (1, 4), (5, 12))
        hs.add(self.es_f_up)
        check_hs(self.es_f_up, 1, 8, 9984, 14, 1, 1, (0, 1), (2, 5), (6, 13))

        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space already exists$"):
            hs.add(self.es_s_j)

        check_hs(self.es_f_dn, 0, 8, 9984, 14, 0, 0, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_f_up, 1, 8, 9984, 14, 1, 1, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_b_x, 2, 8, 9984, 14, 2, 5, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s_i, 3, 8, 9984, 14, 6, 6, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s_j, 4, 8, 9984, 14, 7, 7, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s1_j, 5, 8, 9984, 14, 8, 9, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s32_i, 6, 8, 9984, 14, 10, 11, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s32_j, 7, 8, 9984, 14, 12, 13, (0, 1), (2, 5), (6, 13))
        self.assertEqual(hs.algebra_bit_range(FERMION), (0, 1))
        self.assertEqual(hs.algebra_bit_range(BOSON), (2, 5))
        self.assertEqual(hs.algebra_bit_range(SPIN), (6, 13))

    def test_from_expression(self):
        expr = 2.0 * S_p("i", 0, spin=3 / 2) * S_m("j", 0, spin=3 / 2) \
            + 5.0 * n("up", 0) * n("dn", 0)

        hs1 = HilbertSpace(expr)
        self.assertEqual(len(hs1), 4)
        self.assertEqual(hs1.total_n_bits, 6)
        self.assertEqual(hs1.dim, 64)
        self.assertEqual(hs1.vec_size, 64)
        self.assertFalse(hs1.is_sparse)
        self.assertTrue(self.es_f_dn in hs1)
        self.assertEqual(hs1.index(self.es_f_dn), 0)
        self.assertEqual(hs1.es_dim(self.es_f_dn), 2)
        self.assertEqual(hs1.bit_range(self.es_f_dn), (0, 0))
        self.assertTrue(self.es_f_up in hs1)
        self.assertEqual(hs1.index(self.es_f_up), 1)
        self.assertEqual(hs1.es_dim(self.es_f_up), 2)
        self.assertEqual(hs1.bit_range(self.es_f_up), (1, 1))
        self.assertTrue(self.es_s32_i in hs1)
        self.assertEqual(hs1.index(self.es_s32_i), 2)
        self.assertEqual(hs1.es_dim(self.es_s32_i), 4)
        self.assertEqual(hs1.bit_range(self.es_s32_i), (2, 3))
        self.assertTrue(self.es_s32_j in hs1)
        self.assertEqual(hs1.index(self.es_s32_j), 3)
        self.assertEqual(hs1.es_dim(self.es_s32_j), 4)
        self.assertEqual(hs1.bit_range(self.es_s32_j), (4, 5))

        expr += a_dag("x", 0) + a("y", 0)

        hs2 = HilbertSpace(expr, dim_boson=13)

        self.assertEqual(len(hs2), 6)
        self.assertEqual(hs2.total_n_bits, 14)
        self.assertEqual(hs2.dim, 10816)
        self.assertEqual(hs2.vec_size, 16384)
        self.assertTrue(hs2.is_sparse)
        self.assertTrue(self.es_f_dn in hs2)
        self.assertEqual(hs2.index(self.es_f_dn), 0)
        self.assertEqual(hs2.es_dim(self.es_f_dn), 2)
        self.assertEqual(hs2.bit_range(self.es_f_dn), (0, 0))
        self.assertTrue(self.es_f_up in hs2)
        self.assertEqual(hs2.index(self.es_f_up), 1)
        self.assertEqual(hs2.es_dim(self.es_f_up), 2)
        self.assertEqual(hs2.bit_range(self.es_f_up), (1, 1))
        self.assertTrue(self.es_b_x in hs2)
        self.assertEqual(hs2.index(self.es_b_x), 2)
        self.assertEqual(hs2.es_dim(self.es_b_x), 13)
        self.assertEqual(hs2.bit_range(self.es_b_x), (2, 5))
        self.assertTrue(self.es_b_y in hs2)
        self.assertEqual(hs2.index(self.es_b_y), 3)
        self.assertEqual(hs2.es_dim(self.es_b_y), 13)
        self.assertEqual(hs2.bit_range(self.es_b_y), (6, 9))
        self.assertTrue(self.es_s32_i in hs2)
        self.assertEqual(hs2.index(self.es_s32_i), 4)
        self.assertEqual(hs2.es_dim(self.es_s32_i), 4)
        self.assertEqual(hs2.bit_range(self.es_s32_i), (10, 11))
        self.assertTrue(self.es_s32_j in hs2)
        self.assertEqual(hs2.index(self.es_s32_j), 5)
        self.assertEqual(hs2.es_dim(self.es_s32_j), 4)
        self.assertEqual(hs2.bit_range(self.es_s32_j), (12, 13))

        # foreach()

        # Dense Hilbert space
        st1 = []
        foreach(hs1, lambda i: st1.append(i))
        self.assertEqual(st1, list(range(64)))

        # Sparse Hilbert space
        expr2 = 5.0 * n("up", 0) * n("dn", 0) + 2.0 * S_z("i", 0, spin=1) + \
            S_z("i", 0, spin=3 / 2)

        hs3 = HilbertSpace(expr2, dim_boson=1)
        self.assertEqual(hs3.dim, 2 * 2 * 3 * 4)
        self.assertTrue(hs3.is_sparse)

        st3 = []
        st3_ref = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,            # State |S_z=-3/2>
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  # State |S_z=-1/2>
            32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,  # State |S_z=1/2>
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59   # State |S_z=3/2>
        ]
        foreach(hs3, lambda i: st3.append(i))
        self.assertEqual(st3, st3_ref)

    def test_very_big_space(self):
        hs1 = HilbertSpace()
        for i in range(31):
            hs1.add(make_space_spin(3 / 2, "s", i))

        with self.assertRaisesRegex(
            RuntimeError,
            "Hilbert space size is not representable by a 64-bit integer "
            "\\(n_bits = 64\\)"
        ):
            hs1.add(make_space_spin(3 / 2, "s", 31))

        expr = ExpressionR(1.0)
        for i in range(31):
            expr *= S_p("s", i, spin=3 / 2)
        expr *= S_p("s", 31, spin=1 / 2)
        hs2 = HilbertSpace(expr)
        self.assertEqual(hs2.total_n_bits, 63)

        expr *= S_p("s", 32, spin=3 / 2)
        with self.assertRaisesRegex(
            RuntimeError,
            "Hilbert space size is not representable by a 64-bit integer "
            "\\(n_bits = 65\\)"
        ):
            HilbertSpace(expr)
