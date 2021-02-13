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

from pycommute.expression import (
    ExpressionR,
    FERMION, BOSON, SPIN,
    n, a, a_dag, S_p, S_m
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
        # Bosonic elementary spaces (4 bits)
        cls.es_b_x = make_space_boson(4, "x", 0)
        cls.es_b_y = make_space_boson(4, "y", 0)
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

        hs1 = HilbertSpace([
            self.es_s32_i, self.es_s32_j,
            self.es_s1_i, self.es_s1_j,
            self.es_s_i, self.es_s_j,
            self.es_b_x, self.es_b_y,
            self.es_f_dn, self.es_f_up
        ])
        self.assertEqual(len(hs1), 10)
        self.assertEqual(hs1.total_n_bits, 20)
        self.assertEqual(hs1.dim, 1048576)

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
        self.assertEqual(hs.dim, 16384)
        self.assertEqual(hs.algebra_bit_range(FERMION), (0, 1))
        self.assertEqual(hs.algebra_bit_range(BOSON), (2, 5))
        self.assertEqual(hs.algebra_bit_range(SPIN), (6, 13))
        self.assertTrue(self.es_f_dn in hs)
        self.assertEqual(hs.bit_range(self.es_f_dn), (0, 0))
        self.assertEqual(hs.basis_state_index(self.es_f_dn, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_f_dn, 1), 1)
        self.assertTrue(self.es_f_up in hs)
        self.assertEqual(hs.bit_range(self.es_f_up), (1, 1))
        self.assertEqual(hs.basis_state_index(self.es_f_up, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_f_up, 1), 2)
        self.assertTrue(self.es_b_x in hs)
        self.assertEqual(hs.bit_range(self.es_b_x), (2, 5))
        self.assertEqual(hs.basis_state_index(self.es_b_x, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_b_x, 1), 4)
        self.assertEqual(hs.basis_state_index(self.es_b_x, 5), 20)
        self.assertFalse(self.es_b_y in hs)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.bit_range(self.es_b_y)
        self.assertTrue(self.es_s_i in hs)
        self.assertEqual(hs.bit_range(self.es_s_i), (6, 6))
        self.assertEqual(hs.basis_state_index(self.es_s_i, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s_i, 1), 64)
        self.assertTrue(self.es_s_j in hs)
        self.assertEqual(hs.bit_range(self.es_s_j), (7, 7))
        self.assertFalse(self.es_s1_i in hs)
        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space not found$"):
            hs.bit_range(self.es_s1_i)
        self.assertTrue(self.es_s1_j in hs)
        self.assertEqual(hs.bit_range(self.es_s1_j), (8, 9))
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 1), 256)
        self.assertEqual(hs.basis_state_index(self.es_s1_j, 2), 512)
        self.assertTrue(self.es_s32_i in hs)
        self.assertEqual(hs.bit_range(self.es_s32_i), (10, 11))
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 1), 1024)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 2), 2048)
        self.assertEqual(hs.basis_state_index(self.es_s32_i, 3), 3072)
        self.assertTrue(self.es_s32_j in hs)
        self.assertEqual(hs.bit_range(self.es_s32_j), (12, 13))
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 0), 0)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 1), 4096)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 2), 8192)
        self.assertEqual(hs.basis_state_index(self.es_s32_j, 3), 12288)

    def test_add(self):
        hs = HilbertSpace([self.es_s32_i])

        def check_hs(es,
                     size,
                     total_n_bits,
                     b, e,
                     fermion, boson, spin):
            self.assertEqual(len(hs), size)
            self.assertEqual(hs.total_n_bits, total_n_bits)
            self.assertEqual(hs.dim, 1 << total_n_bits)
            self.assertTrue(es in hs)
            self.assertEqual(hs.bit_range(es), (b, e))

            if fermion is not None:
                self.assertEqual(hs.algebra_bit_range(FERMION), fermion)
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % FERMION
                ):
                    hs.algebra_bit_range(FERMION)

            if boson is not None:
                self.assertEqual(hs.algebra_bit_range(BOSON), boson)
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % BOSON
                ):
                    hs.algebra_bit_range(BOSON)

            if spin is not None:
                self.assertEqual(hs.algebra_bit_range(SPIN), spin)
            else:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^No elementary spaces with algebra ID %i$" % SPIN
                ):
                    hs.algebra_bit_range(SPIN)

        check_hs(self.es_s32_i, 1, 2, 0, 1, None, None, (0, 1))
        hs.add(self.es_s32_j)
        check_hs(self.es_s32_j, 2, 4, 2, 3, None, None, (0, 3))
        hs.add(self.es_s1_j)
        check_hs(self.es_s1_j, 3, 6, 0, 1, None, None, (0, 5))
        hs.add(self.es_s_i)
        check_hs(self.es_s_i, 4, 7, 0, 0, None, None, (0, 6))
        hs.add(self.es_s_j)
        check_hs(self.es_s_j, 5, 8, 1, 1, None, None, (0, 7))
        hs.add(self.es_b_x)
        check_hs(self.es_b_x, 6, 12, 0, 3, None, (0, 3), (4, 11))
        hs.add(self.es_f_dn)
        check_hs(self.es_f_dn, 7, 13, 0, 0, (0, 0), (1, 4), (5, 12))
        hs.add(self.es_f_up)
        check_hs(self.es_f_up, 8, 14, 1, 1, (0, 1), (2, 5), (6, 13))

        with self.assertRaisesRegex(RuntimeError,
                                    "^Elementary space already exists$"):
            hs.add(self.es_s_j)

        check_hs(self.es_f_dn, 8, 14, 0, 0, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_f_up, 8, 14, 1, 1, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_b_x, 8, 14, 2, 5, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s_i, 8, 14, 6, 6, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s_j, 8, 14, 7, 7, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s1_j, 8, 14, 8, 9, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s32_i, 8, 14, 10, 11, (0, 1), (2, 5), (6, 13))
        check_hs(self.es_s32_j, 8, 14, 12, 13, (0, 1), (2, 5), (6, 13))
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
        self.assertTrue(self.es_f_dn in hs1)
        self.assertEqual(hs1.bit_range(self.es_f_dn), (0, 0))
        self.assertTrue(self.es_f_up in hs1)
        self.assertEqual(hs1.bit_range(self.es_f_up), (1, 1))
        self.assertTrue(self.es_s32_i in hs1)
        self.assertEqual(hs1.bit_range(self.es_s32_i), (2, 3))
        self.assertTrue(self.es_s32_j in hs1)
        self.assertEqual(hs1.bit_range(self.es_s32_j), (4, 5))

        # foreach()
        count = 0

        def counter(i):
            nonlocal count
            count += i
        foreach(hs1, counter)
        self.assertEqual(count, 2016)

        expr += a_dag("x", 0) + a("y", 0)

        hs2 = HilbertSpace(expr, bits_per_boson=4)

        self.assertEqual(len(hs2), 6)
        self.assertEqual(hs2.total_n_bits, 14)
        self.assertEqual(hs2.dim, 16384)
        self.assertTrue(self.es_f_dn in hs2)
        self.assertEqual(hs2.bit_range(self.es_f_dn), (0, 0))
        self.assertTrue(self.es_f_up in hs2)
        self.assertEqual(hs2.bit_range(self.es_f_up), (1, 1))
        self.assertTrue(self.es_b_x in hs2)
        self.assertEqual(hs2.bit_range(self.es_b_x), (2, 5))
        self.assertTrue(self.es_b_y in hs2)
        self.assertEqual(hs2.bit_range(self.es_b_y), (6, 9))
        self.assertTrue(self.es_s32_i in hs2)
        self.assertEqual(hs2.bit_range(self.es_s32_i), (10, 11))
        self.assertTrue(self.es_s32_j in hs2)
        self.assertEqual(hs2.bit_range(self.es_s32_j), (12, 13))

    def test_very_big_space(self):
        hs1 = HilbertSpace()
        for i in range(32):
            hs1.add(make_space_spin(3 / 2, "s", i))

        with self.assertRaisesRegex(
            RuntimeError,
            "Hilbert space size is not representable by a 64-bit integer "
            "\\(n_bits = 66\\)"
        ):
            hs1.add(make_space_spin(3 / 2, "s", 32))

        expr = ExpressionR(1.0)
        for i in range(32):
            expr *= S_p("s", i, spin=3 / 2)
        hs2 = HilbertSpace(expr)
        self.assertEqual(hs2.total_n_bits, 64)

        expr *= S_p("s", 32, spin=3 / 2)
        with self.assertRaisesRegex(
            RuntimeError,
            "Hilbert space size is not representable by a 64-bit integer "
            "\\(n_bits = 66\\)"
        ):
            HilbertSpace(expr)
