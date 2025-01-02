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

from itertools import product
from copy import copy, deepcopy
from pycommute.expression import make_fermion, make_boson, make_spin
from pycommute.expression import SpinComponent
from pycommute.expression import Monomial


class TestMonomial(TestCase):

    # Check that elements of `v` are pairwise distinct
    def check_equality(self, v):
        for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
            self.assertEqual((x == y), (i == j))
            self.assertEqual((x != y), (i != j))

    # Check that elements of `v` are ordered
    def check_less_greater(self, v):
        for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
            self.assertEqual((x < y), (i < j))
            self.assertEqual((x > y), (i > j))

    @classmethod
    def setUpClass(cls):
        cls.Cdag_dn = make_fermion(True, "dn", 0)
        cls.A_y = make_boson(False, "y", 0)
        cls.Sp_i = make_spin(SpinComponent.PLUS, "i", 0)
        cls.S1z_j = make_spin(1, SpinComponent.Z, "j", 0)

        cls.basis_gens = [cls.Cdag_dn, cls.A_y, cls.Sp_i, cls.S1z_j]
        cls.monomials = []

    def test_init(self):
        self.monomials.append(Monomial())
        self.assertEqual(len(self.monomials[-1]), 0)

        for order in range(1, 5):
            for gens in product(*((self.basis_gens,) * order)):
                self.monomials.append(Monomial([*gens]))
            self.assertEqual(len(self.monomials[-1]), order)
            self.assertEqual(self.monomials[-1],
                             Monomial((self.S1z_j,) * order))

    def test_copy(self):
        m = Monomial([self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        mc = copy(m)
        self.assertEqual(mc, m)
        self.assertNotEqual(id(mc), id(m))
        mdc = deepcopy(m)
        self.assertEqual(mdc, m)
        self.assertNotEqual(id(mdc), id(m))

    def test_equality_ordering(self):
        self.check_equality(self.monomials)
        self.check_less_greater(self.monomials)

    def test_ordering(self):
        self.assertTrue(Monomial().is_ordered)
        self.assertTrue(Monomial([self.S1z_j] * 4).is_ordered)
        m121 = Monomial([self.S1z_j, self.A_y, self.S1z_j, self.A_y])
        self.assertFalse(m121.is_ordered)
        m22 = Monomial([self.A_y, self.A_y, self.S1z_j, self.S1z_j])
        self.assertTrue(m22.is_ordered)

    def test_get_item(self):
        m4 = Monomial([self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        self.assertEqual(len(m4), 4)
        self.assertEqual(m4[0], self.Cdag_dn)
        self.assertEqual(m4[1], self.A_y)
        self.assertEqual(m4[2], self.Sp_i)
        self.assertEqual(m4[3], self.S1z_j)
        with self.assertRaises(IndexError):
            m4[4]

        l6 = [self.Cdag_dn, self.A_y, self.Sp_i,
              self.S1z_j, self.Cdag_dn, self.A_y]
        m6 = Monomial(l6)
        self.assertEqual(m6[1:5:2], Monomial(l6[1:5:2]))

    def test_contains(self):
        m3 = Monomial([self.Cdag_dn, self.A_y, self.S1z_j])
        self.assertTrue(self.A_y in m3)
        self.assertTrue(self.Sp_i not in m3)

    def test_str(self):
        self.assertEqual(str(Monomial()), "")
        m1 = Monomial([self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        self.assertEqual(str(m1), "C+(dn,0)A(y,0)S+(i,0)S1z(j,0)")
        m2 = Monomial([self.Cdag_dn, self.A_y, self.A_y, self.S1z_j])
        self.assertEqual(str(m2), "C+(dn,0)[A(y,0)]^2S1z(j,0)")
        m3 = Monomial([self.Cdag_dn, self.Cdag_dn, self.S1z_j, self.S1z_j])
        self.assertEqual(str(m3), "[C+(dn,0)]^2[S1z(j,0)]^2")

    def test_swap_generators(self):
        m = Monomial([self.Sp_i, self.S1z_j, self.Cdag_dn, self.A_y])
        m.swap_generators(1, 3)
        mnew = Monomial([self.Sp_i, self.A_y, self.Cdag_dn, self.S1z_j])
        self.assertEqual(m, mnew)

    def test_iterators(self):
        m0 = Monomial()
        self.assertEqual([g for g in m0], [])
        self.assertEqual([g for g in reversed(m0)], [])

        m4 = Monomial([self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        self.assertEqual([g for g in m4],
                         [self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        self.assertEqual([g for g in reversed(m4)],
                         [self.S1z_j, self.Sp_i, self.A_y, self.Cdag_dn])

    def test_concatenation(self):
        m0 = Monomial()
        m1 = Monomial([self.Cdag_dn, self.A_y])
        m2 = Monomial([self.Sp_i, self.S1z_j])
        m3 = Monomial([self.Cdag_dn, self.A_y, self.Sp_i, self.S1z_j])
        m4 = Monomial([self.Sp_i, self.S1z_j, self.Cdag_dn, self.A_y])
        m5 = Monomial([self.Cdag_dn, self.A_y, self.Sp_i,
                       self.S1z_j, self.Cdag_dn, self.A_y])
        self.assertEqual(m1 * m0, m1)
        self.assertEqual(m0 * m1, m1)
        self.assertEqual(m1 * m2, m3)
        self.assertEqual(m2 * m1, m4)
        self.assertEqual(m1 * m2 * m1, m5)

        self.assertEqual(m2 * self.Cdag_dn,
                         Monomial([self.Sp_i, self.S1z_j, self.Cdag_dn]))
        self.assertEqual(self.Cdag_dn * m2,
                         Monomial([self.Cdag_dn, self.Sp_i, self.S1z_j]))
