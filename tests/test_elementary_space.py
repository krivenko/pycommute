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
from pycommute.loperator import (
    make_space_fermion, make_space_boson, make_space_spin
)
from pycommute.expression import (FERMION, BOSON, SPIN)


class TestElementarySpace(TestCase):

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
        # Fermionic elementary spaces
        cls.fermion_es = [make_space_fermion("dn", 0),
                          make_space_fermion("up", 0)]
        # Bosonic elementary spaces (4 bits)
        cls.boson_es = [make_space_boson(4, "x", 0),
                        make_space_boson(4, "y", 0)]
        # Spin-1/2 algebra elementary spaces
        cls.spin_es = [make_space_spin(0.5, "i", 0),
                       make_space_spin(0.5, "j", 0)]
        # Spin-1 algebra elementary spaces
        cls.spin1_es = [make_space_spin(1.0, "i", 0),
                        make_space_spin(1.0, "j", 0)]
        # Spin-3/2 algebra elementary spaces
        cls.spin32_es = [make_space_spin(3 / 2, "i", 0),
                         make_space_spin(3 / 2, "j", 0)]

    def test_fermion(self):
        for es in self.fermion_es:
            self.assertEqual(es.algebra_id, FERMION)
            self.assertEqual(es.n_bits, 1)
        self.check_equality(self.fermion_es)
        self.check_less_greater(self.fermion_es)

    def test_boson(self):
        for es in self.boson_es:
            self.assertEqual(es.algebra_id, BOSON)
            self.assertEqual(es.n_bits, 4)
        self.check_equality(self.boson_es)
        self.check_less_greater(self.boson_es)

    def test_spin12(self):
        for es in self.spin_es:
            self.assertEqual(es.algebra_id, SPIN)
            self.assertEqual(es.n_bits, 1)
        self.check_equality(self.spin_es)
        self.check_less_greater(self.spin_es)

    def test_spin1(self):
        for es in self.spin1_es:
            self.assertEqual(es.algebra_id, SPIN)
            self.assertEqual(es.n_bits, 2)
        self.check_equality(self.spin1_es)
        self.check_less_greater(self.spin1_es)

    def test_spin32(self):
        for es in self.spin32_es:
            self.assertEqual(es.algebra_id, SPIN)
            self.assertEqual(es.n_bits, 2)
        self.check_equality(self.spin32_es)
        self.check_less_greater(self.spin32_es)

    def test_different_elementary_spaces(self):
        all_es = sum([self.fermion_es,
                      self.boson_es,
                      self.spin_es,
                      self.spin1_es,
                      self.spin32_es], [])
        self.check_equality(all_es)
        self.check_less_greater(all_es)
