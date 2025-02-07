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
from pycommute.expression import Indices
from pycommute.expression import FERMION, GeneratorFermion, make_fermion
from pycommute.expression import BOSON, GeneratorBoson, make_boson
from pycommute.expression import SPIN, GeneratorSpin, make_spin
from pycommute.expression import SpinComponent


# All other Generator* tests
class TestGenerator(TestCase):

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

    # Test constructors
    def test_init(self):
        f1 = GeneratorFermion(True, Indices(0, "up"))
        f2 = GeneratorFermion(True, Indices(0, "up"))
        f3 = GeneratorFermion(False, Indices(0, "up"))
        f4 = GeneratorFermion(False, Indices("dn", 0))
        self.assertEqual(f1, f2)
        self.assertNotEqual(f1, f4)
        self.assertNotEqual(f3, f4)

        b1 = GeneratorBoson(True, Indices(0, "up"))
        b2 = GeneratorBoson(True, Indices(0, "up"))
        b3 = GeneratorBoson(False, Indices(0, "up"))
        b4 = GeneratorBoson(False, Indices("dn", 0))
        self.assertEqual(b1, b2)
        self.assertNotEqual(b1, b4)
        self.assertNotEqual(b3, b4)

        s1 = GeneratorSpin(0.5, SpinComponent.PLUS, Indices("xxx"))
        s2 = GeneratorSpin(SpinComponent.PLUS, Indices("xxx"))
        s3 = GeneratorSpin(1.5, SpinComponent.PLUS, Indices("xxx"))
        s4 = GeneratorSpin(1.5, SpinComponent.MINUS, Indices("xxx"))
        s5 = GeneratorSpin(1.5, SpinComponent.MINUS, Indices("yyy"))
        self.assertEqual(s1, s2)
        self.assertNotEqual(s2, s3)
        self.assertNotEqual(s3, s4)
        self.assertNotEqual(s4, s5)

    @classmethod
    def setUpClass(cls):
        # Fermionic generators
        cls.fermion_ops = [make_fermion(True, "dn", 0),
                           make_fermion(True, "up", 0),
                           make_fermion(False, "up", 0),
                           make_fermion(False, "dn", 0)]
        # Bosonic generators
        cls.boson_ops = [make_boson(True, "x"),
                         make_boson(True, "y"),
                         make_boson(False, "y"),
                         make_boson(False, "x")]
        # Spin-1/2 algebra generators
        cls.spin_ops = [make_spin(SpinComponent.PLUS, 1),
                        make_spin(SpinComponent.MINUS, 1),
                        make_spin(SpinComponent.Z, 1),
                        make_spin(SpinComponent.PLUS, 2),
                        make_spin(SpinComponent.MINUS, 2),
                        make_spin(SpinComponent.Z, 2)]
        # Spin-1 algebra generators
        cls.spin1_ops = [make_spin(1, SpinComponent.PLUS, 1),
                         make_spin(1, SpinComponent.MINUS, 1),
                         make_spin(1, SpinComponent.Z, 1),
                         make_spin(1, SpinComponent.PLUS, 2),
                         make_spin(1, SpinComponent.MINUS, 2),
                         make_spin(1, SpinComponent.Z, 2)]
        # Spin-3/2 algebra generators
        cls.spin32_ops = [make_spin(3 / 2, SpinComponent.PLUS, 1),
                          make_spin(3 / 2, SpinComponent.MINUS, 1),
                          make_spin(3 / 2, SpinComponent.Z, 1),
                          make_spin(3 / 2, SpinComponent.PLUS, 2),
                          make_spin(3 / 2, SpinComponent.MINUS, 2),
                          make_spin(3 / 2, SpinComponent.Z, 2)]

    def test_fermion(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.fermion_ops)),
                             [Indices("dn", 0),
                              Indices("up", 0),
                              Indices("up", 0),
                              Indices("dn", 0)])

        for op in self.fermion_ops:
            self.assertEqual(op.algebra_id, FERMION)

        for i, op in enumerate(self.fermion_ops):
            self.assertEqual(op.dagger, (i < 2))

        self.check_equality(self.fermion_ops)
        self.check_less_greater(self.fermion_ops)

        for op in self.fermion_ops:
            self.assertIsInstance(op, GeneratorFermion)
            self.assertNotIsInstance(op, GeneratorBoson)
            self.assertNotIsInstance(op, GeneratorSpin)

        self.assertListEqual(list(map(str, self.fermion_ops)),
                             ["C+(dn,0)", "C+(up,0)", "C(up,0)", "C(dn,0)"])

    def test_boson(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.boson_ops)),
                             [Indices("x"),
                              Indices("y"),
                              Indices("y"),
                              Indices("x")])

        for op in self.boson_ops:
            self.assertEqual(op.algebra_id, BOSON)

        for i, op in enumerate(self.boson_ops):
            self.assertEqual(op.dagger, (i < 2))

        self.check_equality(self.boson_ops)
        self.check_less_greater(self.boson_ops)

        for op in self.boson_ops:
            self.assertNotIsInstance(op, GeneratorFermion)
            self.assertIsInstance(op, GeneratorBoson)
            self.assertNotIsInstance(op, GeneratorSpin)

        self.assertListEqual(list(map(str, self.boson_ops)),
                             ["A+(x)", "A+(y)", "A(y)", "A(x)"])

    def test_spin12(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin_ops)),
                             [Indices(1)] * 3 + [Indices(2)] * 3)

        for op in self.spin_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 0.5)
            self.assertEqual(op.multiplicity, 2)

        self.check_equality(self.spin_ops)
        self.check_less_greater(self.spin_ops)

        for op in self.spin_ops:
            self.assertNotIsInstance(op, GeneratorFermion)
            self.assertNotIsInstance(op, GeneratorBoson)
            self.assertIsInstance(op, GeneratorSpin)

        self.assertListEqual(list(map(str, self.spin_ops)),
                             ["S+(1)",
                              "S-(1)",
                              "Sz(1)",
                              "S+(2)",
                              "S-(2)",
                              "Sz(2)"])

    def test_spin1(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin1_ops)),
                             [Indices(1)] * 3 + [Indices(2)] * 3)

        for op in self.spin1_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 1)
            self.assertEqual(op.multiplicity, 3)

        self.check_equality(self.spin1_ops)
        self.check_less_greater(self.spin1_ops)

        for op in self.spin1_ops:
            self.assertNotIsInstance(op, GeneratorFermion)
            self.assertNotIsInstance(op, GeneratorBoson)
            self.assertIsInstance(op, GeneratorSpin)

        self.assertListEqual(list(map(str, self.spin1_ops)),
                             ["S1+(1)",
                              "S1-(1)",
                              "S1z(1)",
                              "S1+(2)",
                              "S1-(2)",
                              "S1z(2)"])

    def test_spin32(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin32_ops)),
                             [Indices(1)] * 3 + [Indices(2)] * 3)

        for op in self.spin32_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 3 / 2)
            self.assertEqual(op.multiplicity, 4)

        self.check_equality(self.spin32_ops)
        self.check_less_greater(self.spin32_ops)

        for op in self.spin32_ops:
            self.assertNotIsInstance(op, GeneratorFermion)
            self.assertNotIsInstance(op, GeneratorBoson)
            self.assertIsInstance(op, GeneratorSpin)

        self.assertListEqual(list(map(str, self.spin32_ops)),
                             ["S3/2+(1)",
                              "S3/2-(1)",
                              "S3/2z(1)",
                              "S3/2+(2)",
                              "S3/2-(2)",
                              "S3/2z(2)"])

    def test_different_algebras(self):
        all_ops = sum([self.fermion_ops,
                       self.boson_ops,
                       self.spin_ops,
                       self.spin1_ops,
                       self.spin32_ops], [])

        self.check_equality(all_ops)
        self.check_less_greater(all_ops)
