#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from itertools import product
from pycommute.expression import Indices, LinearFunctionGen
from pycommute.expression import FERMION, GeneratorFermion, make_fermion
from pycommute.expression import BOSON, GeneratorBoson, make_boson
from pycommute.expression import SPIN, GeneratorSpin, make_spin
from pycommute.expression import SpinComponent, swap_with
from pycommute.expression import is_fermion, is_boson, is_spin

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

    def check_linear_function_0(self, lf, const_term):
        self.assertEqual(lf.const_term, const_term)
        self.assertEqual(len(lf.terms), 0)

    def check_linear_function_1(self, lf, const_term, coeff, gen):
        self.assertEqual(lf.const_term, const_term)
        self.assertEqual(len(lf.terms), 1)
        self.assertEqual(lf.terms[0], (gen, coeff))

    def check_generator_spin_swap_with(self, v, one_half = False):
        f = LinearFunctionGen()
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                c = v[j].swap_with(v[i], f)
                if one_half:
                    if j%3 == 1 and i == j - 1: # S_- S_+ = 1/2 - S_z
                        self.assertEqual(c, 0)
                        self.check_linear_function_1(f, 1/2, -1, v[i + 2])
                    elif j%3 == 2 and i == j - 2: # S_z S_+ = 1/2 S_+
                        self.assertEqual(c, 0)
                        self.check_linear_function_1(f, 0, 1/2, v[i])
                    elif j%3 == 2 and i == j - 1: # S_z S_- = -1/2 S_-
                        self.assertEqual(c, 0)
                        self.check_linear_function_1(f, 0, -1/2, v[i])
                    else:
                        self.assertEqual(c, 1)
                        self.check_linear_function_0(f, 0)
                else:
                    self.assertEqual(c, 1)
                    if j%3 == 1 and i == j - 1: # S_- S_+ = S_+ * S_- - 2*S_z
                        self.check_linear_function_1(f, 0, -2, v[i + 2])
                    elif j%3 == 2 and i == j - 2: # S_z S_+ = S_+ * S_z + S_+
                        self.check_linear_function_1(f, 0, 1, v[i])
                    elif j%3 == 2 and i == j - 1: # S_z S_- = S_- * S_z - S_-
                        self.check_linear_function_1(f, 0, -1, v[i])
                    else:
                        self.check_linear_function_0(f, 0)

    def check_generator_spin_simplify_prod(self, v, one_half = False):
        f = LinearFunctionGen()
        for i in range(len(v)):
            for j in range(i, len(v)):
                c = v[i].simplify_prod(v[j], f)
                if one_half:
                    if i%3 == 0 and j == i: # S_+ * S_+ = 0
                        self.assertTrue(c)
                        self.check_linear_function_0(f, 0)
                    elif i%3 == 1 and j == i: # S_- * S_- = 0
                        self.assertTrue(c)
                        self.check_linear_function_0(f, 0)
                    elif i%3 == 2 and j == i: # S_z * S_z = 1/4
                        self.assertTrue(c)
                        self.check_linear_function_0(f, 1/4)
                    elif i%3 == 0 and j == i + 1: # S_+ * S_- = 1/2 + S_z
                        self.assertTrue(c)
                        self.check_linear_function_1(f, 1/2, 1, v[i + 2])
                    elif i%3 == 0 and j == i + 2: # S_+ * S_z = -1/2 S_+
                        self.assertTrue(c)
                        self.check_linear_function_1(f, 0, -1/2, v[i])
                    elif i%3 == 1 and j == i + 1: # S_- * S_z = 1/2 S_-
                        self.assertTrue(c)
                        self.check_linear_function_1(f, 0, 1/2, v[i])
                    else:
                        self.assertFalse(c)
                else:
                    # No simplifications for higher spins
                    self.assertFalse(c)

    def check_conj(self, v, ref):
        f = LinearFunctionGen()
        for n, g in enumerate(v):
            g.conj(f)
            self.check_linear_function_1(f, 0, 1.0, v[ref[n]])

    # Test LinearFunctionGen
    def test_LinearFunctionGen(self):
        lf0 = LinearFunctionGen()
        self.assertEqual(lf0.const_term, 0)
        self.assertTrue(lf0.vanishing)
        self.assertEqual(lf0.terms, [])
        lf0.const_term = 4.0
        self.assertEqual(lf0.const_term, 4.0)
        self.assertFalse(lf0.vanishing)

        lf1 = LinearFunctionGen(5.0)
        self.assertEqual(lf1.const_term, 5.0)
        self.assertFalse(lf1.vanishing)
        self.assertEqual(lf1.terms, [])

        lf2 = LinearFunctionGen(3.0, [(make_fermion(True, "dn", 0), 2.0),
                                      (make_boson(False, "y"), -1.0)])
        self.assertEqual(lf2.const_term, 3.0)
        self.assertFalse(lf2.vanishing)
        self.assertEqual(lf2.terms, [(make_fermion(True, "dn", 0), 2.0),
                                     (make_boson(False, "y"), -1.0)])
        lf2.terms = [(make_spin(SpinComponent.PLUS, 1), 6.0)]
        self.assertEqual(lf2.terms, [(make_spin(SpinComponent.PLUS, 1), 6.0)])
        lf2.const_term = 0
        lf2.terms = []
        self.assertTrue(lf2.vanishing)

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
        cls.spin32_ops = [make_spin(3/2, SpinComponent.PLUS, 1),
                          make_spin(3/2, SpinComponent.MINUS, 1),
                          make_spin(3/2, SpinComponent.Z, 1),
                          make_spin(3/2, SpinComponent.PLUS, 2),
                          make_spin(3/2, SpinComponent.MINUS, 2),
                          make_spin(3/2, SpinComponent.Z, 2)]

    def test_fermion(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.fermion_ops)),
                             [Indices("dn", 0),
                              Indices("up", 0),
                              Indices("up", 0),
                              Indices("dn", 0)])

        lin_f = LinearFunctionGen()

        for op in self.fermion_ops:
            self.assertEqual(op.algebra_id, FERMION)
            self.assertFalse(op.reduce_power(3, lin_f))
            self.assertFalse(op.reduce_power(4, lin_f))

        for i, op in enumerate(self.fermion_ops):
            self.assertEqual(op.dagger, (i < 2))

        self.check_equality(self.fermion_ops)
        self.check_less_greater(self.fermion_ops)

        for op in self.fermion_ops:
            self.assertTrue(is_fermion(op))
            self.assertFalse(is_boson(op))
            self.assertFalse(is_spin(op))

        f = LinearFunctionGen()
        for i in range(len(self.fermion_ops)):
            for j in range(i + 1, len(self.fermion_ops)):
                c = self.fermion_ops[j].swap_with(self.fermion_ops[i], f)
                self.assertEqual(c, -1)
                self.check_linear_function_0(f, \
                  ((j == 2 and i == 1) or (j == 3 and i == 0)))

        self.check_conj(self.fermion_ops, [3, 2, 1, 0])

        self.assertListEqual(list(map(str, self.fermion_ops)),
                             ["C+(dn,0)", "C+(up,0)", "C(up,0)", "C(dn,0)"])

    def test_boson(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.boson_ops)),
                      [Indices("x"), Indices("y"), Indices("y"), Indices("x")])

        lin_f = LinearFunctionGen()

        for op in self.boson_ops:
            self.assertEqual(op.algebra_id, BOSON)
            self.assertFalse(op.reduce_power(3, lin_f))
            self.assertFalse(op.reduce_power(4, lin_f))

        for i, op in enumerate(self.boson_ops):
            self.assertEqual(op.dagger, (i < 2))

        self.check_equality(self.boson_ops)
        self.check_less_greater(self.boson_ops)

        for op in self.boson_ops:
            self.assertFalse(is_fermion(op))
            self.assertTrue(is_boson(op))
            self.assertFalse(is_spin(op))

        f = LinearFunctionGen()
        for i in range(len(self.boson_ops)):
            for j in range(i + 1, len(self.boson_ops)):
                c = self.boson_ops[j].swap_with(self.boson_ops[i], f)
                self.assertEqual(c, 1)
                self.check_linear_function_0(f, \
                  ((j == 2 and i == 1) or (j == 3 and i == 0)))

        self.check_conj(self.boson_ops, [3, 2, 1, 0])

        self.assertListEqual(list(map(str, self.boson_ops)),
                             ["A+(x)", "A+(y)", "A(y)", "A(x)"])

    def test_spin12(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin_ops)),
                             [Indices(1)]*3 + [Indices(2)]*3)

        lin_f = LinearFunctionGen()

        for op in self.spin_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 0.5)
            self.assertEqual(op.multiplicity, 2)
            if op.component == SpinComponent.Z:
                self.assertFalse(op.reduce_power(3, lin_f))
                self.assertFalse(op.reduce_power(4, lin_f))
            else:
                self.assertTrue(op.reduce_power(3, lin_f))
                self.assertTrue(lin_f.vanishing)
                self.assertTrue(op.reduce_power(4, lin_f))
                self.assertTrue(lin_f.vanishing)

        self.check_equality(self.spin_ops)
        self.check_less_greater(self.spin_ops)

        for op in self.spin_ops:
            self.assertFalse(is_fermion(op))
            self.assertFalse(is_boson(op))
            self.assertTrue(is_spin(op))

        self.check_generator_spin_swap_with(self.spin_ops, True)
        self.check_generator_spin_simplify_prod(self.spin_ops, True)

        self.check_conj(self.spin_ops, [1, 0, 2, 4, 3, 5])

        self.assertListEqual(list(map(str, self.spin_ops)),
                             ["S+(1)","S-(1)","Sz(1)","S+(2)","S-(2)","Sz(2)"])

    def test_spin1(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin1_ops)),
                             [Indices(1)]*3 + [Indices(2)]*3)

        lin_f = LinearFunctionGen()

        for op in self.spin1_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 1)
            self.assertEqual(op.multiplicity, 3)
            if op.component == SpinComponent.Z:
                self.assertFalse(op.reduce_power(3, lin_f))
                self.assertFalse(op.reduce_power(4, lin_f))
            else:
                self.assertTrue(op.reduce_power(3, lin_f))
                self.check_linear_function_0(lin_f, 0)
                self.assertTrue(op.reduce_power(4, lin_f))
                self.check_linear_function_0(lin_f, 0)

        self.check_equality(self.spin1_ops)
        self.check_less_greater(self.spin1_ops)

        for op in self.spin1_ops:
            self.assertFalse(is_fermion(op))
            self.assertFalse(is_boson(op))
            self.assertTrue(is_spin(op))

        self.check_generator_spin_swap_with(self.spin1_ops)
        self.check_generator_spin_simplify_prod(self.spin1_ops)

        self.check_conj(self.spin1_ops, [1, 0, 2, 4, 3, 5])

        self.assertListEqual(list(map(str, self.spin1_ops)),
            ["S1+(1)","S1-(1)","S1z(1)","S1+(2)","S1-(2)","S1z(2)"])

    def test_spin32(self):
        self.assertListEqual(list(map(lambda g: g.indices, self.spin32_ops)),
                             [Indices(1)]*3 + [Indices(2)]*3)

        lin_f = LinearFunctionGen()

        for op in self.spin32_ops:
            self.assertEqual(op.algebra_id, SPIN)
            self.assertEqual(op.spin, 3/2)
            self.assertEqual(op.multiplicity, 4)
            if op.component == SpinComponent.Z:
                self.assertFalse(op.reduce_power(3, lin_f))
                self.assertFalse(op.reduce_power(4, lin_f))
            else:
                self.assertFalse(op.reduce_power(3, lin_f))
                self.assertTrue(op.reduce_power(4, lin_f))
                self.check_linear_function_0(lin_f, 0)

        self.check_equality(self.spin32_ops)
        self.check_less_greater(self.spin32_ops)

        for op in self.spin32_ops:
            self.assertFalse(is_fermion(op))
            self.assertFalse(is_boson(op))
            self.assertTrue(is_spin(op))

        self.check_generator_spin_swap_with(self.spin32_ops)
        self.check_generator_spin_simplify_prod(self.spin32_ops)

        self.check_conj(self.spin32_ops, [1, 0, 2, 4, 3, 5])

        self.assertListEqual(list(map(str, self.spin32_ops)),
            ["S3/2+(1)","S3/2-(1)","S3/2z(1)","S3/2+(2)","S3/2-(2)","S3/2z(2)"])

    def test_different_algebras(self):
        all_ops = sum([self.fermion_ops,
                       self.boson_ops,
                       self.spin_ops,
                       self.spin1_ops,
                       self.spin32_ops], [])

        self.check_equality(all_ops)
        self.check_less_greater(all_ops)

        f = LinearFunctionGen()
        for i in range(len(all_ops)):
            for j in range(i + 1, len(all_ops)):
                c = swap_with(all_ops[j], all_ops[i], f)
                if all_ops[j].algebra_id != all_ops[i].algebra_id:
                    self.assertEqual(c, 1)
                    self.check_linear_function_0(f, 0)
