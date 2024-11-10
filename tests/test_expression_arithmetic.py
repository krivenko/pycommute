#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2024 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from pycommute.expression import (
    ExpressionR, ExpressionC,
    c, c_dag, a, a_dag,
    make_complex
)


class TestExpressionArithemtic(TestCase):

    def test_unary_minus(self):
        expr_r = c_dag(1, "up")
        self.assertEqual(str(-expr_r), "-1*C+(1,up)")

    def test_inplace_addition(self):
        # Real
        expr_r = c_dag(1, "up")
        expr_r += c(2, "dn")
        self.assertEqual(str(expr_r), "1*C+(1,up) + 1*C(2,dn)")
        expr_r += ExpressionR()
        self.assertEqual(str(expr_r), "1*C+(1,up) + 1*C(2,dn)")
        expr_r += -c_dag(1, "up")
        self.assertEqual(str(expr_r), "1*C(2,dn)")
        # Complex
        expr_c = make_complex(c_dag(1, "up"))
        expr_c += c(2, "dn")
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        expr_c += ExpressionC()
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        expr_c += -c_dag(1, "up")
        self.assertEqual(str(expr_c), "(1,0)*C(2,dn)")

    def test_inplace_addition_const(self):
        # Real
        expr_r = c_dag(1, "up")
        expr_r += 4.0
        self.assertEqual(str(expr_r), "4 + 1*C+(1,up)")
        expr_r += 0.0
        self.assertEqual(str(expr_r), "4 + 1*C+(1,up)")
        expr_r += -4.0
        self.assertEqual(str(expr_r), "1*C+(1,up)")
        # Real and complex
        expr_c = make_complex(c_dag(1, "up"))
        expr_c += 4.0
        self.assertEqual(str(expr_c), "(4,0) + (1,0)*C+(1,up)")
        expr_c += 0.0
        self.assertEqual(str(expr_c), "(4,0) + (1,0)*C+(1,up)")
        expr_c += -4.0
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up)")

    def test_inplace_subtraction(self):
        # Real
        expr_r = c_dag(1, "up")
        expr_r -= c(2, "dn")
        self.assertEqual(str(expr_r), "1*C+(1,up) + -1*C(2,dn)")
        expr_r -= ExpressionR()
        self.assertEqual(str(expr_r), "1*C+(1,up) + -1*C(2,dn)")
        expr_r -= c_dag(1, "up")
        self.assertEqual(str(expr_r), "-1*C(2,dn)")
        # Complex
        expr_c = make_complex(c_dag(1, "up"))
        expr_c -= c(2, "dn")
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up) + (-1,0)*C(2,dn)")
        expr_c -= ExpressionC()
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up) + (-1,0)*C(2,dn)")
        expr_c -= c_dag(1, "up")
        self.assertEqual(str(expr_c), "(-1,0)*C(2,dn)")

    def test_inplace_subtraction_const(self):
        # Real
        expr_r = c_dag(1, "up")
        expr_r -= 4.0
        self.assertEqual(str(expr_r), "-4 + 1*C+(1,up)")
        expr_r -= 0.0
        self.assertEqual(str(expr_r), "-4 + 1*C+(1,up)")
        expr_r -= -4.0
        self.assertEqual(str(expr_r), "1*C+(1,up)")
        # Real and complex
        expr_c = make_complex(c_dag(1, "up"))
        expr_c -= 4.0
        self.assertEqual(str(expr_c), "(-4,0) + (1,0)*C+(1,up)")
        expr_c -= 0.0
        self.assertEqual(str(expr_c), "(-4,0) + (1,0)*C+(1,up)")
        expr_c -= -4.0
        self.assertEqual(str(expr_c), "(1,0)*C+(1,up)")

    def test_inplace_multiplication(self):
        # Real
        expr_r = c(2, "dn")
        expr_r *= c_dag(1, "up")
        self.assertEqual(str(expr_r), "-1*C+(1,up)C(2,dn)")
        expr_r *= a(0, "x")
        self.assertEqual(str(expr_r), "-1*C+(1,up)C(2,dn)A(0,x)")
        expr_r *= a_dag(0, "y")
        self.assertEqual(str(expr_r), "-1*C+(1,up)C(2,dn)A+(0,y)A(0,x)")
        expr_r *= ExpressionR(2)
        self.assertEqual(str(expr_r), "-2*C+(1,up)C(2,dn)A+(0,y)A(0,x)")
        expr_r *= ExpressionR()
        self.assertEqual(str(expr_r), "0")
        # Complex
        expr_c = make_complex(c(2, "dn"))
        expr_c *= make_complex(c_dag(1, "up"))
        self.assertEqual(str(expr_c), "(-1,0)*C+(1,up)C(2,dn)")
        expr_c *= make_complex(a(0, "x"))
        self.assertEqual(str(expr_c), "(-1,0)*C+(1,up)C(2,dn)A(0,x)")
        expr_c *= make_complex(a_dag(0, "y"))
        self.assertEqual(str(expr_c), "(-1,0)*C+(1,up)C(2,dn)A+(0,y)A(0,x)")
        expr_c *= ExpressionC(2)
        self.assertEqual(str(expr_c), "(-2,0)*C+(1,up)C(2,dn)A+(0,y)A(0,x)")
        expr_c *= ExpressionC()
        self.assertEqual(str(expr_c), "(0,0)")

    def test_inplace_multiplication_const(self):
        # Real
        expr_r = c_dag(1, "up")
        expr_r *= 4.0
        self.assertEqual(str(expr_r), "4*C+(1,up)")
        expr_r *= 0.0
        self.assertEqual(str(expr_r), "0")
        # Real and complex
        expr_c = make_complex(c_dag(1, "up"))
        expr_c *= 4.0
        self.assertEqual(str(expr_c), "(4,0)*C+(1,up)")
        expr_c *= 0.0
        self.assertEqual(str(expr_c), "(0,0)")

    def test_addition(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r + c(2, "dn"), ExpressionR)
        self.assertIsInstance(c(2, "dn") + expr_r, ExpressionR)
        self.assertEqual(str(ExpressionR() + ExpressionR()), "0")
        self.assertEqual(str(expr_r + ExpressionR()), "1*C+(1,up)")
        self.assertEqual(str(ExpressionR() + expr_r), "1*C+(1,up)")
        self.assertEqual(str(expr_r + c(2, "dn")), "1*C+(1,up) + 1*C(2,dn)")
        self.assertEqual(str(c(2, "dn") + expr_r), "1*C+(1,up) + 1*C(2,dn)")
        expr_r += c(2, "dn")

        self.assertEqual(str(expr_r + ExpressionR()), "1*C+(1,up) + 1*C(2,dn)")
        self.assertEqual(str(ExpressionR() + expr_r), "1*C+(1,up) + 1*C(2,dn)")
        self.assertEqual(str(expr_r + a(0, "x")),
                         "1*C+(1,up) + 1*C(2,dn) + 1*A(0,x)")
        self.assertEqual(str(a(0, "x") + expr_r),
                         "1*C+(1,up) + 1*C(2,dn) + 1*A(0,x)")
        self.assertEqual(str(expr_r + (-c_dag(1, "up"))), "1*C(2,dn)")
        self.assertEqual(str((-c_dag(1, "up")) + expr_r), "1*C(2,dn)")
        self.assertEqual(
            str((c_dag(1, "up") + c(2, "dn")) + (c(2, "dn") + 2.0)),
            "2 + 1*C+(1,up) + 2*C(2,dn)"
        )

        # Real and complex
        expr1 = make_complex(c_dag(1, "up"))
        expr2 = c(2, "dn")
        self.assertIsInstance(expr1 + expr2, ExpressionC)
        self.assertIsInstance(expr2 + expr1, ExpressionC)
        self.assertEqual(str(ExpressionC() + ExpressionR()), "(0,0)")
        self.assertEqual(str(ExpressionR() + ExpressionC()), "(0,0)")
        self.assertEqual(str(expr1 + ExpressionR()), "(1,0)*C+(1,up)")
        self.assertEqual(str(ExpressionR() + expr1), "(1,0)*C+(1,up)")
        self.assertEqual(str(expr2 + ExpressionC()), "(1,0)*C(2,dn)")
        self.assertEqual(str(ExpressionC() + expr2), "(1,0)*C(2,dn)")
        self.assertEqual(str(expr1 + expr2), "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        self.assertEqual(str(expr2 + expr1), "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        expr1 += expr2
        self.assertEqual(str(expr1 + ExpressionR()),
                         "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        self.assertEqual(str(ExpressionR() + expr1),
                         "(1,0)*C+(1,up) + (1,0)*C(2,dn)")
        self.assertEqual(str(expr1 + a(0, "x")),
                         "(1,0)*C+(1,up) + (1,0)*C(2,dn) + (1,0)*A(0,x)")
        self.assertEqual(str(a(0, "x") + expr1),
                         "(1,0)*C+(1,up) + (1,0)*C(2,dn) + (1,0)*A(0,x)")
        self.assertEqual(str(expr1 + (-make_complex(c_dag(1, "up")))),
                         "(1,0)*C(2,dn)")
        self.assertEqual(str((-make_complex(c_dag(1, "up"))) + expr1),
                         "(1,0)*C(2,dn)")
        self.assertEqual(
            str(make_complex(c_dag(1, "up") + c(2, "dn")) + (c(2, "dn") + 2.0)),
            "(2,0) + (1,0)*C+(1,up) + (2,0)*C(2,dn)"
        )

    def test_addition_const(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r + 2, ExpressionR)
        self.assertIsInstance(2 + expr_r, ExpressionR)
        self.assertEqual(str(expr_r + 0), "1*C+(1,up)")
        self.assertEqual(str(0 + expr_r), "1*C+(1,up)")
        self.assertEqual(str(expr_r + 2), "2 + 1*C+(1,up)")
        self.assertEqual(str(2 + expr_r), "2 + 1*C+(1,up)")
        expr_r += 2.0
        self.assertEqual(str(expr_r + 0), "2 + 1*C+(1,up)")
        self.assertEqual(str(0 + expr_r), "2 + 1*C+(1,up)")
        self.assertEqual(str(expr_r + 2), "4 + 1*C+(1,up)")
        self.assertEqual(str(2 + expr_r), "4 + 1*C+(1,up)")
        self.assertEqual(str(expr_r + (-2)), "1*C+(1,up)")
        self.assertEqual(str((-2) + expr_r), "1*C+(1,up)")
        # Complex and real
        expr_c = make_complex(c_dag(1, "up"))
        self.assertIsInstance(expr_c + 2.0, ExpressionC)
        self.assertIsInstance(2.0 + expr_c, ExpressionC)
        self.assertEqual(str(expr_c + 0.0), "(1,0)*C+(1,up)")
        self.assertEqual(str(0.0 + expr_c), "(1,0)*C+(1,up)")
        self.assertEqual(str(expr_c + 2.0), "(2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0 + expr_c), "(2,0) + (1,0)*C+(1,up)")
        expr_c += 2.0
        self.assertEqual(str(expr_c + 0.0), "(2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(0.0 + expr_c), "(2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(expr_c + 2.0), "(4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0 + expr_c), "(4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(expr_c + (-2.0)), "(1,0)*C+(1,up)")
        self.assertEqual(str((-2.0) + expr_c), "(1,0)*C+(1,up)")
        # Real and complex
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r + 2.0j, ExpressionC)
        self.assertIsInstance(2.0j + expr_r, ExpressionC)
        self.assertEqual(str(expr_r + 0.0j), "(1,0)*C+(1,up)")
        self.assertEqual(str(0.0j + expr_r), "(1,0)*C+(1,up)")
        self.assertEqual(str(expr_r + 2.0j), "(0,2) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0j + expr_r), "(0,2) + (1,0)*C+(1,up)")
        expr_r += 2.0
        self.assertEqual(str(expr_r + 0.0j), "(2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(0.0j + expr_r), "(2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(expr_r + 2 + 0j), "(4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2 + 0j + expr_r), "(4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(expr_r + (-2 + 0j)), "(1,0)*C+(1,up)")
        self.assertEqual(str((-2 + 0j) + expr_r), "(1,0)*C+(1,up)")

    def test_subtraction(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r - c(2, "dn"), ExpressionR)
        self.assertIsInstance(c(2, "dn") - expr_r, ExpressionR)
        self.assertEqual(str(ExpressionR() - ExpressionR()), "0")
        self.assertEqual(str(expr_r - ExpressionR()), "1*C+(1,up)")
        self.assertEqual(str(ExpressionR() - expr_r), "-1*C+(1,up)")
        self.assertEqual(str(expr_r - c(2, "dn")), "1*C+(1,up) + -1*C(2,dn)")
        self.assertEqual(str(c(2, "dn") - expr_r), "-1*C+(1,up) + 1*C(2,dn)")
        expr_r -= c(2, "dn")
        self.assertEqual(str(expr_r + ExpressionR()), "1*C+(1,up) + -1*C(2,dn)")
        self.assertEqual(str(ExpressionR() - expr_r), "-1*C+(1,up) + 1*C(2,dn)")
        self.assertEqual(str(expr_r - a(0, "x")),
                         "1*C+(1,up) + -1*C(2,dn) + -1*A(0,x)")
        self.assertEqual(str(a(0, "x") - expr_r),
                         "-1*C+(1,up) + 1*C(2,dn) + 1*A(0,x)")
        self.assertEqual(str(expr_r - c_dag(1, "up")), "-1*C(2,dn)")
        self.assertEqual(str(c_dag(1, "up") - expr_r), "1*C(2,dn)")
        self.assertEqual(
            str((c_dag(1, "up") + c(2, "dn")) - (c(2, "dn") + 2.0)),
            "-2 + 1*C+(1,up)"
        )
        # Real and complex
        expr1 = make_complex(c_dag(1, "up"))
        expr2 = c(2, "dn")
        self.assertIsInstance(expr1 - expr2, ExpressionC)
        self.assertIsInstance(expr2 - expr1, ExpressionC)
        self.assertEqual(str(ExpressionC() - ExpressionR()), "(0,0)")
        self.assertEqual(str(ExpressionR() - ExpressionC()), "(0,0)")
        self.assertEqual(str(expr1 - ExpressionR()), "(1,0)*C+(1,up)")
        self.assertEqual(str(ExpressionR() - expr1), "(-1,-0)*C+(1,up)")
        self.assertEqual(str(expr2 - ExpressionC()), "(1,-0)*C(2,dn)")
        self.assertEqual(str(ExpressionC() - expr2), "(-1,0)*C(2,dn)")
        self.assertEqual(str(expr1 - expr2),
                         "(1,0)*C+(1,up) + (-1,0)*C(2,dn)")
        self.assertEqual(str(expr2 - expr1),
                         "(-1,-0)*C+(1,up) + (1,-0)*C(2,dn)")
        expr1 -= expr2
        self.assertEqual(str(expr1 - ExpressionR()),
                         "(1,0)*C+(1,up) + (-1,0)*C(2,dn)")
        self.assertEqual(str(ExpressionR() - expr1),
                         "(-1,-0)*C+(1,up) + (1,-0)*C(2,dn)")
        self.assertEqual(str(expr1 - a(0, "x")),
                         "(1,0)*C+(1,up) + (-1,0)*C(2,dn) + (-1,0)*A(0,x)")
        self.assertEqual(str(a(0, "x") - expr1),
                         "(-1,-0)*C+(1,up) + (1,-0)*C(2,dn) + (1,-0)*A(0,x)")
        self.assertEqual(str(expr1 - make_complex(c_dag(1, "up"))),
                         "(-1,0)*C(2,dn)")
        self.assertEqual(str(make_complex(c_dag(1, "up")) - expr1),
                         "(1,0)*C(2,dn)")
        self.assertEqual(
            str(make_complex(c_dag(1, "up") + c(2, "dn")) - (c(2, "dn") + 2.0)),
            "(-2,0) + (1,0)*C+(1,up)"
        )

    def test_subtraction_const(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r - 2, ExpressionR)
        self.assertIsInstance(2 - expr_r, ExpressionR)
        self.assertEqual(str(expr_r - 0), "1*C+(1,up)")
        self.assertEqual(str(0 - expr_r), "-1*C+(1,up)")
        self.assertEqual(str(expr_r - 2), "-2 + 1*C+(1,up)")
        self.assertEqual(str(2 - expr_r), "2 + -1*C+(1,up)")
        expr_r -= 2.0
        self.assertEqual(str(expr_r - 0), "-2 + 1*C+(1,up)")
        self.assertEqual(str(0 - expr_r), "2 + -1*C+(1,up)")
        self.assertEqual(str(expr_r - 2), "-4 + 1*C+(1,up)")
        self.assertEqual(str(2 - expr_r), "4 + -1*C+(1,up)")
        self.assertEqual(str(expr_r - (-2)), "1*C+(1,up)")
        self.assertEqual(str((-2) - expr_r), "-1*C+(1,up)")
        # Complex and real
        expr_c = make_complex(c_dag(1, "up"))
        self.assertIsInstance(expr_c - 2.0, ExpressionC)
        self.assertIsInstance(2.0 - expr_c, ExpressionC)
        self.assertEqual(str(expr_c - 0.0), "(1,0)*C+(1,up)")
        self.assertEqual(str(0.0 - expr_c), "(-1,-0)*C+(1,up)")
        self.assertEqual(str(expr_c - 2.0), "(-2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0 - expr_c), "(2,0) + (-1,-0)*C+(1,up)")
        expr_c -= 2.0
        self.assertEqual(str(expr_c - 0.0), "(-2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(0.0 - expr_c), "(2,-0) + (-1,-0)*C+(1,up)")
        self.assertEqual(str(expr_c - 2.0), "(-4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0 - expr_c), "(4,-0) + (-1,-0)*C+(1,up)")
        self.assertEqual(str(expr_c - (-2.0)), "(1,0)*C+(1,up)")
        self.assertEqual(str((-2.0) - expr_c), "(-1,-0)*C+(1,up)")
        # Real and complex
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r - 2j, ExpressionC)
        self.assertIsInstance(2j - expr_r, ExpressionC)
        self.assertEqual(str(expr_r - 0.0j), "(1,0)*C+(1,up)")
        self.assertEqual(str(0.0j - expr_r), "(-1,0)*C+(1,up)")
        self.assertEqual(str(expr_r - 2.0j), "(0,-2) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0j - expr_r), "(0,2) + (-1,0)*C+(1,up)")
        expr_r -= 2.0
        self.assertEqual(str(expr_r - 0.0j), "(-2,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(0.0j - expr_r), "(2,0) + (-1,0)*C+(1,up)")
        self.assertEqual(str(expr_r - 2.0 + 0j), "(-4,0) + (1,0)*C+(1,up)")
        self.assertEqual(str(2.0 + 0j - expr_r), "(4,0) + (-1,0)*C+(1,up)")
        self.assertEqual(str(expr_r - (-2.0 + 0j)), "(1,0)*C+(1,up)")
        self.assertEqual(str((-2.0 + 0j) - expr_r), "(-1,0)*C+(1,up)")

    def test_multiplication(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r * c(2, "dn"), ExpressionR)
        self.assertIsInstance(c(2, "dn") * expr_r, ExpressionR)
        self.assertEqual(str(ExpressionR() * ExpressionR()), "0")
        self.assertEqual(str(expr_r * ExpressionR()), "0")
        self.assertEqual(str(ExpressionR() * expr_r), "0")
        self.assertEqual(str(expr_r * c(2, "dn")), "1*C+(1,up)C(2,dn)")
        self.assertEqual(str(c(2, "dn") * expr_r), "-1*C+(1,up)C(2,dn)")
        expr_r *= c(2, "dn")
        self.assertEqual(str(expr_r * a(0, "x")), "1*C+(1,up)C(2,dn)A(0,x)")
        self.assertEqual(str(a(0, "x") * expr_r), "1*C+(1,up)C(2,dn)A(0,x)")
        self.assertEqual(str(expr_r * c_dag(1, "up")), "0")
        self.assertEqual(str(c_dag(1, "up") * expr_r), "0")
        # Real and complex
        expr1 = make_complex(c_dag(1, "up"))
        expr2 = c(2, "dn")
        self.assertIsInstance(expr1 * expr2, ExpressionC)
        self.assertIsInstance(expr2 * expr1, ExpressionC)
        self.assertEqual(str(ExpressionC() * ExpressionR()), "(0,0)")
        self.assertEqual(str(ExpressionR() * ExpressionC()), "(0,0)")
        self.assertEqual(str(expr1 * ExpressionR()), "(0,0)")
        self.assertEqual(str(ExpressionR() * expr1), "(0,0)")
        self.assertEqual(str(expr2 * ExpressionC()), "(0,0)")
        self.assertEqual(str(ExpressionC() * expr2), "(0,0)")
        self.assertEqual(str(expr1 * expr2), "(1,0)*C+(1,up)C(2,dn)")
        self.assertEqual(str(expr2 * expr1), "(-1,0)*C+(1,up)C(2,dn)")
        expr1 *= expr2
        self.assertEqual(str(expr1 * a(0, "x")), "(1,0)*C+(1,up)C(2,dn)A(0,x)")
        self.assertEqual(str(a(0, "x") * expr1), "(1,0)*C+(1,up)C(2,dn)A(0,x)")
        self.assertEqual(str(expr1 * make_complex(c_dag(1, "up"))), "(0,0)")
        self.assertEqual(str(make_complex(c_dag(1, "up")) * expr1), "(0,0)")

    def test_multiplication_const(self):
        # Real
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r * 2, ExpressionR)
        self.assertIsInstance(2 * expr_r, ExpressionR)
        self.assertEqual(str(expr_r * 0), "0")
        self.assertEqual(str(0 * expr_r), "0")
        self.assertEqual(str(expr_r * 2), "2*C+(1,up)")
        self.assertEqual(str(2 * expr_r), "2*C+(1,up)")
        # Complex and real
        expr_c = make_complex(c_dag(1, "up"))
        self.assertIsInstance(expr_c * 2.0, ExpressionC)
        self.assertIsInstance(2.0 * expr_c, ExpressionC)
        self.assertEqual(str(expr_c * 0.0), "(0,0)")
        self.assertEqual(str(0.0 * expr_c), "(0,0)")
        self.assertEqual(str(expr_c * 2.0), "(2,0)*C+(1,up)")
        self.assertEqual(str(2.0 * expr_c), "(2,0)*C+(1,up)")
        # Real and complex
        expr_r = c_dag(1, "up")
        self.assertIsInstance(expr_r * 2.0j, ExpressionC)
        self.assertIsInstance(2.0j * expr_r, ExpressionC)
        self.assertEqual(str(expr_r * 0.0j), "(0,0)")
        self.assertEqual(str(0.0j * expr_r), "(0,0)")
        self.assertEqual(str(expr_r * 2.0j), "(0,2)*C+(1,up)")
        self.assertEqual(str(2.0j * expr_r), "(0,2)*C+(1,up)")
