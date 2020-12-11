#
# This file is part of libcommute, a C++11/14/17 header-only library allowing
# to manipulate polynomial expressions with quantum-mechanical operators.
#
# Copyright (C) 2016-2020 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from pycommute.expression import *

class TestExpression(TestCase):

    def test_init(self):
        self.assertEqual(str(ExpressionR()), "0")
        self.assertEqual(str(ExpressionR(1e-100)), "0")
        expr_r_const = ExpressionR(2)
        self.assertEqual(str(expr_r_const), "2")
        expr_c_const = ExpressionC(2)
        self.assertEqual(str(expr_c_const), "(2,0)")
        self.assertEqual(str(ExpressionC(expr_r_const)), "(2,0)")
        mon = Monomial([make_fermion(True, 1, "up"),
                        make_fermion(False, 2, "dn")])
        self.assertEqual(str(ExpressionR(1e-100, mon)), "0")
        expr_mon = ExpressionR(3, mon)
        self.assertEqual(str(expr_mon), "3*C+(1,up)C(2,dn)")
        expr_mon.clear()
        self.assertEqual(len(expr_mon), 0)

    # TODO: test_S_z_products()
    # TODO: test_unary_minus()
    # TODO: test_iter()
    # TODO: test_transform()
    # TODO: test_conj()
    # TODO: test_spin12_products()
    # TODO: test_powers_of_S_z()
    # TODO: test_Heisenberg()
