#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2024 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from pycommute.expression import (
    make_fermion, make_boson, make_spin,
    Monomial,
    SpinComponent,
    c, c_dag, n, a, a_dag, S_p, S_m, S_x, S_y, S_z,
    make_complex
)


class TestFactories(TestCase):

    def check_monomial(self, expr, ref_coeff, *generators):
        ref_monomial = Monomial(list(generators))
        # Real
        self.assertEqual(len(expr), 1)
        self.assertEqual(*iter(expr), (ref_monomial, ref_coeff))
        # Complex
        expr_c = make_complex(expr)
        self.assertEqual(len(expr_c), 1)
        self.assertEqual(*iter(expr_c), (ref_monomial, complex(ref_coeff)))

    def test_make_complex(self):
        expr = make_complex(4.0 * c_dag(1, "up") * c(2, "dn") + 1.0)
        self.assertEqual(expr, (4 + 0j) * c_dag(1, "up") * c(2, "dn") + 1 + 0j)

    def test_real_complex(self):
        # Fermion
        self.check_monomial(c_dag(1, "up"), 1, make_fermion(True, 1, "up"))
        self.check_monomial(c(2, "dn"), 1, make_fermion(False, 2, "dn"))
        self.check_monomial(
            n(1, "dn"),
            1,
            make_fermion(True, 1, "dn"), make_fermion(False, 1, "dn")
        )
        # Boson
        self.check_monomial(a_dag(0, "x"), 1, make_boson(True, 0, "x"))
        self.check_monomial(a(0, "y"), 1, make_boson(False, 0, "y"))
        # Spin 1/2
        self.check_monomial(S_p(0, "x"), 1,
                            make_spin(SpinComponent.PLUS, 0, "x"))
        self.check_monomial(S_m(0, "x"), 1,
                            make_spin(SpinComponent.MINUS, 0, "x"))
        self.check_monomial(S_z(0, "x"), 1,
                            make_spin(SpinComponent.Z, 0, "x"))
        # Spin 1
        self.check_monomial(S_p(0, "x", spin=1), 1,
                            make_spin(1.0, SpinComponent.PLUS, 0, "x"))
        self.check_monomial(S_m(0, "x", spin=1), 1,
                            make_spin(1.0, SpinComponent.MINUS, 0, "x"))
        self.check_monomial(S_z(0, "x", spin=1), 1,
                            make_spin(1.0, SpinComponent.Z, 0, "x"))
        # Spin 3/2
        self.check_monomial(S_p(0, "x", spin=3 / 2), 1,
                            make_spin(3 / 2, SpinComponent.PLUS, 0, "x"))
        self.check_monomial(S_m(0, "x", spin=3 / 2), 1,
                            make_spin(3 / 2, SpinComponent.MINUS, 0, "x"))
        self.check_monomial(S_z(0, "x", spin=3 / 2), 1,
                            make_spin(3 / 2, SpinComponent.Z, 0, "x"))

    def test_complex_only(self):
        # Spin 1/2
        self.assertEqual(make_complex(S_p(0, "x")),
                         S_x(0, "x") + 1j * S_y(0, "x"))
        self.assertEqual(make_complex(S_m(0, "x")),
                         S_x(0, "x") - 1j * S_y(0, "x"))
        # Spin 1
        self.assertEqual(make_complex(S_p(0, "x", spin=1)),
                         S_x(0, "x", spin=1) + 1j * S_y(0, "x", spin=1))
        self.assertEqual(make_complex(S_m(0, "x", spin=1)),
                         S_x(0, "x", spin=1) - 1j * S_y(0, "x", spin=1))
        # Spin 3/2
        self.assertEqual(make_complex(S_p(0, "x", spin=3 / 2)),
                         S_x(0, "x", spin=3 / 2) + 1j * S_y(0, "x", spin=3 / 2))
        self.assertEqual(make_complex(S_m(0, "x", spin=3 / 2)),
                         S_x(0, "x", spin=3 / 2) - 1j * S_y(0, "x", spin=3 / 2))

    def test_mixed_indices(self):
        expr = c_dag(0, "up") * c(1, "dn") \
            + a_dag("x") * n() + a("y") * n() + S_p() * S_m()
        self.assertEqual(
            str(expr),
            "0.5 + 1*Sz() + 1*C+(0,up)C(1,dn) + 1*C+()C()A+(x) + 1*C+()C()A(y)"
        )
