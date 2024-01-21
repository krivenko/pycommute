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

from copy import copy, deepcopy

from pycommute.expression import (
    GeneratorBoson,
    Monomial,
    ExpressionR, ExpressionC,
    SpinComponent,
    make_fermion, make_boson, make_spin,
    c, c_dag, a, a_dag, S_p, S_m, S_x, S_y, S_z,
    conj,
    transform,
    hc
)


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

    def test_copy(self):
        expr = 4.0 * c_dag(1, "up") * c(2, "dn") + 1.0 \
            + 3.0 * a(0, "x") + 2.0 * a_dag(0, "y")
        exprc = copy(expr)
        self.assertEqual(exprc, expr)
        self.assertNotEqual(id(exprc), id(expr))
        exprdc = deepcopy(expr)
        self.assertEqual(exprdc, expr)
        self.assertNotEqual(id(exprdc), id(expr))

    def test_S_z_products(self):
        mon_sz = Monomial(
            [make_fermion(True, 1)]
            + [make_spin(SpinComponent.Z, 1)] * 4
            + [make_fermion(True, 2)]
            + [make_spin(SpinComponent.Z, 2)] * 4
            + [make_spin(SpinComponent.Z, 3)] * 3
            + [make_fermion(True, 4)]
            + [make_spin(SpinComponent.Z, 4)] * 3
        )
        expr_sz = ExpressionR(1.0, mon_sz)
        self.assertEqual(expr_sz,
                         c_dag(1) * 0.25 * 0.25 * c_dag(2)
                         * 0.25 * 0.25
                         * 0.25 * S_z(3) * c_dag(4)
                         * 0.25 * S_z(4)
                         )

    def test_iter(self):
        expr0 = ExpressionR()
        self.assertEqual([_ for _ in expr0], [])
        expr = 4.0 * c_dag(1, "up") * c(2, "dn") + 1.0 \
            + 3.0 * a(0, "x") + 2.0 * a_dag(0, "y")
        ref = [(Monomial(), 1.0),
               (Monomial([make_boson(True, 0, "y")]), 2.0),
               (Monomial([make_boson(False, 0, "x")]), 3.0),
               (Monomial([make_fermion(True, 1, "up"),
                          make_fermion(False, 2, "dn")]),
               4.0)]
        self.assertEqual([_ for _ in expr], ref)

    def test_transform(self):
        expr = 4.0 * c_dag(1, "up") * c(2, "dn") + 1.0 \
            + 3.0 * a(0, "x") + 2.0 * a_dag(0, "y")

        # Multiply coefficients in front of bosonic operators by 2j
        def f(m, c):
            if len(m) > 0 and isinstance(m[0], GeneratorBoson):
                return 2 * c
            else:
                return 0
        new_expr = transform(expr, f)
        self.assertEqual(new_expr, 6.0 * a(0, "x") + 4.0 * a_dag(0, "y"))

    def test_conj(self):
        expr = 4.0 * c_dag(1, "up") * c(2, "dn") + 1.0 \
            + 3.0 * a(0, "x") + 2j * a_dag(0, "y")
        ref = 4.0 * c_dag(2, "dn") * c(1, "up") + 1.0 \
            + 3.0 * a_dag(0, "x") + -2j * a(0, "y")
        self.assertEqual(conj(expr), ref)

    def test_spin12_products(self):
        self.assertEqual(S_z() * S_z(), ExpressionR(0.25))
        self.assertEqual(S_p() * S_p(), ExpressionR())
        self.assertEqual(S_m() * S_m(), ExpressionR())
        self.assertEqual(S_p() * S_z(), -0.5 * S_p())
        self.assertEqual(S_z() * S_m(), -0.5 * S_m())
        self.assertEqual(S_z() * S_p(), 0.5 * S_p())
        self.assertEqual(S_m() * S_z(), 0.5 * S_m())
        self.assertEqual(S_p() * S_m(), 0.5 + S_z())
        self.assertEqual(S_m() * S_p(), 0.5 - S_z())

    def test_powers_of_S_z(self):
        s = ExpressionR()
        for n in range(1, 12):
            p = c_dag()
            for _ in range(n):
                p *= S_z()
            p *= a()
            s += p
        self.assertEqual(
            s,
            c_dag() * (341.0 / 1024 + (1365.0 / 1024) * S_z()) * a()
        )

    def test_Heisenberg(self):

        # Addition of 3D vectors
        def add(S1, S2):
            return (S1[0] + S2[0], S1[1] + S2[1], S1[2] + S2[2])

        # Dot-product of 3D vectors
        def dot(S1, S2):
            return S1[0] * S2[0] + S1[1] * S2[1] + S1[2] * S2[2]

        # Cross-product of vectors
        def cross(S1, S2):
            return (S1[1] * S2[2] - S1[2] * S2[1],
                    S1[2] * S2[0] - S1[0] * S2[2],
                    S1[0] * S2[1] - S1[1] * S2[0])

        N = 6
        S = [(S_x(i), S_y(i), S_z(i)) for i in range(N)]
        H = sum(dot(S[i], S[(i + 1) % N]) for i in range(N))
        S_tot = (ExpressionC(),) * 3
        for i in range(N):
            S_tot += add(S_tot, S[i])

        # H must commute with the total spin
        self.assertEqual(len(H * S_tot[0] - S_tot[0] * H), 0)
        self.assertEqual(len(H * S_tot[1] - S_tot[1] * H), 0)
        self.assertEqual(len(H * S_tot[2] - S_tot[2] * H), 0)

        # Q3 is a higher-order integral of motion
        Q3 = sum(dot(cross(S[i], S[(i + 1) % N]), S[(i + 2) % N])
                 for i in range(N))
        self.assertEqual(len(H * Q3 - Q3 * H), 0)

    def test_hc(self):
        # Real
        expr = 2 * c_dag("up", 1) * c("up", 2)
        self.assertEqual(expr + hc, expr + conj(expr))
        self.assertEqual(expr - hc, expr - conj(expr))
        # Complex
        expr = (2 + 2j) * c_dag("up", 1) * c("up", 2)
        self.assertEqual(expr + hc, expr + conj(expr))
        self.assertEqual(expr - hc, expr - conj(expr))
