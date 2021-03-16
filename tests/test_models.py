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
    ExpressionR, ExpressionC,
    FERMION, BOSON,
    c, c_dag, n, a, a_dag, S_p, S_m, S_x, S_y, S_z
)
from pycommute.models import (
    tight_binding,
    dispersion,
    zeeman,
    hubbard_int,
    bose_hubbard_int,
    extended_hubbard_int,
    t_j_int,
    ising,
    heisenberg,
    anisotropic_heisenberg,
    biquadratic_spin_int,
    dzyaloshinskii_moriya
)

import numpy as np
from functools import partial

S1_p, S1_m = partial(S_p, spin=1), partial(S_m, spin=1)
S1_x = partial(S_x, spin=1)
S1_y = partial(S_y, spin=1)
S1_z = partial(S_z, spin=1)


class TestModels(TestCase):

    def test_tight_binding(self):
        sites = [("a", 0), ("b", 1), ("c", 2)]
        t = np.array([[1.0, 0.0, 0.5],
                      [0.0, 2.0, 0.0],
                      [0.5, 0.0, 3.0]])

        H1 = tight_binding(t)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = c_dag(0) * c(0) + 2 * c_dag(1) * c(1) + 3.0 * c_dag(2) * c(2)
        ref1 += 0.5 * c_dag(0) * c(2) + 0.5 * c_dag(2) * c(0)
        self.assertEqual(H1, ref1)

        H2 = tight_binding(1j * t)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = tight_binding(t, statistics=FERMION)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = ref1
        self.assertEqual(H3, ref3)

        H4 = tight_binding(t, sites)
        self.assertIsInstance(H4, ExpressionR)
        ref4 = c_dag("a", 0) * c("a", 0) + 2 * c_dag("b", 1) * c("b", 1) \
            + 3.0 * c_dag("c", 2) * c("c", 2)
        ref4 += 0.5 * c_dag("a", 0) * c("c", 2) \
            + 0.5 * c_dag("c", 2) * c("a", 0)
        self.assertEqual(H4, ref4)

        H5 = tight_binding(t, statistics=BOSON)
        self.assertIsInstance(H5, ExpressionR)
        ref5 = a_dag(0) * a(0) + 2 * a_dag(1) * a(1) + 3.0 * a_dag(2) * a(2)
        ref5 += 0.5 * a_dag(0) * a(2) + 0.5 * a_dag(2) * a(0)
        self.assertEqual(H5, ref5)

    def test_dispersion(self):
        sites = [("a", 0), ("b", 1), ("c", 2)]
        eps = np.array([1.0, 2.0, 3.0])

        H1 = dispersion(eps)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = c_dag(0) * c(0) + 2 * c_dag(1) * c(1) + 3.0 * c_dag(2) * c(2)
        self.assertEqual(H1, ref1)

        H2 = dispersion(1j * eps)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = dispersion(eps, statistics=FERMION)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = ref1
        self.assertEqual(H3, ref3)

        H4 = dispersion(eps, sites)
        self.assertIsInstance(H4, ExpressionR)
        ref4 = c_dag("a", 0) * c("a", 0) + 2 * c_dag("b", 1) * c("b", 1) \
            + 3.0 * c_dag("c", 2) * c("c", 2)
        self.assertEqual(H4, ref4)

        H5 = dispersion(eps, statistics=BOSON)
        self.assertIsInstance(H5, ExpressionR)
        ref5 = a_dag(0) * a(0) + 2 * a_dag(1) * a(1) + 3.0 * a_dag(2) * a(2)
        self.assertEqual(H5, ref5)

    def test_zeeman(self):
        b1 = np.array([1, 2, 3, 0], dtype=float)
        b2 = np.array([[1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 3],
                       [0, 0, 0]], dtype=float)
        indices = ([("up", 0), ("up", 1), ("up", 2), ("up", 3)],
                   [("dn", 0), ("dn", 1), ("dn", 2), ("dn", 3)])

        H1 = zeeman(b1)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = n(0, "up") - n(0, "dn") + 2.0 * (n(1, "up") - n(1, "dn")) \
            + 3.0 * (n(2, "up") - n(2, "dn"))
        self.assertEqual(H1, ref1)

        H2 = zeeman(1j * b1)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = zeeman(b1, indices)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = n("up", 0) - n("dn", 0) + 2.0 * (n("up", 1) - n("dn", 1)) \
            + 3.0 * (n("up", 2) - n("dn", 2))
        self.assertEqual(H3, ref3)

        H4 = zeeman(b2)
        self.assertIsInstance(H4, ExpressionC)
        ref4 = (c_dag(0, "up") * c(0, "dn") + c_dag(0, "dn") * c(0, "up")) \
            + 2.0j * (c_dag(1, "dn") * c(1, "up")
                      - c_dag(1, "up") * c(1, "dn")) \
            + 3.0 * (n(2, "up") - n(2, "dn"))
        self.assertEqual(H4, ref4)

        H5 = zeeman(1j * b2)
        self.assertIsInstance(H5, ExpressionC)
        ref5 = 1j * ref4
        self.assertEqual(H5, ref5)

        H6 = zeeman(b2, indices)
        self.assertIsInstance(H6, ExpressionC)
        ref6 = (c_dag("up", 0) * c("dn", 0) + c_dag("dn", 0) * c("up", 0)) \
            + 2.0j * (c_dag("dn", 1) * c("up", 1)
                      - c_dag("up", 1) * c("dn", 1)) \
            + 3.0 * (n("up", 2) - n("dn", 2))
        self.assertEqual(H6, ref6)

    def test_hubbard_int(self):
        U = np.array([1, 2, 3, 4], dtype=float)
        indices = ([("up", 0), ("up", 1), ("up", 2), ("up", 3)],
                   [("dn", 0), ("dn", 1), ("dn", 2), ("dn", 3)])

        H1 = hubbard_int(U)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = n(0, "up") * n(0, "dn") + 2.0 * n(1, "up") * n(1, "dn") \
            + 3.0 * n(2, "up") * n(2, "dn") + 4.0 * n(3, "up") * n(3, "dn")
        self.assertEqual(H1, ref1)

        H2 = hubbard_int(1j * U)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = hubbard_int(U, indices)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = n("up", 0) * n("dn", 0) + 2.0 * n("up", 1) * n("dn", 1) \
            + 3.0 * n("up", 2) * n("dn", 2) + 4.0 * n("up", 3) * n("dn", 3)
        self.assertEqual(H3, ref3)

    def test_bose_hubbard_int(self):
        U = np.array([1, 2, 3, 4], dtype=float)
        indices = [("a", 0), ("b", 1), ("c", 2), ("d", 3)]

        def nb(*args):
            return a_dag(*args) * a(*args)

        H1 = bose_hubbard_int(U)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = nb(0) * (nb(0) - 1) + 2.0 * nb(1) * (nb(1) - 1) \
            + 3.0 * nb(2) * (nb(2) - 1) + 4.0 * nb(3) * (nb(3) - 1)
        self.assertEqual(H1, ref1)

        H2 = bose_hubbard_int(1j * U)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = bose_hubbard_int(U, indices)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = nb("a", 0) * (nb("a", 0) - 1) \
            + 2.0 * nb("b", 1) * (nb("b", 1) - 1) \
            + 3.0 * nb("c", 2) * (nb("c", 2) - 1) \
            + 4.0 * nb("d", 3) * (nb("d", 3) - 1)
        self.assertEqual(H3, ref3)

    def test_extended_hubbard_int(self):
        V = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]], dtype=float)
        indices = ([("up", 0), ("up", 1), ("up", 2), ("up", 3)],
                   [("dn", 0), ("dn", 1), ("dn", 2), ("dn", 3)])

        H1 = extended_hubbard_int(V)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = 0.5 * (n(0, "up") + n(0, "dn")) * (n(1, "up") + n(1, "dn")) \
            + 1.0 * (n(1, "up") + n(1, "dn")) * (n(2, "up") + n(2, "dn")) \
            + 1.5 * (n(2, "up") + n(2, "dn")) * (n(3, "up") + n(3, "dn"))
        self.assertEqual(H1, ref1)

        H2 = extended_hubbard_int(1j * V)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = extended_hubbard_int(V, indices)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = 0.5 * (n("up", 0) + n("dn", 0)) * (n("up", 1) + n("dn", 1)) \
            + 1.0 * (n("up", 1) + n("dn", 1)) * (n("up", 2) + n("dn", 2)) \
            + 1.5 * (n("up", 2) + n("dn", 2)) * (n("up", 3) + n("dn", 3))
        self.assertEqual(H3, ref3)

    def test_t_j_int(self):
        J = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]], dtype=float)
        indices = ([("up", 0), ("up", 1), ("up", 2), ("up", 3)],
                   [("dn", 0), ("dn", 1), ("dn", 2), ("dn", 3)])

        H1 = t_j_int(J)
        self.assertIsInstance(H1, ExpressionR)
        sp = [c_dag(i, "up") * c(i, "dn") for i in range(4)]
        sm = [c_dag(i, "dn") * c(i, "up") for i in range(4)]
        sz = [0.5 * (n(i, "up") - n(i, "dn")) for i in range(4)]
        ni = [n(i, "up") + n(i, "dn") for i in range(4)]

        ref1 = 0.5 * (sp[0] * sm[1] + sm[0] * sp[1]) + sz[0] * sz[1]
        ref1 += -0.25 * ni[0] * ni[1]
        ref1 += 2.0 * (0.5 * (sp[1] * sm[2] + sm[1] * sp[2]) + sz[1] * sz[2])
        ref1 += -0.5 * ni[1] * ni[2]
        ref1 += 3.0 * (0.5 * (sp[2] * sm[3] + sm[2] * sp[3]) + sz[2] * sz[3])
        ref1 += -0.75 * ni[2] * ni[3]
        self.assertEqual(H1, ref1)

        H2 = t_j_int(1j * J)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = t_j_int(J, indices)
        self.assertIsInstance(H3, ExpressionR)
        sp = [c_dag("up", i) * c("dn", i) for i in range(4)]
        sm = [c_dag("dn", i) * c("up", i) for i in range(4)]
        sz = [0.5 * (n("up", i) - n("dn", i)) for i in range(4)]
        ni = [n("up", i) + n("dn", i) for i in range(4)]

        ref3 = 0.5 * (sp[0] * sm[1] + sm[0] * sp[1]) + sz[0] * sz[1]
        ref3 += -0.25 * ni[0] * ni[1]
        ref3 += 2.0 * (0.5 * (sp[1] * sm[2] + sm[1] * sp[2]) + sz[1] * sz[2])
        ref3 += -0.5 * ni[1] * ni[2]
        ref3 += 3.0 * (0.5 * (sp[2] * sm[3] + sm[2] * sp[3]) + sz[2] * sz[3])
        ref3 += -0.75 * ni[2] * ni[3]
        self.assertEqual(H3, ref3)

    def test_ising(self):
        J = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]], dtype=float)
        h_l = np.array([0.3, 0.4, 0.5, 0.0])
        h_t = np.array([0.0, 0.6, 0.7, 0.8])
        sites = [('a', 0), ('b', 1), ('c', 2), ('d', 3)]

        H1 = ising(J)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = - 1.0 * S_z(0) * S_z(1) \
               - 2.0 * S_z(1) * S_z(2) \
               - 3.0 * S_z(2) * S_z(3)
        self.assertEqual(H1, ref1)

        H2 = ising(1j * J)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = ising(J, h_l=h_l)
        self.assertIsInstance(H3, ExpressionR)
        ref3 = ref1 - 0.3 * S_z(0) - 0.4 * S_z(1) - 0.5 * S_z(2)
        self.assertEqual(H3, ref3)

        H4 = ising(J, h_l=1j * h_l)
        self.assertIsInstance(H4, ExpressionC)
        ref4 = ref1 - 1j * (0.3 * S_z(0) + 0.4 * S_z(1) + 0.5 * S_z(2))
        self.assertEqual(H4, ref4)

        H5 = ising(J, h_t=h_t)
        self.assertIsInstance(H5, ExpressionC)
        ref5 = ref1 - 0.6 * S_x(1) - 0.7 * S_x(2) - 0.8 * S_x(3)
        self.assertEqual(H5, ref5)

        H6 = ising(J, h_t=1j * h_t)
        self.assertIsInstance(H6, ExpressionC)
        ref6 = ref1 - 1j * (0.6 * S_x(1) + 0.7 * S_x(2) + 0.8 * S_x(3))
        self.assertEqual(H6, ref6)

        H7 = ising(J, h_l, h_t, sites=sites)
        self.assertIsInstance(H7, ExpressionC)
        ref7 = - 1.0 * S_z('a', 0) * S_z('b', 1) \
               - 2.0 * S_z('b', 1) * S_z('c', 2) \
               - 3.0 * S_z('c', 2) * S_z('d', 3) \
               - 0.3 * S_z('a', 0) - 0.4 * S_z('b', 1) - 0.5 * S_z('c', 2) \
               - 0.6 * S_x('b', 1) - 0.7 * S_x('c', 2) - 0.8 * S_x('d', 3)
        self.assertEqual(H7, ref7)

        H8 = ising(J, h_l, h_t, sites=sites, spin=1)
        self.assertIsInstance(H8, ExpressionC)
        ref8 = - 1.0 * S1_z('a', 0) * S1_z('b', 1) \
               - 2.0 * S1_z('b', 1) * S1_z('c', 2) \
               - 3.0 * S1_z('c', 2) * S1_z('d', 3) \
               - 0.3 * S1_z('a', 0) - 0.4 * S1_z('b', 1) - 0.5 * S1_z('c', 2)\
               - 0.6 * S1_x('b', 1) - 0.7 * S1_x('c', 2) - 0.8 * S1_x('d', 3)
        self.assertEqual(H8, ref8)

    def test_heisenberg(self):
        J = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]], dtype=float)
        h = np.array([[0.3, 0, 0],
                      [0, 0.4, 0],
                      [0, 0, 0.5],
                      [0, 0, 0]])
        sites = [('a', 0), ('b', 1), ('c', 2), ('d', 3)]

        H1 = heisenberg(J)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = - 0.5 * S_p(0) * S_p(1) \
               - 0.5 * S_m(0) * S_m(1) \
               - 1.0 * S_z(0) * S_z(1) \
               - 1.0 * S_p(1) * S_p(2) \
               - 1.0 * S_m(1) * S_m(2) \
               - 2.0 * S_z(1) * S_z(2) \
               - 1.5 * S_p(2) * S_p(3) \
               - 1.5 * S_m(2) * S_m(3) \
               - 3.0 * S_z(2) * S_z(3)
        self.assertEqual(H1, ref1)

        H2 = heisenberg(1j * J)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = heisenberg(J, h)
        self.assertIsInstance(H3, ExpressionC)
        ref3 = ref1 - 0.3 * S_x(0) - 0.4 * S_y(1) - 0.5 * S_z(2)
        self.assertEqual(H3, ref3)

        H4 = heisenberg(J, h, sites=sites)
        self.assertIsInstance(H4, ExpressionC)
        ref4 = - 0.5 * S_p('a', 0) * S_p('b', 1) \
               - 0.5 * S_m('a', 0) * S_m('b', 1) \
               - 1.0 * S_z('a', 0) * S_z('b', 1) \
               - 1.0 * S_p('b', 1) * S_p('c', 2) \
               - 1.0 * S_m('b', 1) * S_m('c', 2) \
               - 2.0 * S_z('b', 1) * S_z('c', 2) \
               - 1.5 * S_p('c', 2) * S_p('d', 3) \
               - 1.5 * S_m('c', 2) * S_m('d', 3) \
               - 3.0 * S_z('c', 2) * S_z('d', 3) \
               - 0.3 * S_x('a', 0) - 0.4 * S_y('b', 1) - 0.5 * S_z('c', 2)
        self.assertEqual(H4, ref4)

        H5 = heisenberg(J, h, sites=sites, spin=1)
        self.assertIsInstance(H5, ExpressionC)
        ref5 = - 0.5 * S1_p('a', 0) * S1_p('b', 1) \
               - 0.5 * S1_m('a', 0) * S1_m('b', 1) \
               - 1.0 * S1_z('a', 0) * S1_z('b', 1) \
               - 1.0 * S1_p('b', 1) * S1_p('c', 2) \
               - 1.0 * S1_m('b', 1) * S1_m('c', 2) \
               - 2.0 * S1_z('b', 1) * S1_z('c', 2) \
               - 1.5 * S1_p('c', 2) * S1_p('d', 3) \
               - 1.5 * S1_m('c', 2) * S1_m('d', 3) \
               - 3.0 * S1_z('c', 2) * S1_z('d', 3) \
               - 0.3 * S1_x('a', 0) - 0.4 * S1_y('b', 1) - 0.5 * S1_z('c', 2)
        self.assertEqual(H5, ref5)

    def test_anisotropic_heisenberg(self):
        Jx = np.array([[0, 1, 0, 0],
                       [0, 0, 2, 0],
                       [0, 0, 0, 3],
                       [0, 0, 0, 0]], dtype=float)
        Jy = np.array([[0, 4, 0, 0],
                       [0, 0, 5, 0],
                       [0, 0, 0, 6],
                       [0, 0, 0, 0]], dtype=float)
        Jz = np.array([[0, 7, 0, 0],
                       [0, 0, 8, 0],
                       [0, 0, 0, 9],
                       [0, 0, 0, 0]], dtype=float)
        h = np.array([[0.3, 0, 0],
                      [0, 0.4, 0],
                      [0, 0, 0.5],
                      [0, 0, 0]])
        sites = [('a', 0), ('b', 1), ('c', 2), ('d', 3)]

        H1 = anisotropic_heisenberg((Jx, Jy, Jz))
        self.assertIsInstance(H1, ExpressionC)
        ref1 = - 1.0 * S_x(0) * S_x(1) \
               - 4.0 * S_y(0) * S_y(1) \
               - 7.0 * S_z(0) * S_z(1) \
               - 2.0 * S_x(1) * S_x(2) \
               - 5.0 * S_y(1) * S_y(2) \
               - 8.0 * S_z(1) * S_z(2) \
               - 3.0 * S_x(2) * S_x(3) \
               - 6.0 * S_y(2) * S_y(3) \
               - 9.0 * S_z(2) * S_z(3)
        self.assertEqual(H1, ref1)

        H2 = anisotropic_heisenberg((Jx, Jy, Jz), h)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = ref1 - 0.3 * S_x(0) - 0.4 * S_y(1) - 0.5 * S_z(2)
        self.assertEqual(H2, ref2)

        H3 = anisotropic_heisenberg((Jx, Jy, Jz), h, sites=sites)
        self.assertIsInstance(H3, ExpressionC)
        ref3 = - 1.0 * S_x('a', 0) * S_x('b', 1) \
               - 4.0 * S_y('a', 0) * S_y('b', 1) \
               - 7.0 * S_z('a', 0) * S_z('b', 1) \
               - 2.0 * S_x('b', 1) * S_x('c', 2) \
               - 5.0 * S_y('b', 1) * S_y('c', 2) \
               - 8.0 * S_z('b', 1) * S_z('c', 2) \
               - 3.0 * S_x('c', 2) * S_x('d', 3) \
               - 6.0 * S_y('c', 2) * S_y('d', 3) \
               - 9.0 * S_z('c', 2) * S_z('d', 3) \
               - 0.3 * S_x('a', 0) - 0.4 * S_y('b', 1) - 0.5 * S_z('c', 2)
        self.assertEqual(H3, ref3)

        H4 = anisotropic_heisenberg((Jx, Jy, Jz), h, sites=sites, spin=1)
        self.assertIsInstance(H4, ExpressionC)
        ref4 = - 1.0 * S1_x('a', 0) * S1_x('b', 1) \
               - 4.0 * S1_y('a', 0) * S1_y('b', 1) \
               - 7.0 * S1_z('a', 0) * S1_z('b', 1) \
               - 2.0 * S1_x('b', 1) * S1_x('c', 2) \
               - 5.0 * S1_y('b', 1) * S1_y('c', 2) \
               - 8.0 * S1_z('b', 1) * S1_z('c', 2) \
               - 3.0 * S1_x('c', 2) * S1_x('d', 3) \
               - 6.0 * S1_y('c', 2) * S1_y('d', 3) \
               - 9.0 * S1_z('c', 2) * S1_z('d', 3) \
               - 0.3 * S1_x('a', 0) - 0.4 * S1_y('b', 1) - 0.5 * S1_z('c', 2)
        self.assertEqual(H4, ref4)

    def test_biquadratic_spin_int(self):
        J = np.array([[0, 1, 0, 0],
                      [0, 0, 2, 0],
                      [0, 0, 0, 3],
                      [0, 0, 0, 0]], dtype=float)
        sites = [('a', 0), ('b', 1), ('c', 2), ('d', 3)]

        S0S1 = 0.5 * (S1_p(0) * S1_p(1) + S1_m(0) * S1_m(1)) \
            + S1_z(0) * S1_z(1)
        S1S2 = 0.5 * (S1_p(1) * S1_p(2) + S1_m(1) * S1_m(2)) \
            + S1_z(1) * S1_z(2)
        S2S3 = 0.5 * (S1_p(2) * S1_p(3) + S1_m(2) * S1_m(3)) \
            + S1_z(2) * S1_z(3)

        H1 = biquadratic_spin_int(J)
        self.assertIsInstance(H1, ExpressionR)
        ref1 = 1.0 * S0S1 * S0S1 + 2.0 * S1S2 * S1S2 + 3.0 * S2S3 * S2S3
        self.assertEqual(H1, ref1)

        H2 = biquadratic_spin_int(1j * J)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1j * ref1
        self.assertEqual(H2, ref2)

        H3 = biquadratic_spin_int(J, sites=sites)
        self.assertIsInstance(H3, ExpressionR)

        S0S1 = 0.5 * (S1_p('a', 0) * S1_p('b', 1)
                      + S1_m('a', 0) * S1_m('b', 1)) \
            + S1_z('a', 0) * S1_z('b', 1)
        S1S2 = 0.5 * (S1_p('b', 1) * S1_p('c', 2)
                      + S1_m('b', 1) * S1_m('c', 2)) \
            + S1_z('b', 1) * S1_z('c', 2)
        S2S3 = 0.5 * (S1_p('c', 2) * S1_p('d', 3)
                      + S1_m('c', 2) * S1_m('d', 3)) \
            + S1_z('c', 2) * S1_z('d', 3)

        ref3 = 1.0 * S0S1 * S0S1 + 2.0 * S1S2 * S1S2 + 3.0 * S2S3 * S2S3
        self.assertEqual(H3, ref3)

        H4 = biquadratic_spin_int(J, sites=sites, spin=1 / 2)
        self.assertIsInstance(H4, ExpressionR)

        S0S1 = 0.5 * (S_p('a', 0) * S_p('b', 1)
                      + S_m('a', 0) * S_m('b', 1)) + S_z('a', 0) * S_z('b', 1)
        S1S2 = 0.5 * (S_p('b', 1) * S_p('c', 2)
                      + S_m('b', 1) * S_m('c', 2)) + S_z('b', 1) * S_z('c', 2)
        S2S3 = 0.5 * (S_p('c', 2) * S_p('d', 3)
                      + S_m('c', 2) * S_m('d', 3)) + S_z('c', 2) * S_z('d', 3)

        ref4 = 1.0 * S0S1 * S0S1 + 2.0 * S1S2 * S1S2 + 3.0 * S2S3 * S2S3
        self.assertEqual(H4, ref4)

    def test_dzyaloshinskii_moriya(self):
        D = np.zeros((4, 4, 3), dtype=float)
        D[0, 1, :] = [1.0, 0, 0]
        D[1, 2, :] = [0, 2.0, 0]
        D[2, 3, :] = [0, 0, 3.0]
        sites = [('a', 0), ('b', 1), ('c', 2), ('d', 3)]

        H1 = dzyaloshinskii_moriya(D)
        self.assertIsInstance(H1, ExpressionC)
        ref1 = 1.0 * (S_y(0) * S_z(1) - S_z(0) * S_y(1)) \
            + 2.0 * (S_z(1) * S_x(2) - S_x(1) * S_z(2)) \
            + 3.0 * (S_x(2) * S_y(3) - S_y(2) * S_x(3))
        self.assertEqual(H1, ref1)

        H2 = dzyaloshinskii_moriya(D, sites=sites)
        self.assertIsInstance(H2, ExpressionC)
        ref2 = 1.0 * (S_y('a', 0) * S_z('b', 1) - S_z('a', 0) * S_y('b', 1)) \
            + 2.0 * (S_z('b', 1) * S_x('c', 2) - S_x('b', 1) * S_z('c', 2)) \
            + 3.0 * (S_x('c', 2) * S_y('d', 3) - S_y('c', 2) * S_x('d', 3))
        self.assertEqual(H2, ref2)

        H3 = dzyaloshinskii_moriya(D, spin=1)
        self.assertIsInstance(H3, ExpressionC)
        ref3 = 1.0 * (S1_y(0) * S1_z(1) - S1_z(0) * S1_y(1)) \
            + 2.0 * (S1_z(1) * S1_x(2) - S1_x(1) * S1_z(2)) \
            + 3.0 * (S1_x(2) * S1_y(3) - S1_y(2) * S1_x(3))
        self.assertEqual(H3, ref3)
