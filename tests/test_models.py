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
    c, c_dag, a, a_dag, S_p, S_m, S_x, S_y, S_z
)
from pycommute.models import (
    tight_binding,
    heisenberg
)

import numpy as np
from functools import partial


class TestModels(TestCase):

    def test_tight_binding(self):
        indices = ["A", "B", "C"]
        M = np.array([[1.0, 0.0, 0.5],
                      [0.0, 2.0, 0.0],
                      [0.5, 0.0, 3.0]])
        H1 = tight_binding(M, indices)
        H2 = tight_binding(M, indices, FERMION)
        H3 = tight_binding(M, indices, BOSON)

        ref = c_dag("A") * c("A") + 2 * c_dag("B") * c("B") \
            + 3.0 * c_dag("C") * c("C")
        ref += 0.5 * c_dag("A") * c("C") + 0.5 * c_dag("C") * c("A")
        self.assertEqual(H1, ref)
        self.assertEqual(H2, ref)
        ref = a_dag("A") * a("A") + 2 * a_dag("B") * a("B") \
            + 3.0 * a_dag("C") * a("C")
        ref += 0.5 * a_dag("A") * a("C") + 0.5 * a_dag("C") * a("A")
        self.assertEqual(H3, ref)

        M = np.array([[1.0, 0.0, 0.5j],
                      [0.0, 2.0, 0.0],
                      [-0.5j, 0.0, 3.0]])
        H4 = tight_binding(M, indices)
        H5 = tight_binding(M, indices, FERMION)
        H6 = tight_binding(M, indices, BOSON)

        ref = c_dag("A") * c("A") + 2 * c_dag("B") * c("B") \
            + 3.0 * c_dag("C") * c("C")
        ref += 0.5j * c_dag("A") * c("C") - 0.5j * c_dag("C") * c("A")
        self.assertEqual(H4, ref)
        self.assertEqual(H5, ref)
        ref = a_dag("A") * a("A") + 2 * a_dag("B") * a("B") \
            + 3.0 * a_dag("C") * a("C")
        ref += 0.5j * a_dag("A") * a("C") - 0.5j * a_dag("C") * a("A")
        self.assertEqual(H6, ref)

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
        ref_1 = - 0.5 * S_p(0) * S_p(1) \
                - 0.5 * S_m(0) * S_m(1) \
                - 1.0 * S_z(0) * S_z(1) \
                - 1.0 * S_p(1) * S_p(2) \
                - 1.0 * S_m(1) * S_m(2) \
                - 2.0 * S_z(1) * S_z(2) \
                - 1.5 * S_p(2) * S_p(3) \
                - 1.5 * S_m(2) * S_m(3) \
                - 3.0 * S_z(2) * S_z(3)
        self.assertEqual(H1, ref_1)

        H2 = heisenberg(1j * J)
        self.assertIsInstance(H2, ExpressionC)
        ref_2 = 1j * ref_1
        self.assertEqual(H2, ref_2)

        H3 = heisenberg(J, h)
        self.assertIsInstance(H3, ExpressionC)
        ref_3 = ref_1 - 0.3 * S_x(0) - 0.4 * S_y(1) - 0.5 * S_z(2)
        self.assertEqual(H3, ref_3)

        H4 = heisenberg(J, h, sites)
        self.assertIsInstance(H4, ExpressionC)
        ref_4 = - 0.5 * S_p('a', 0) * S_p('b', 1) \
                - 0.5 * S_m('a', 0) * S_m('b', 1) \
                - 1.0 * S_z('a', 0) * S_z('b', 1) \
                - 1.0 * S_p('b', 1) * S_p('c', 2) \
                - 1.0 * S_m('b', 1) * S_m('c', 2) \
                - 2.0 * S_z('b', 1) * S_z('c', 2) \
                - 1.5 * S_p('c', 2) * S_p('d', 3) \
                - 1.5 * S_m('c', 2) * S_m('d', 3) \
                - 3.0 * S_z('c', 2) * S_z('d', 3) \
                - 0.3 * S_x('a', 0) - 0.4 * S_y('b', 1) - 0.5 * S_z('c', 2)
        self.assertEqual(H4, ref_4)

        H5 = heisenberg(J, h, sites, spin=1)
        self.assertIsInstance(H5, ExpressionC)
        S1_p = partial(S_p, spin=1)
        S1_m = partial(S_m, spin=1)
        S1_x = partial(S_x, spin=1)
        S1_y = partial(S_y, spin=1)
        S1_z = partial(S_z, spin=1)
        ref_5 = - 0.5 * S1_p('a', 0) * S1_p('b', 1) \
                - 0.5 * S1_m('a', 0) * S1_m('b', 1) \
                - 1.0 * S1_z('a', 0) * S1_z('b', 1) \
                - 1.0 * S1_p('b', 1) * S1_p('c', 2) \
                - 1.0 * S1_m('b', 1) * S1_m('c', 2) \
                - 2.0 * S1_z('b', 1) * S1_z('c', 2) \
                - 1.5 * S1_p('c', 2) * S1_p('d', 3) \
                - 1.5 * S1_m('c', 2) * S1_m('d', 3) \
                - 3.0 * S1_z('c', 2) * S1_z('d', 3) \
                - 0.3 * S1_x('a', 0) - 0.4 * S1_y('b', 1) - 0.5 * S1_z('c', 2)
        self.assertEqual(H5, ref_5)
