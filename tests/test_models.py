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

from pycommute.models import *

class TestModels(TestCase):

    def test_tight_binding(self):
        indices = ["A", "B", "C"]
        M = np.array([[1.0, 0.0, 0.5],
                      [0.0, 2.0, 0.0],
                      [0.5, 0.0, 3.0]])
        H1 = tight_binding(M, indices)
        H2 = tight_binding(M, indices, FERMION)
        H3 = tight_binding(M, indices, BOSON)

        ref = c_dag("A")*c("A") + 2*c_dag("B")*c("B") + 3.0*c_dag("C")*c("C")
        ref += 0.5*c_dag("A")*c("C") + 0.5*c_dag("C")*c("A")
        self.assertEqual(H1, ref)
        self.assertEqual(H2, ref)
        ref = a_dag("A")*a("A") + 2*a_dag("B")*a("B") + 3.0*a_dag("C")*a("C")
        ref += 0.5*a_dag("A")*a("C") + 0.5*a_dag("C")*a("A")
        self.assertEqual(H3, ref)

        M = np.array([[1.0, 0.0, 0.5j],
                      [0.0, 2.0, 0.0],
                      [-0.5j, 0.0, 3.0]])
        H4 = tight_binding(M, indices)
        H5 = tight_binding(M, indices, FERMION)
        H6 = tight_binding(M, indices, BOSON)

        ref = c_dag("A")*c("A") + 2*c_dag("B")*c("B") + 3.0*c_dag("C")*c("C")
        ref += 0.5j*c_dag("A")*c("C") - 0.5j*c_dag("C")*c("A")
        self.assertEqual(H4, ref)
        self.assertEqual(H5, ref)
        ref = a_dag("A")*a("A") + 2*a_dag("B")*a("B") + 3.0*a_dag("C")*a("C")
        ref += 0.5j*a_dag("A")*a("C") - 0.5j*a_dag("C")*a("A")
        self.assertEqual(H6, ref)
