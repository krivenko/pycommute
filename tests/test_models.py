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
    FERMION, BOSON,
    c, c_dag, a, a_dag
)
from pycommute.models import (
    tight_binding,
)
from pycommute.lattices import (
    hypercubic_lattice
)

import numpy as np
from numpy.testing import assert_allclose


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

    def make_adj_matrix(self, size, edges):
        matrix = np.zeros((size, size), dtype=int)
        for e in edges:
            matrix[e[0], e[1]] += 1
            matrix[e[1], e[0]] += 1
        return matrix

    def test_hypercubic_lattice(self):
        shape = (2, 3)

        lat = hypercubic_lattice(shape,
                                 lambda ij: ("a%d" % ij[0], "b%d" % ij[1]))
        self.assertEqual(
            lat.node_names,
            [("a0", "b0"), ("a0", "b1"), ("a0", "b2"),
             ("a1", "b0"), ("a1", "b1"), ("a1", "b2")]
        )

        lat = hypercubic_lattice(shape)
        self.assertEqual(
          lat.node_names,
          [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        )

        i00, i01, i02, i10, i11, i12 = range(6)

        # periodic = (False, False)
        lat1 = hypercubic_lattice(shape, periodic=False)
        lat2 = hypercubic_lattice(shape, periodic=(False, False))
        self.assertEqual(len(lat1.subgraph_matrices), 2)
        self.assertEqual(len(lat2.subgraph_matrices), 2)
        ref_NN = self.make_adj_matrix(6,
            [(i00, i01), (i01, i02), (i10, i11), (i11, i12),
             (i00, i10), (i01, i11), (i02, i12)]
        )
        assert_allclose(lat1.subgraph_matrices['NN'], ref_NN)
        assert_allclose(lat2.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(6,
            [(i00, i11), (i01, i10), (i01, i12), (i02, i11)]
        )
        assert_allclose(lat1.subgraph_matrices['NNN'], ref_NNN)
        assert_allclose(lat2.subgraph_matrices['NNN'], ref_NNN)

        # periodic = (False, True)
        lat = hypercubic_lattice(shape, periodic=(False, True))
        self.assertEqual(len(lat.subgraph_matrices), 2)
        ref_NN = self.make_adj_matrix(6,
            [(i00, i01), (i01, i02), (i10, i11), (i11, i12),
             (i00, i10), (i01, i11), (i02, i12),
             (i00, i02), (i10, i12)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(6,
            [(i00, i11), (i01, i10), (i01, i12), (i02, i11),
             (i00, i12), (i10, i02)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        # periodic = (True, False)
        lat = hypercubic_lattice(shape, periodic=(True, False))
        self.assertEqual(len(lat.subgraph_matrices), 2)
        ref_NN = self.make_adj_matrix(6,
            [(i00, i01), (i01, i02), (i10, i11), (i11, i12),
             (i00, i10), (i01, i11), (i02, i12),
             (i10, i00), (i11, i01), (i12, i02)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(6,
            [(i00, i11), (i01, i10), (i01, i12), (i02, i11),
             (i10, i01), (i11, i00), (i11, i02), (i12, i01)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        # periodic = (True, True)
        lat1 = hypercubic_lattice(shape, periodic=True)
        lat2 = hypercubic_lattice(shape, periodic=(True, True))
        self.assertEqual(len(lat1.subgraph_matrices), 2)
        self.assertEqual(len(lat2.subgraph_matrices), 2)
        ref_NN = self.make_adj_matrix(6,
            [(i00, i01), (i01, i02), (i10, i11), (i11, i12),
             (i00, i10), (i01, i11), (i02, i12),
             (i10, i00), (i11, i01), (i12, i02),
             (i00, i02), (i10, i12)]
        )
        assert_allclose(lat1.subgraph_matrices['NN'], ref_NN)
        assert_allclose(lat2.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(6,
            [(i00, i11), (i01, i10), (i01, i12), (i02, i11),
             (i10, i01), (i11, i00), (i11, i02), (i12, i01),
             (i00, i12), (i10, i02),
             (i10, i02), (i00, i12)]
        )
        assert_allclose(lat1.subgraph_matrices['NNN'], ref_NNN)
        assert_allclose(lat2.subgraph_matrices['NNN'], ref_NNN)
