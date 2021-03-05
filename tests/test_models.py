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
    hypercubic_lattice,
    chain,
    square_lattice,
    cubic_lattice,
    triangular_lattice
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

    def make_adj_matrix(self, mat, edges):
        if isinstance(mat, int):
            matrix = np.zeros((mat, mat), dtype=int)
        else:
            matrix = mat.copy()

        for e in edges:
            matrix[e[0], e[1]] += 1
            matrix[e[1], e[0]] += 1
        return matrix

    def index_translator(self, indices):
        return tuple("abc"[d] + str(i) for d, i in enumerate(indices))

    def test_hypercubic_lattice(self):
        shape = (2, 3)

        lat = hypercubic_lattice(shape, self.index_translator)
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
        ref_NN = self.make_adj_matrix(ref_NN, [(i00, i02), (i10, i12)])
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(ref_NNN, [(i00, i12), (i10, i02)])
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
        ref_NN = self.make_adj_matrix(ref_NN, [(i00, i02), (i10, i12)])
        assert_allclose(lat1.subgraph_matrices['NN'], ref_NN)
        assert_allclose(lat2.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(ref_NNN,
            [(i00, i12), (i10, i02), (i10, i02), (i00, i12)]
        )
        assert_allclose(lat1.subgraph_matrices['NNN'], ref_NNN)
        assert_allclose(lat2.subgraph_matrices['NNN'], ref_NNN)

        assert_allclose(
            lat1.adjacency_matrix({'NN': 2, 'NNN': 3}),
            2 * ref_NN + 3 * ref_NNN
        )

    def test_chain(self):
        c = chain(4, lambda i: "a%d" % i)
        self.assertEqual(c.node_names, [("a0",), ("a1",), ("a2",), ("a3",)])

        self.assertEqual(len(c.subgraph_matrices), 1)
        assert_allclose(
            c.subgraph_matrices['NN'],
            hypercubic_lattice((4,), periodic=True).subgraph_matrices['NN']
        )

        c = chain(4, lambda i: "a%d" % i, periodic=False)
        self.assertEqual(len(c.subgraph_matrices), 1)
        assert_allclose(
            c.subgraph_matrices['NN'],
            hypercubic_lattice((4,), periodic=False).subgraph_matrices['NN']
        )

    def test_square_lattice(self):
        lat = square_lattice(2, 2, lambda ij: ("a%d" % ij[0], "b%d" % ij[1]))
        self.assertEqual(
            lat.node_names,
            [("a0", "b0"), ("a0", "b1"), ("a1", "b0"), ("a1", "b1")]
        )

        lat = square_lattice(4, 4)
        self.assertEqual(len(lat.subgraph_matrices), 2)
        for sg in ('NN', 'NNN'):
            assert_allclose(
              lat.subgraph_matrices[sg],
              hypercubic_lattice((4, 4), periodic=True).subgraph_matrices[sg]
            )
        lat = square_lattice(4, 4, periodic=False)
        self.assertEqual(len(lat.subgraph_matrices), 2)
        for sg in ('NN', 'NNN'):
            assert_allclose(
              lat.subgraph_matrices[sg],
              hypercubic_lattice((4, 4), periodic=False).subgraph_matrices[sg]
            )

    def test_cubic_lattice(self):
        lat = cubic_lattice(
            2, 2, 2,
            lambda ijk: ("a%d" % ijk[0], "b%d" % ijk[1], "c%d" % ijk[2])
        )
        self.assertEqual(
            lat.node_names,
            [("a0", "b0", "c0"), ("a0", "b0", "c1"),
             ("a0", "b1", "c0"), ("a0", "b1", "c1"),
             ("a1", "b0", "c0"), ("a1", "b0", "c1"),
             ("a1", "b1", "c0"), ("a1", "b1", "c1")]
        )

        lat = cubic_lattice(4, 4, 4)
        self.assertEqual(len(lat.subgraph_matrices), 3)
        for sg in ('NN', 'NNN', 'NNNN'):
            assert_allclose(
              lat.subgraph_matrices[sg],
              hypercubic_lattice((4, 4, 4), periodic=True).subgraph_matrices[sg]
            )
        lat = cubic_lattice(4, 4, 4, periodic=False)
        self.assertEqual(len(lat.subgraph_matrices), 3)
        for sg in ('NN', 'NNN', 'NNNN'):
            assert_allclose(
              lat.subgraph_matrices[sg],
              hypercubic_lattice((4, 4, 4), periodic=False).subgraph_matrices[sg]
            )

    def test_triangular_lattice(self):
        # triangle
        lat = triangular_lattice(
            "triangle",
            lambda ij: ("a%d" % ij[0], "b%d" % ij[1]),
            l = 3
        )
        self.assertEqual(
            lat.node_names,
            [("a0", "b0"), ("a0", "b1"), ("a0", "b2"), ("a0", "b3"),
             ("a1", "b0"), ("a1", "b1"), ("a1", "b2"),
             ("a2", "b0"), ("a2", "b1"),
             ("a3", "b0")]
        )

        lat = triangular_lattice("triangle", l = 3)
        self.assertEqual(len(lat.subgraph_matrices), 2)

        i00, i01, i02, i03, i10, i11, i12, i20, i21, i30 = range(10)
        ref_NN = self.make_adj_matrix(10,
            [(i00, i10), (i10, i20), (i20, i30),
             (i01, i11), (i11, i21), (i02, i12),
             (i00, i01), (i01, i10), (i10, i11),
             (i11, i20), (i20, i21), (i21, i30),
             (i01, i02), (i02, i11), (i11, i12), (i12, i21),
             (i02, i03), (i03, i12)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(10,
            [(i11, i00), (i11, i30), (i11, i03),
             (i01, i12), (i12, i20), (i20, i01),
             (i02, i21), (i21, i10), (i10, i02)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        # parallelogram
        lat = triangular_lattice(
            "parallelogram",
            lambda ij: ("a%d" % ij[0], "b%d" % ij[1]),
            l=2,
            m=1
        )
        self.assertEqual(
            lat.node_names,
            [("a0", "b0"), ("a0", "b1"),
             ("a1", "b0"), ("a1", "b1"),
             ("a2", "b0"), ("a2", "b1")]
        )

        lat = triangular_lattice("parallelogram", l=2, m=1, periodic=False)
        self.assertEqual(len(lat.subgraph_matrices), 2)

        i00, i01, i10, i11, i20, i21 = range(6)
        ref_NN = self.make_adj_matrix(6,
            [(i00, i01), (i10, i11), (i20, i21),
             (i00, i10), (i10, i20), (i01, i11), (i11, i21),
             (i01, i10), (i11, i20)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(6,
            [(i00, i11), (i01, i20), (i10, i21)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        lat = triangular_lattice("parallelogram", l=2, m=1, periodic=True)
        self.assertEqual(len(lat.subgraph_matrices), 2)

        i00, i01, i10, i11, i20, i21 = range(6)
        ref_NN = self.make_adj_matrix(ref_NN,
            [(i00, i01), (i00, i11), (i10, i11), (i10, i21),
             (i20, i21), (i20, i01),
             (i00, i20), (i01, i21), (i00, i21)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(ref_NNN,
            [(i00, i21), (i00, i20), (i00, i21), (i00, i10), (i00, i11),
             (i01, i10), (i01, i21), (i01, i10), (i01, i20), (i01, i11),
             (i11, i20), (i11, i20), (i11, i21),
             (i10, i20), (i10, i21)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        # hexagon
        lat = triangular_lattice(
            "hexagon",
            lambda ij: ("a%d" % ij[0], "b%d" % ij[1]),
            l=3,
            periodic=False
        )
        self.assertEqual(
            lat.node_names,
            [("a0", "b2"), ("a0", "b3"), ("a0", "b4"), ("a0", "b5"),
             ("a1", "b1"), ("a1", "b2"), ("a1", "b3"),
             ("a1", "b4"), ("a1", "b5"),
             ("a2", "b0"), ("a2", "b1"), ("a2", "b2"),
             ("a2", "b3"), ("a2", "b4"), ("a2", "b5"),
             ("a3", "b0"), ("a3", "b1"), ("a3", "b2"), ("a3", "b3"),
             ("a3", "b4"),
             ("a4", "b0"), ("a4", "b1"), ("a4", "b2"), ("a4", "b3"),
             ("a5", "b0"), ("a5", "b1"), ("a5", "b2")
             ]
        )

        lat = triangular_lattice("hexagon", l=2, periodic=False)
        self.assertEqual(
            lat.node_names,
            [(0, 1), (0, 2), (0, 3),
             (1, 0), (1, 1), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 2),
             (3, 0), (3, 1)
             ]
        )
        self.assertEqual(len(lat.subgraph_matrices), 2)

        i01, i02, i03, i10, i11, i12, i13, i20, i21, i22, i30, i31 = range(12)
        ref_NN = self.make_adj_matrix(12,
            [(i01, i10), (i01, i11), (i01, i02),
             (i02, i11), (i02, i12), (i02, i03),
             (i03, i12), (i03, i13),
             (i10, i20), (i10, i11),
             (i11, i20), (i11, i21), (i11, i12),
             (i12, i21), (i12, i22), (i12, i13),
             (i13, i22),
             (i20, i30), (i20, i21),
             (i21, i30), (i21, i31), (i21, i22),
             (i22, i31),
             (i30, i31)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)
        ref_NNN = self.make_adj_matrix(12,
            [(i10, i02), (i11, i03), (i20, i12), (i21, i13), (i30, i22),
             (i01, i20), (i11, i30), (i02, i21), (i12, i31), (i03, i22),
             (i01, i12), (i02, i13), (i10, i21), (i20, i31), (i11, i22)]
        )
        assert_allclose(lat.subgraph_matrices['NNN'], ref_NNN)

        lat = triangular_lattice("hexagon", l=2)
        self.assertEqual(len(lat.subgraph_matrices), 2)

        ref_NN = self.make_adj_matrix(ref_NN,
            [(i01, i30), (i01, i13), (i01, i22),
             (i10, i22), (i10, i31), (i10, i03),
             (i20, i03), (i20, i13),
             (i30, i13), (i30, i02),
             (i31, i02), (i31, i03)]
        )
        assert_allclose(lat.subgraph_matrices['NN'], ref_NN)

        # TODO: ref_NNN

