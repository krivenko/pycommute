#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2025 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from pycommute.expression import (Indices, a_dag)
from pycommute.loperator import (
    ESpaceFermion,
    ESpaceBoson,
    HilbertSpace,
    LOperatorR,
    CompressedStateViewR,
    foreach
)

import numpy as np
from numpy.testing import assert_equal, assert_allclose


class TestCompressedStateView(TestCase):
    """View of a compressed state vector"""

    def test_map_index(self):
        # Dense Hilbert space
        hs = HilbertSpace([ESpaceFermion(Indices(0)),
                           ESpaceFermion(Indices(1))])

        st = np.zeros((4,), dtype=float)
        view = CompressedStateViewR(st, hs)
        for index in range(hs.vec_size):
            self.assertEqual(view.map_index(index), index)

        # Sparse Hilbert space
        hs = HilbertSpace([ESpaceFermion(Indices(0)),
                           ESpaceFermion(Indices(1)),
                           ESpaceBoson(3, Indices(2)),
                           ESpaceBoson(4, Indices(3)),
                           ESpaceBoson(4, Indices(4)),
                           ESpaceBoson(3, Indices(5)),
                           ESpaceBoson(3, Indices(6))])

        st = np.zeros((hs.dim,), dtype=float)
        view = CompressedStateViewR(st, hs)
        serial_indices = []
        foreach(hs, lambda index: serial_indices.append(view.map_index(index)))
        assert_equal(serial_indices, list(range(hs.dim)))

    def test_loperator(self):
        hs = HilbertSpace([ESpaceFermion(Indices(0)),
                           ESpaceFermion(Indices(1)),
                           ESpaceBoson(3, Indices(1)),
                           ESpaceBoson(3, Indices(2)),
                           ESpaceBoson(4, Indices(3))])

        src = np.zeros((hs.dim,), dtype=float)
        dst = np.zeros((hs.dim,), dtype=float)
        dst_ref = np.zeros((hs.dim,), dtype=float)

        src_view = CompressedStateViewR(src, hs)
        dst_view = CompressedStateViewR(dst, hs)

        b1op = LOperatorR(a_dag(1), hs)
        b2op = LOperatorR(a_dag(2), hs)

        # src = |0>|0>|0>|0>
        src[:] = 0
        src[dst_view.map_index(0)] = 1

        # dst = |0>|1>|0>|0>
        b1op(src_view, dst_view)
        dst_ref[:] = 0
        dst_ref[dst_view.map_index(1 << 2)] = 1
        assert_allclose(dst, dst_ref)

        # dst = |0>|1>|1>|0>
        src[:] = dst
        b2op(src_view, dst_view)
        dst_ref[:] = 0
        dst_ref[dst_view.map_index((1 << 2) + (1 << 4))] = 1
        assert_allclose(dst, dst_ref)

        # dst = |0>|2>|1>|0>
        src[:] = dst
        b1op(src_view, dst_view)
        dst_ref[:] = 0
        dst_ref[dst_view.map_index((2 << 2) + (1 << 4))] = np.sqrt(2)
        assert_allclose(dst, dst_ref)

        # dst = |0>|2>|2>|0>
        src[:] = dst
        b2op(src_view, dst_view)
        dst_ref[:] = 0
        dst_ref[dst_view.map_index((2 << 2) + (2 << 4))] = 2
        assert_allclose(dst, dst_ref)

        # src = |0>|0>|1>|0>
        # dst = |0>|0>|2>|0>
        src[:] = 0
        src[dst_view.map_index(1 << 4)] = 1
        b2op(src_view, dst_view)
        dst_ref[:] = 0
        dst_ref[dst_view.map_index(2 << 4)] = np.sqrt(2)
        assert_allclose(dst, dst_ref)
