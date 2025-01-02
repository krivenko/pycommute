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

from pycommute.expression import (n, a, a_dag, make_complex)
from pycommute.loperator import (
    make_space_fermion, make_space_boson,
    HilbertSpace,
    LOperatorR, LOperatorC,
    n_fermion_sector_size,
    n_fermion_sector_basis_states,
    NFermionSectorViewR, NFermionSectorViewC
)

import numpy as np
from numpy.testing import assert_allclose


# Compute the number of set bits in 'n' among the first 'M' bits
def popcount(n, M):
    count = 0
    for i in range(M):
        count += n & 1
        n >>= 1
    return count


class TestNFermionSector(TestCase):
    """N-fermion sector view of a state vector"""

    def wrong_n_error_msg(self, M, N):
        return f"^Sector with {N} fermions does not exist " \
               f"\\(there are {M} fermionic modes in it\\)$"

    def test_n_fermion_sector_size(self):
        hs = HilbertSpace()

        # Empty Hilbert space
        self.assertEqual(n_fermion_sector_size(hs, 0), 0)
        self.assertEqual(n_fermion_sector_size(hs, 1), 0)

        # Purely fermionic Hilbert spaces
        hs.add(make_space_fermion(0))
        self.assertEqual(n_fermion_sector_size(hs, 0), 1)
        self.assertEqual(n_fermion_sector_size(hs, 1), 1)
        hs.add(make_space_fermion(1))
        self.assertEqual(n_fermion_sector_size(hs, 0), 1)
        self.assertEqual(n_fermion_sector_size(hs, 1), 2)
        self.assertEqual(n_fermion_sector_size(hs, 2), 1)
        hs.add(make_space_fermion(2))
        self.assertEqual(n_fermion_sector_size(hs, 0), 1)
        self.assertEqual(n_fermion_sector_size(hs, 1), 3)
        self.assertEqual(n_fermion_sector_size(hs, 2), 3)
        self.assertEqual(n_fermion_sector_size(hs, 3), 1)

        # Fermions and bosons
        hs.add(make_space_boson(4, 3))
        self.assertEqual(n_fermion_sector_size(hs, 0), 16)
        self.assertEqual(n_fermion_sector_size(hs, 1), 48)
        self.assertEqual(n_fermion_sector_size(hs, 2), 48)
        self.assertEqual(n_fermion_sector_size(hs, 3), 16)

        # Purely bosonic Hilbert space
        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])
        self.assertEqual(n_fermion_sector_size(hs_b, 0), 32)
        self.assertEqual(n_fermion_sector_size(hs_b, 1), 0)

    def test_init_exceptions(self):
        M = 5
        st = np.array([], dtype=float)
        hs = HilbertSpace([make_space_fermion(i) for i in range(M)])

        # Multiple fermions
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(M, M + 1)):
            NFermionSectorViewR(st, hs, M + 1)
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(M, M + 1)):
            NFermionSectorViewC(st, hs, M + 1)

        # Fermions and bosons
        hs.add(make_space_boson(2, 5))
        hs.add(make_space_boson(3, 6))

        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(M, M + 1)):
            NFermionSectorViewR(st, hs, M + 1)
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(M, M + 1)):
            NFermionSectorViewC(st, hs, M + 1)

        # Purely bosonic Hilbert space
        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        with self.assertRaisesRegex(RuntimeError, self.wrong_n_error_msg(0, 1)):
            NFermionSectorViewR(st, hs_b, 1)
        with self.assertRaisesRegex(RuntimeError, self.wrong_n_error_msg(0, 1)):
            NFermionSectorViewC(st, hs_b, 1)

    def test_map_index(self):
        M = 8
        st = np.array([], dtype=float)

        hs = HilbertSpace()

        def check_map_index(view, m, n):
            mapped_indices = []
            for index in range(hs.dim):
                if popcount(index, m) == n:
                    mapped_indices.append(view.map_index(index))
            mapped_indices.sort()
            mapped_indices_ref = list(range(n_fermion_sector_size(hs, n)))
            self.assertEqual(mapped_indices, mapped_indices_ref)

        # Purely fermionic Hilbert spaces
        for i in range(M):
            hs.add(make_space_fermion(i))

        for N in range(M + 1):
            check_map_index(NFermionSectorViewR(st, hs, N), M, N)
            check_map_index(NFermionSectorViewC(st, hs, N), M, N)

        # Fermions and bosons
        hs.add(make_space_boson(2, M))
        hs.add(make_space_boson(2, M + 1))

        for N in range(M + 1):
            check_map_index(NFermionSectorViewR(st, hs, N), M, N)
            check_map_index(NFermionSectorViewC(st, hs, N), M, N)

        # Purely bosonic Hilbert space
        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        ref_indices = list(range(hs_b.dim))

        view = NFermionSectorViewR(st, hs_b, 0)
        indices = [view.map_index(index) for index in range(hs_b.dim)]
        self.assertEqual(indices, ref_indices)

        view = NFermionSectorViewC(st, hs_b, 0)
        indices = [view.map_index(index) for index in range(hs_b.dim)]
        self.assertEqual(indices, ref_indices)

    def test_n_fermion_sector_basis_states(self):
        M = 8
        st = np.array([], dtype=float)

        hs = HilbertSpace()

        self.assertEqual(n_fermion_sector_basis_states(hs, 0), [])

        with self.assertRaisesRegex(RuntimeError, self.wrong_n_error_msg(0, 1)):
            n_fermion_sector_basis_states(hs, 1)

        def build_basis_states_ref(hs, N):
            view = NFermionSectorViewR(st, hs, N)
            basis_states = [-1] * n_fermion_sector_size(hs, N)
            for index in range(hs.dim):
                if popcount(index, M) == N:
                    basis_states[view.map_index(index)] = index
            return basis_states

        # Purely fermionic Hilbert spaces
        for i in range(M):
            hs.add(make_space_fermion(i))

        for N in range(M + 1):
            ref = build_basis_states_ref(hs, N)
            self.assertEqual(n_fermion_sector_basis_states(hs, N), ref)

        # Fermions and bosons
        hs.add(make_space_boson(2, M))
        hs.add(make_space_boson(2, M + 1))

        for N in range(M + 1):
            ref = build_basis_states_ref(hs, N)
            self.assertEqual(n_fermion_sector_basis_states(hs, N), ref)

        # Purely bosonic Hilbert space
        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        with self.assertRaisesRegex(RuntimeError, self.wrong_n_error_msg(0, 1)):
            n_fermion_sector_basis_states(hs_b, 1)

        ref = list(range(n_fermion_sector_size(hs_b, 0)))
        self.assertEqual(n_fermion_sector_basis_states(hs_b, 0), ref)

    def test_loperator(self):
        M = 5

        hs = HilbertSpace([make_space_boson(2, 0)])
        for i in range(M):
            hs.add(make_space_fermion(i))

        H = (n(0) + n(1) + n(2) + n(3) + n(4)) * (a_dag(0) + a(0))

        for src_type, dst_type, lop_type in [(float, float, LOperatorR),
                                             (float, complex, LOperatorR),
                                             (complex, complex, LOperatorR),
                                             (float, complex, LOperatorC),
                                             (complex, complex, LOperatorC)]:
            src_view_type = NFermionSectorViewR if (src_type == float) \
                else NFermionSectorViewC
            dst_view_type = NFermionSectorViewR if (dst_type == float) \
                else NFermionSectorViewC

            Hop = lop_type(H if lop_type == LOperatorR else make_complex(H), hs)

            for N in range(M + 1):
                src = np.zeros(n_fermion_sector_size(hs, N), dtype=src_type)
                view_src = src_view_type(src, hs, N)
                dst = np.zeros(n_fermion_sector_size(hs, N), dtype=dst_type)
                view_dst = dst_view_type(dst, hs, N)

                # 1 boson, fermions in the first N modes
                index_in_f = sum(2 ** i for i in range(N))

                src[view_src.map_index(index_in_f + (2 ** M))] = 1

                Hop(view_src, view_dst)

                ref = np.zeros(n_fermion_sector_size(hs, N), dtype=dst_type)
                # 0 bosons
                ref[view_dst.map_index(index_in_f)] = N
                # 2 bosons
                ref[view_dst.map_index(index_in_f + (2 ** (M + 1)))] = \
                    N * np.sqrt(2)
                assert_allclose(dst, ref)
