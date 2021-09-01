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
from itertools import product
from math import comb

from pycommute.expression import (n, a, a_dag, make_complex)
from pycommute.loperator import (
    make_space_fermion, make_space_boson,
    HilbertSpace,
    LOperatorR, LOperatorC,
    n_fermion_multisector_size,
    n_fermion_multisector_basis_states,
    NFermionMultiSectorViewR, NFermionMultiSectorViewC
)

import numpy as np
from numpy.testing import assert_allclose


# Compute the number of set bits in the given bit range of 'n'
def popcount(n, bit_range):
    count = 0
    for i in range(bit_range[1] + 1):
        if i >= bit_range[0]:
            count += n & 1
        n >>= 1
    return count


class TestNFermionMultiSector(TestCase):
    """N-fermion multisector view of a state vector"""

    M_total = 11

    N5_max = 1
    Na_max = 4
    Nb_max = 4

    def sde(self, N):
        return ([], N)

    def sd0(self, N):
        return ([(0,)], N)

    def sd5(self, N):
        return ([(5,)], N)

    def sda(self, N):
        return ([(1,), (2,), (6,), (7,)], N)

    def sdb(self, N):
        return ([(3,), (4,), (8,), (9,)], N)

    def sd5_index_selector(self, N):
        return lambda index: popcount(index, (5, 5)) == N

    def sdab_index_selector(self, N1, N2):
        return lambda index: \
            (popcount(index, (1, 2)) + popcount(index, (6, 7)) == N1) and \
            (popcount(index, (3, 4)) + popcount(index, (8, 9)) == N2)

    def sda5b_index_selector(self, N1, N2, N3):
        return lambda index: \
            (popcount(index, (1, 2)) + popcount(index, (6, 7)) == N1) and \
            (popcount(index, (5, 5)) == N2) and \
            (popcount(index, (3, 4)) + popcount(index, (8, 9)) == N3)

    ab_size_ref = [1, 4, 6, 4, 1]

    def wrong_indices_error_msg(self, indices):
        return f"Fermionic elementary space with indices {indices} " \
            "is not part of this Hilbert space"

    def wrong_n_error_msg(self, M, N):
        return f"^Sector with {N} fermions does not exist " \
               f"\\(there are {M} fermionic modes in it\\)$"

    def test_n_fermion_multisector_size(self):
        hs = HilbertSpace()

        #
        # Empty Hilbert space
        #

        self.assertEqual(n_fermion_multisector_size(hs, []), 0)
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(1)):
            n_fermion_multisector_size(hs, [self.sda(1)])

        #
        # Empty sectors
        #

        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(0)]), 0)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(1)]), 0)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(0),
                                                         self.sde(0)]), 0)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(1),
                                                         self.sde(1)]), 0)

        #
        # Purely fermionic Hilbert spaces
        #

        for i in range(self.M_total):
            hs.add(make_space_fermion(i))

        # Empty sectors
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(0)]), 2048)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(1)]), 0)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(0),
                                                         self.sde(0)]), 2048)
        self.assertEqual(n_fermion_multisector_size(hs, [self.sde(1),
                                                         self.sde(1)]), 0)

        with self.assertRaisesRegex(RuntimeError,
                                    "^Some of the sectors overlap$"):
            n_fermion_multisector_size(hs, [self.sda(2), ([(6,)], 1)])

        self.assertEqual(n_fermion_multisector_size(hs, []), 2048)

        # One sector
        for N in range(self.Na_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N)]),
                             self.ab_size_ref[N] * 128)
        for N in range(self.N5_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sd5(N)]),
                             1024)
        for N in range(self.Nb_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sdb(N)]),
                             self.ab_size_ref[N] * 128)

        # Two sectors
        for N1, N2 in product(range(self.Na_max + 1), range(self.Nb_max + 1)):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N1),
                                                             self.sdb(N2)]),
                             self.ab_size_ref[N1] * self.ab_size_ref[N2] * 8)

        # Three sectors
        for N1, N2, N3 in product(range(self.Na_max + 1),
                                  range(self.N5_max + 1),
                                  range(self.Nb_max + 1)):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N1),
                                                             self.sd5(N2),
                                                             self.sdb(N3)]),
                             self.ab_size_ref[N1] * self.ab_size_ref[N3] * 4)

        # Mixture with the empty sector
        for N in range(self.Na_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N),
                                                             self.sde(0)]),
                             self.ab_size_ref[N] * 128)
        for N in range(self.N5_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sd5(N),
                                                             self.sde(0)]),
                             1024)
        for N in range(self.Nb_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sdb(N),
                                                             self.sde(0)]),
                             self.ab_size_ref[N] * 128)

        #
        # Fermions and bosons
        #

        hs.add(make_space_boson(2, 0))

        self.assertEqual(n_fermion_multisector_size(hs, []), 8192)

        # One sector
        for N in range(self.Na_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N)]),
                             self.ab_size_ref[N] * 512)
        for N in range(self.N5_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sd5(N)]),
                             4096)
        for N in range(self.Nb_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sdb(N)]),
                             self.ab_size_ref[N] * 512)

        # Two sectors
        for N1, N2 in product(range(self.Na_max + 1), range(self.Nb_max + 1)):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N1),
                                                             self.sdb(N2)]),
                             self.ab_size_ref[N1] * self.ab_size_ref[N2] * 32)

        # Three sectors
        for N1, N2, N3 in product(range(self.Na_max + 1),
                                  range(self.N5_max + 1),
                                  range(self.Nb_max + 1)):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N1),
                                                             self.sd5(N2),
                                                             self.sdb(N3)]),
                             self.ab_size_ref[N1] * self.ab_size_ref[N3] * 16)

        # Mixture with the empty sector
        for N in range(self.Na_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sda(N),
                                                             self.sde(0)]),
                             self.ab_size_ref[N] * 512)
        for N in range(self.N5_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sd5(N),
                                                             self.sde(0)]),
                             4096)
        for N in range(self.Nb_max + 1):
            self.assertEqual(n_fermion_multisector_size(hs, [self.sdb(N),
                                                             self.sde(0)]),
                             self.ab_size_ref[N] * 512)

        #
        # Purely bosonic Hilbert space
        #

        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])
        self.assertEqual(n_fermion_multisector_size(hs_b, []), 32)
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(1)):
            n_fermion_multisector_size(hs_b, [self.sda(1)])

    def test_init_exceptions(self):
        st = np.array([], dtype=float)
        hs = HilbertSpace()

        #
        # Empty Hilbert space
        #

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(1)):
                ViewType(st, hs, [self.sda(0)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(1)):
                ViewType(st, hs, [self.sda(1)])

        #
        # One fermion
        #

        hs.add(make_space_fermion(0))

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_n_error_msg(1, 2)):
                ViewType(st, hs, [self.sd0(2)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(1)):
                ViewType(st, hs, [self.sda(0)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(1)):
                ViewType(st, hs, [self.sda(1)])

        #
        # Multiple fermions
        #

        for i in range(1, self.M_total):
            hs.add(make_space_fermion(i))

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            for N in range(self.Na_max):
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sde(0), self.sda(N), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sde(0), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sda(N), self.sde(0)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_n_error_msg(1, 2)):
                ViewType(st, hs, [self.sd0(2)])

        #
        # Fermions and bosons
        #

        hs.add(make_space_boson(1, 5))
        hs.add(make_space_boson(2, 6))

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            for N in range(self.Na_max):
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sde(0), self.sda(N), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sde(0), self.sda(N)])
                with self.assertRaisesRegex(RuntimeError,
                                            "^Some of the sectors overlap$"):
                    ViewType(st, hs, [self.sda(N), self.sda(N), self.sde(0)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_n_error_msg(1, 2)):
                ViewType(st, hs, [self.sd0(2)])

        #
        # Purely bosonic Hilbert space
        #

        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(0)):
                ViewType(st, hs_b, [self.sd0(0)])
            with self.assertRaisesRegex(RuntimeError,
                                        self.wrong_indices_error_msg(0)):
                ViewType(st, hs_b, [self.sd0(1)])

    def test_map_index(self):
        st = np.array([], dtype=float)

        # Check that values returned by map_index() form a continuous
        # sequence [0; expected_multisector_size)
        def check_map_index(view, hs, selector, expected_multisector_size):
            mapped_indices = []
            for index in range(hs.dim):
                if selector(index):
                    mapped_indices.append(view.map_index(index))
            mapped_indices.sort()
            mapped_indices_ref = list(range(expected_multisector_size))
            self.assertEqual(mapped_indices, mapped_indices_ref)

        #
        # Purely fermionic Hilbert spaces
        #

        hs = HilbertSpace([make_space_fermion(i) for i in range(self.M_total)])

        for ViewType in (NFermionMultiSectorViewR,
                         NFermionMultiSectorViewC):
            for N in range(self.N5_max + 1):
                check_map_index(ViewType(st, hs, [self.sd5(N)]),
                                hs,
                                self.sd5_index_selector(N),
                                2 ** (self.M_total - 1))

            for N1, N2 in product(range(self.Na_max + 1),
                                  range(self.Nb_max + 1)):
                check_map_index(ViewType(st, hs, [self.sda(N1), self.sdb(N2)]),
                                hs,
                                self.sdab_index_selector(N1, N2),
                                comb(4, N1) * comb(4, N2)
                                * (2 ** (self.M_total - 8))
                                )

            for N1, N2, N3 in product(range(self.Na_max + 1),
                                      range(self.N5_max + 1),
                                      range(self.Nb_max + 1)):
                check_map_index(ViewType(st, hs, [self.sda(N1),
                                                  self.sd5(N2),
                                                  self.sdb(N3)]),
                                hs,
                                self.sda5b_index_selector(N1, N2, N3),
                                comb(4, N1) * comb(4, N3)
                                * (2 ** (self.M_total - 9))
                                )

        #
        # Fermions and bosons
        #

        hs.add(make_space_boson(2, self.M_total))
        hs.add(make_space_boson(2, self.M_total + 1))

        M = self.M_total + 4

        for ViewType in (NFermionMultiSectorViewR,
                         NFermionMultiSectorViewC):
            for N in range(self.N5_max + 1):
                check_map_index(ViewType(st, hs, [self.sd5(N)]),
                                hs,
                                self.sd5_index_selector(N),
                                2 ** (M - 1))

            for N1, N2 in product(range(self.Na_max + 1),
                                  range(self.Nb_max + 1)):
                check_map_index(ViewType(st, hs, [self.sda(N1), self.sdb(N2)]),
                                hs,
                                self.sdab_index_selector(N1, N2),
                                comb(4, N1) * comb(4, N2) * (2 ** (M - 8))
                                )

            for N1, N2, N3 in product(range(self.Na_max + 1),
                                      range(self.N5_max + 1),
                                      range(self.Nb_max + 1)):
                check_map_index(ViewType(st, hs, [self.sda(N1),
                                                  self.sd5(N2),
                                                  self.sdb(N3)]),
                                hs,
                                self.sda5b_index_selector(N1, N2, N3),
                                comb(4, N1) * comb(4, N3) * (2 ** (M - 9))
                                )

        #
        # Purely bosonic Hilbert space
        #

        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        for ViewType in (NFermionMultiSectorViewR, NFermionMultiSectorViewC):
            view = ViewType(st, hs_b, [self.sde(0)])
            for index in range(hs_b.dim):
                self.assertEqual(view.map_index(index), index)

    def test_n_fermion_sector_basis_states(self):
        st = np.array([], dtype=float)

        # Build a reference list of basis states for
        # n_fermion_multisector_basis_states() by repeatedly calling map_index()
        def build_basis_states_ref(hs, sectors, selector):
            view = NFermionMultiSectorViewR(st, hs, sectors)
            basis_states = [-1] * n_fermion_multisector_size(hs, sectors)
            for index in range(hs.dim):
                if selector(index):
                    basis_states[view.map_index(index)] = index
            return basis_states

        hs = HilbertSpace()

        #
        # Empty Hilbert space
        #

        self.assertEqual(n_fermion_multisector_basis_states(hs, []), [])
        self.assertEqual(n_fermion_multisector_basis_states(hs, [self.sde(0)]),
                         [])

        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs, [self.sde(1)])

        self.assertEqual(n_fermion_multisector_basis_states(hs, [self.sde(0),
                                                                 self.sde(0)]),
                         [])

        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs, [self.sde(0), self.sde(1)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs, [self.sde(1), self.sde(0)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs, [self.sde(1), self.sde(1)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(0)):
            n_fermion_multisector_basis_states(hs, [self.sd0(0)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(0)):
            n_fermion_multisector_basis_states(hs, [self.sd0(1)])

        #
        # Purely fermionic Hilbert spaces
        #

        for i in range(self.M_total):
            hs.add(make_space_fermion(i))

        for N in range(self.N5_max + 1):
            sectors = [self.sd5(N)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sd5_index_selector(N))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        for N1, N2 in product(range(self.Na_max + 1),
                              range(self.Nb_max + 1)):
            sectors = [self.sda(N1), self.sdb(N2)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sdab_index_selector(N1, N2))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        for N1, N2, N3 in product(range(self.Na_max + 1),
                                  range(self.N5_max + 1),
                                  range(self.Nb_max + 1)):
            sectors = [self.sda(N1), self.sd5(N2), self.sdb(N3)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sda5b_index_selector(N1, N2, N3))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        #
        # Fermions and bosons
        #

        hs.add(make_space_boson(2, self.M_total))
        hs.add(make_space_boson(2, self.M_total + 1))

        for N in range(self.N5_max + 1):
            sectors = [self.sd5(N)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sd5_index_selector(N))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        for N1, N2 in product(range(self.Na_max + 1),
                              range(self.Nb_max + 1)):
            sectors = [self.sda(N1), self.sdb(N2)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sdab_index_selector(N1, N2))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        for N1, N2, N3 in product(range(self.Na_max + 1),
                                  range(self.N5_max + 1),
                                  range(self.Nb_max + 1)):
            sectors = [self.sda(N1), self.sd5(N2), self.sdb(N3)]
            ref = build_basis_states_ref(hs,
                                         sectors,
                                         self.sda5b_index_selector(N1, N2, N3))
            self.assertEqual(
                n_fermion_multisector_basis_states(hs, sectors),
                ref
            )

        #
        # Purely bosonic Hilbert space
        #

        hs_b = HilbertSpace([make_space_boson(2, 0), make_space_boson(3, 1)])

        ref = list(range(n_fermion_multisector_size(hs_b, [])))

        self.assertEqual(n_fermion_multisector_basis_states(hs_b, []), ref)
        self.assertEqual(n_fermion_multisector_basis_states(hs_b,
                                                            [self.sde(0)]),
                         ref)

        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs_b, [self.sde(1)])

        self.assertEqual(n_fermion_multisector_basis_states(hs_b, [self.sde(0),
                                                                   self.sde(0)]
                                                            ),
                         ref)

        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs_b, [self.sde(0), self.sde(1)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs_b, [self.sde(1), self.sde(0)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_n_error_msg(0, 1)):
            n_fermion_multisector_basis_states(hs_b, [self.sde(1), self.sde(1)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(0)):
            n_fermion_multisector_basis_states(hs_b, [self.sd0(0)])
        with self.assertRaisesRegex(RuntimeError,
                                    self.wrong_indices_error_msg(0)):
            n_fermion_multisector_basis_states(hs_b, [self.sd0(1)])

    def test_loperator(self):
        hs = HilbertSpace([make_space_boson(2, 0)])
        for i in range(self.M_total):
            hs.add(make_space_fermion(i))

        sector_a_modes = [1, 2, 6, 7]
        sector_b_modes = [3, 4, 8, 9]

        Ha = (n(1) + n(2) + n(6) + n(7)) * (a_dag(0) + a(0))
        Hb = (n(3) + n(4) + n(8) + n(9)) * (a_dag(0) + a(0))

        for src_type, dst_type, lop_type in [(float, float, LOperatorR),
                                             (float, complex, LOperatorR),
                                             (complex, complex, LOperatorR),
                                             (float, complex, LOperatorC),
                                             (complex, complex, LOperatorC)]:
            src_view_type = NFermionMultiSectorViewR if (src_type == float) \
                else NFermionMultiSectorViewC
            dst_view_type = NFermionMultiSectorViewR if (dst_type == float) \
                else NFermionMultiSectorViewC

            lop_is_real = lop_type == LOperatorR

            Hopa = lop_type(Ha if lop_is_real else make_complex(Ha), hs)
            Hopb = lop_type(Hb if lop_is_real else make_complex(Hb), hs)

            for N1, N2 in product(range(self.Na_max + 1),
                                  range(self.Nb_max + 1)):
                sectors = [self.sda(N1), self.sdb(N2)]

                src = np.zeros(n_fermion_multisector_size(hs, sectors),
                               dtype=src_type)
                view_src = src_view_type(src, hs, sectors)
                dst = np.zeros(n_fermion_multisector_size(hs, sectors),
                               dtype=dst_type)
                view_dst = dst_view_type(dst, hs, sectors)

                # Input:
                # First N1 modes of sector A are occupied
                # First N2 modes of sector B are occupied
                # 1 boson

                index_in_f = sum(2 ** sector_a_modes[n1] for n1 in range(N1))
                index_in_f += sum(2 ** sector_b_modes[n2] for n2 in range(N2))

                src[view_src.map_index(index_in_f + (2 ** self.M_total))] = 1

                Hopa(view_src, view_dst)
                ref = np.zeros(n_fermion_multisector_size(hs, sectors),
                               dtype=dst_type)
                # 0 bosons
                ref[view_dst.map_index(index_in_f)] = N1
                # 2 bosons
                ref[view_dst.map_index(index_in_f + (2 ** (self.M_total + 1)))]\
                    = N1 * np.sqrt(2)
                assert_allclose(dst, ref)

                Hopb(view_src, view_dst)
                ref = np.zeros(n_fermion_multisector_size(hs, sectors),
                               dtype=dst_type)
                # 0 bosons
                ref[view_dst.map_index(index_in_f)] = N2
                # 2 bosons
                ref[view_dst.map_index(index_in_f + (2 ** (self.M_total + 1)))]\
                    = N2 * np.sqrt(2)
                assert_allclose(dst, ref)
