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

from pycommute.expression import (ExpressionR, c, c_dag, S_p, S_m, S_z)
from pycommute.loperator import (
    HilbertSpace,
    LOperatorR,
    LOperatorC,
    make_space_fermion,
    make_space_spin,
    make_matrix
)

import numpy as np
from numpy.testing import assert_equal
from itertools import product


class TestMakeMatrix(TestCase):

    def c_mat(self, hs, *args):
        n_fermions = len(hs)
        op_index = hs.index(make_space_fermion(*args))
        states = list(product((0, 1), repeat=n_fermions))
        states = [st[::-1] for st in states]
        mat = np.zeros((hs.dim, hs.dim))
        for m, n in product(range(hs.dim), range(hs.dim)):
            if states[m] == tuple(
                    (states[n][i] - 1 if i == op_index else states[n][i])
                    for i in range(n_fermions)):
                mat[m, n] = (-1) ** sum(states[n][:op_index])
        return mat

    def c_dag_mat(self, *args):
        return np.conj(np.transpose(self.c_mat(*args)))

    def S_z_mat(self, hs, *args):
        n_spins = len(hs)
        op_index = hs.index(make_space_spin(1, *args))
        states = list(product((-1, 0, 1), repeat=n_spins))
        states = [st[::-1] for st in states]
        mat = np.zeros((hs.dim, hs.dim))
        for m, n in product(range(hs.dim), range(hs.dim)):
            if m == n:
                mat[m, n] = states[n][op_index]
        return mat

    def S_p_mat(self, hs, *args):
        n_spins = len(hs)
        op_index = hs.index(make_space_spin(1, *args))
        states = list(product((-1, 0, 1), repeat=n_spins))
        states = [st[::-1] for st in states]
        mat = np.zeros((hs.dim, hs.dim))
        for m, n in product(range(hs.dim), range(hs.dim)):
            if states[m] == tuple(
                    (states[n][i] + 1 if i == op_index else states[n][i])
                    for i in range(n_spins)):
                mat[m, n] = np.sqrt(2)
        return mat

    def S_m_mat(self, hs, *args):
        return np.conj(np.transpose(self.S_p_mat(hs, *args)))

    def test_HilbertSpace(self):
        indices = [("dn", 0), ("dn", 1), ("up", 0), ("up", 1)]
        hs = HilbertSpace([make_space_fermion(*ind) for ind in indices])

        for ind in indices:
            c_op = LOperatorR(c(*ind), hs)
            assert_equal(make_matrix(c_op, hs), self.c_mat(hs, *ind))
            c_dag_op = LOperatorR(c_dag(*ind), hs)
            assert_equal(make_matrix(c_dag_op, hs), self.c_dag_mat(hs, *ind))

        H1 = 1.0 * (c_dag("up", 0) * c("up", 1) + c_dag("up", 1) * c("up", 0))
        H1 += 2.0 * (c_dag("dn", 0) * c("dn", 1) + c_dag("dn", 1) * c("dn", 0))
        H1op = LOperatorR(H1, hs)
        ref1 = 1.0 * (self.c_dag_mat(hs, "up", 0) @ self.c_mat(hs, "up", 1)
                      + self.c_dag_mat(hs, "up", 1) @ self.c_mat(hs, "up", 0))
        ref1 += 2.0 * (self.c_dag_mat(hs, "dn", 0) @ self.c_mat(hs, "dn", 1)
                       + self.c_dag_mat(hs, "dn", 1) @ self.c_mat(hs, "dn", 0))
        assert_equal(make_matrix(H1op, hs), ref1)

        H2 = 1.0j * (c_dag("up", 0) * c("up", 1) + c_dag("up", 1) * c("up", 0))
        H2 += 2.0 * (c_dag("dn", 0) * c("dn", 1) + c_dag("dn", 1) * c("dn", 0))
        H2op = LOperatorC(H2, hs)
        ref2 = 1.0j * (self.c_dag_mat(hs, "up", 0) @ self.c_mat(hs, "up", 1)
                       + self.c_dag_mat(hs, "up", 1) @ self.c_mat(hs, "up", 0))
        ref2 += 2.0 * (self.c_dag_mat(hs, "dn", 0) @ self.c_mat(hs, "dn", 1)
                       + self.c_dag_mat(hs, "dn", 1) @ self.c_mat(hs, "dn", 0))
        assert_equal(make_matrix(H2op, hs), ref2)

    def test_sparse_HilbertSpace(self):
        indices = list(range(4))
        hs = HilbertSpace([make_space_spin(1, i) for i in indices])
        self.assertTrue(hs.is_sparse)

        for i in indices:
            S_z_op = LOperatorR(S_z(i, spin=1), hs)
            assert_equal(make_matrix(S_z_op, hs), self.S_z_mat(hs, i))
            S_p_op = LOperatorR(S_p(i, spin=1), hs)
            assert_equal(make_matrix(S_p_op, hs), self.S_p_mat(hs, i))
            S_m_op = LOperatorR(S_m(i, spin=1), hs)
            assert_equal(make_matrix(S_m_op, hs), self.S_m_mat(hs, i))

        # Heisenberg plaquette 2x2
        H = ExpressionR()
        Hmat_ref = np.zeros((hs.dim, hs.dim))
        for i in range(4):
            j = (i + 1) % 4
            H += S_z(i, spin=1) * S_z(j, spin=1) \
                + 0.5 * (S_p(i, spin=1) * S_m(j, spin=1)
                         + S_m(i, spin=1) * S_p(j, spin=1))
            Hmat_ref += self.S_z_mat(hs, i) @ self.S_z_mat(hs, j) \
                + 0.5 * (self.S_p_mat(hs, i) @ self.S_m_mat(hs, j)
                         + self.S_m_mat(hs, i) @ self.S_p_mat(hs, j))
        Hop = LOperatorR(H, hs)
        assert_equal(make_matrix(Hop, hs), Hmat_ref)

    def test_basis_state_indices(self):
        indices = [("dn", 0), ("dn", 1), ("up", 0), ("up", 1)]
        hs = HilbertSpace([make_space_fermion(*ind) for ind in indices])

        # Basis of the N=2 sector
        basis_state_indices = [3, 5, 6, 9, 10, 12]

        H1 = 1.0 * (c_dag("up", 0) * c("up", 1) + c_dag("up", 1) * c("up", 0))
        H1 += 2.0 * (c_dag("dn", 0) * c("dn", 1) + c_dag("dn", 1) * c("dn", 0))
        H1op = LOperatorR(H1, hs)
        ref1 = 1.0 * (self.c_dag_mat(hs, "up", 0) @ self.c_mat(hs, "up", 1)
                      + self.c_dag_mat(hs, "up", 1) @ self.c_mat(hs, "up", 0))
        ref1 += 2.0 * (self.c_dag_mat(hs, "dn", 0) @ self.c_mat(hs, "dn", 1)
                       + self.c_dag_mat(hs, "dn", 1) @ self.c_mat(hs, "dn", 0))
        ref1 = ref1[basis_state_indices, :][:, basis_state_indices]
        assert_equal(make_matrix(H1op, basis_state_indices), ref1)

        H2 = 1.0j * (c_dag("up", 0) * c("up", 1) + c_dag("up", 1) * c("up", 0))
        H2 += 2.0 * (c_dag("dn", 0) * c("dn", 1) + c_dag("dn", 1) * c("dn", 0))
        H2op = LOperatorC(H2, hs)
        ref2 = 1.0j * (self.c_dag_mat(hs, "up", 0) @ self.c_mat(hs, "up", 1)
                       + self.c_dag_mat(hs, "up", 1) @ self.c_mat(hs, "up", 0))
        ref2 += 2.0 * (self.c_dag_mat(hs, "dn", 0) @ self.c_mat(hs, "dn", 1)
                       + self.c_dag_mat(hs, "dn", 1) @ self.c_mat(hs, "dn", 0))
        ref2 = ref2[basis_state_indices, :][:, basis_state_indices]
        assert_equal(make_matrix(H2op, basis_state_indices), ref2)

    def test_left_right_basis_state_indices(self):
        indices = [("dn", 0), ("dn", 1), ("up", 0), ("up", 1)]
        hs = HilbertSpace([make_space_fermion(*ind) for ind in indices])

        # Basis of the N=1 sector
        N1_basis_state_indices = [1, 2, 4, 8]
        # Basis of the N=2 sector
        N2_basis_state_indices = [3, 5, 6, 9, 10, 12]
        # Basis of the N=3 sector
        N3_basis_state_indices = [7, 11, 13, 14]

        for ind1, ind2 in product(indices, indices):
            op1 = LOperatorR(c(*ind1) * c(*ind2), hs)
            ref1 = self.c_mat(hs, *ind1) @ self.c_mat(hs, *ind2)
            ref1 = ref1[N1_basis_state_indices, :][:, N3_basis_state_indices]
            assert_equal(make_matrix(op1,
                                     N1_basis_state_indices,
                                     N3_basis_state_indices),
                         ref1)

            op2 = LOperatorC(1j * c(*ind1) * c(*ind2), hs)
            ref2 = 1j * self.c_mat(hs, *ind1) @ self.c_mat(hs, *ind2)
            ref2 = ref2[N1_basis_state_indices, :][:, N3_basis_state_indices]
            assert_equal(make_matrix(op2,
                                     N1_basis_state_indices,
                                     N3_basis_state_indices),
                         ref2)

        for ind in indices:
            op1 = LOperatorR(c(*ind), hs)
            ref1 = self.c_mat(hs, *ind)
            ref1 = ref1[N2_basis_state_indices, :][:, N3_basis_state_indices]
            assert_equal(make_matrix(op1,
                                     N2_basis_state_indices,
                                     N3_basis_state_indices),
                         ref1)

            op2 = LOperatorC(1j * c(*ind), hs)
            ref2 = 1j * self.c_mat(hs, *ind)
            ref2 = ref2[N2_basis_state_indices, :][:, N3_basis_state_indices]
            assert_equal(make_matrix(op2,
                                     N2_basis_state_indices,
                                     N3_basis_state_indices),
                         ref2)
