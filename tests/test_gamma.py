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

from itertools import product
from pycommute.expression import (Indices,
                                  Generator,
                                  MIN_USER_DEFINED_ALGEBRA_ID,
                                  Monomial,
                                  ExpressionC,
                                  conj)


# Metric
def eta(mu, nu):
    return (mu == nu) * (1.0 if mu == 0 else -1.0)


# 4D Levi-Civita symbol
def eps(i1, i2, i3, i4):
    return ((i4 - i3) * (i4 - i2) * (i4 - i1) * (i3 - i2) * (i3 - i1)
            * (i2 - i1) / 12)


class GeneratorGamma(Generator):

    def __init__(self, index):
        Generator.__init__(self, Indices(index))

    def algebra_id(self):
        return MIN_USER_DEFINED_ALGEBRA_ID

    def swap_with(self, g2, f):
        # c = -1, f(g) = 2\eta(g1, g2)
        assert self > g2
        diag = self.indices == g2.indices
        f.const_term = diag * (2 if (self.indices[0] == 0) else -2)
        f.terms = []
        return -1

    def simplify_prod(self, g2, f):
        # (\Gamma^0)^2 = I_4
        # (\Gamma^k)^2 = -I_4 for k=1,2,3
        if self == g2:
            f.const_term = 1 if (self.indices[0] == 0) else -1
            f.terms = []
            return True
        else:
            return False

    def conj(self, f):
        # Gamma^0 is Hermitian and Gamma^k are anti-Hermitian
        f.const_term = 0
        f.terms = [(self, 1 if (self.indices[0] == 0) else -1)]


class TestIdentities(TestCase):

    @classmethod
    def setUpClass(cls):
        # Gamma matrices
        cls.Gamma = [ExpressionC(1.0, Monomial([GeneratorGamma(mu)]))
                     for mu in range(4)]
        cls.Gammac = [
            ExpressionC(eta(mu, mu), Monomial([GeneratorGamma(mu)]))
            for mu in range(4)]
        cls.Gamma5 = 1j * cls.Gamma[0] * cls.Gamma[1] \
                        * cls.Gamma[2] * cls.Gamma[3]

    def test_anticommutators(self):
        for mu, nu in product(range(4), range(4)):
            self.assertEqual(self.Gamma[mu] * self.Gamma[nu]
                             + self.Gamma[nu] * self.Gamma[mu],
                             ExpressionC(2 * eta(mu, nu)))

    def test_herm_conj(self):
        self.assertEqual(conj(self.Gamma[0]), self.Gamma[0])
        for k in range(1, 4):
            self.assertEqual(conj(self.Gamma[k]), -self.Gamma[k])
        for mu in range(4):
            self.assertEqual(conj(self.Gamma[mu]),
                             self.Gamma[0] * self.Gamma[mu] * self.Gamma[0])

    def test_Gamma5(self):
        for mu in range(4):
            self.assertEqual(self.Gamma5 * self.Gamma[mu]
                             + self.Gamma[mu] * self.Gamma5,
                             ExpressionC())
        self.assertEqual(self.Gamma5 * self.Gamma5, ExpressionC(1))

    #
    # https://en.wikipedia.org/wiki/Gamma_matrices#Miscellaneous_identities
    #

    def test_identity1(self):
        s = sum(self.Gamma[mu] * self.Gammac[mu] for mu in range(4))
        self.assertEqual(s, ExpressionC(4))

    def test_identity2(self):
        for nu in range(4):
            s = sum(self.Gamma[mu] * self.Gamma[nu] * self.Gammac[mu]
                    for mu in range(4))
            self.assertEqual(s, -2 * self.Gamma[nu])

    def test_identity3(self):
        for nu, rho in product(range(4), range(4)):
            s = sum(self.Gamma[mu] * self.Gamma[nu]
                    * self.Gamma[rho] * self.Gammac[mu]
                    for mu in range(4))
            self.assertEqual(s, ExpressionC(4 * eta(nu, rho)))

    def test_identity4(self):
        for nu, rho, sigma in product(range(4), repeat=3):
            s = sum(self.Gamma[mu] * self.Gamma[nu] * self.Gamma[rho]
                    * self.Gamma[sigma] * self.Gammac[mu]
                    for mu in range(4))
            self.assertEqual(
                s,
                -2 * self.Gamma[sigma] * self.Gamma[rho] * self.Gamma[nu]
            )

    def test_identity5(self):
        for mu, nu, rho in product(range(4), repeat=3):
            lhs = self.Gamma[mu] * self.Gamma[nu] * self.Gamma[rho]
            rhs = eta(mu, nu) * self.Gamma[rho] + eta(nu, rho) * self.Gamma[mu]\
                - eta(mu, rho) * self.Gamma[nu]
            for sigma in range(4):
                rhs += -1j * eps(sigma, mu, nu, rho) \
                    * self.Gammac[sigma] * self.Gamma5
            self.assertEqual(lhs, rhs)
