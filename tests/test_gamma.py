#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from unittest import TestCase

from itertools import product
from pycommute.expression import *

class GeneratorGamma(Generator):

    def __init__(self, index):
        Generator.__init__(self, index)

    def algebra_id(self):
        return MIN_USER_DEFINED_ALGEBRA_ID

    def swap_with(self, g2, f):
        # c = -1, f(g) = 2\eta(g1, g2)
        assert self > g2
        diag = self.indices == g2.indices
        f.set(diag * (2 if (self.indices[0] == 0) else -2))
        return -1

    def simplify_prod(self, g2, f):
        # (\Gamma^0)^2 = I_4
        # (\Gamma^k)^2 = -I_4 for k=1,2,3
        if self == g2:
            f.set(1 if (self.indices[0] == 0) else -1)
            return True
        else:
            return False

    def conj(self, f):
        # Gamma^0 is Hermitian and Gamma^k are anti-Hermitian
        f.set(0, self, 1 if (self.indices[0] == 0) else -1)

class TestIdentities(TestCase):

    @classmethod
    def setUpClass(cls):
        # Metric
        cls.eta = lambda mu, nu: (mu == nu) * (1 if mu == 0 else -1)
        # Gamma matrices
        cls.Gamma = [ExpressionC(1.0, Monomial([GeneratorGamma(mu)]))
                     for mu in range(4)]
        cls.Gammac = [ExpressionC(cls.eta(mu, mu), Monomial([GeneratorGamma(mu)]))
                      for mu in range(4)]

    def test_anticommutators(self):
        for mu, nu in product(range(4), range(4)):
            self.assertEqual(self.Gamma[mu] * self.Gamma[nu] + \
              self.Gamma[nu] * self.Gamma[mu], 2 * self.eta(mu, nu))
