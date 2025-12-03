#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2020-2025 Igor Krivenko
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

#
# ## User-defined algebra of Dirac gamma matrices
#

# Algebra generators and expressions
from pycommute.expression import (Indices,
                                  Generator,
                                  MIN_USER_DEFINED_ALGEBRA_ID,
                                  Monomial,
                                  ExpressionC,
                                  conj)


#
# Our generator type: a gamma matrix with one integer index \mu = 0, ..., 3
#
class GeneratorGamma(Generator):

    # Initialize the base class by passing the index \mu to it.
    def __init__(self, index):
        Generator.__init__(self, Indices(index))

    # Algebra ID of this generator. In this case, the lowest algebra ID
    # available to user-defined algebras. Further algebras can use IDs
    # MIN_USER_DEFINED_ALGEBRA_ID + 1, ...
    def algebra_id(self):
        return MIN_USER_DEFINED_ALGEBRA_ID

    # This method will be called for g1 = self and g2 such that g1 > g2.
    # We must transform the product g1 * g2 and put it into the
    # canonical order,
    #
    # g1 * g2 -> -g2 * g1 + 2\eta(g1, g2) = -g2 * g1 + f.
    def swap_with(self, g2, f):
        # Update the LinearFunctionGen object 'f'
        #
        # Set the constant term 2\eta(g1, g2), where \eta is the diagonal
        # Minkowski metric tensor.
        mu = self.indices[0]
        nu = g2.indices[0]
        f.const_term = (mu == nu) * (2 if mu == 0 else -2)
        # 'f' gets no contributions linear in the generators.
        f.terms = []
        return -1     # Return the coefficient in front of g2 * g1.

    # This method will be called for g1 = self and g2 that are already
    # canonically ordered, g1 <= g2.
    #
    # It tries to simplify squares of gamma matrices,
    # f = (\gamma^0)^2 = I_4
    # f = (\gamma^k)^2 = -I_4 for k = 1,2,3.
    def simplify_prod(self, g2, f):
        if self == g2:
            # Update the LinearFunctionGen object 'f'.
            #
            # The constant term of 'f' is equal to either 1 or -1, depending
            # on which gamma matrix is being squared.
            mu = self.indices[0]
            f.const_term = 1 if (mu == 0) else -1
            # 'f' gets no contributions linear in the generators.
            f.terms = []
            return True   # A square of a gamma matrix has been simplified.
        else:
            return False  # No simplification has been applied.

    # Compute Hermitian conjugate of a gamma matrix g = self,
    # f = (\gamma_0)^+ = \gamma_0
    # f = (\gamma_k)^+ = -\gamma_k, k=1,2,3.
    #
    # N.B.: If this method is not overridden in the subclass, all generators
    # of the new algebra are assumed to be Hermitian.
    def conj(self, f):
        # Update the LinearFunctionGen object 'f'
        #
        # Hermitian conjugate of 'g' contains no constant contribution.
        f.const_term = 0
        # The only linear in generators term is proportional to the argument
        # g = self, and the coefficient in front of it depends on the index.
        mu = self.indices[0]
        f.terms = [(self, 1 if (mu == 0) else -1)]

    # String representation of this generator
    def __repr__(self):
        mu = self.indices[0]
        return f"γ^{mu}"


# Convenience function: Make an expression out of one gamma matrix.
def gamma(mu):
    return ExpressionC(1.0, Monomial([GeneratorGamma(mu)]))


#
# ### Check that expressions with gamma matrices behave as expected
#

# Minkowski metric tensor
def eta(mu, nu):
    return (mu == nu) * (1.0 if mu == 0 else -1.0)


# Check the commutation relations
for mu in range(4):
    for nu in range(4):
        gamma_mu = gamma(mu)
        gamma_nu = gamma(nu)

        print(f"{{{gamma_mu}, {gamma_nu}}} - 2η({mu}, {nu}) =",
              (gamma_mu * gamma_nu + gamma_nu * gamma_mu - 2 * eta(mu, nu)))

# \gamma^5
gamma5 = 1j * gamma(0) * gamma(1) * gamma(2) * gamma(3)

# \gamma^5 is Hermitian ...
print("γ^5 - conj(γ^5) =", gamma5 - conj(gamma5))
# ... and anti-commutes with \gamma_\mu.
for mu in range(4):
    gamma_mu = gamma(mu)
    print(f"{{γ^5, {gamma_mu}}} =", gamma5 * gamma_mu + gamma_mu * gamma5)
