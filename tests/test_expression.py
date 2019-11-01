#
# This file is part of libcommute, a C++11/14/17 header-only library allowing
# to manipulate polynomial expressions with quantum-mechanical operators.
#
# Copyright (C) 2016-2019 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from itertools import product
from pycommute.expression import *

# Check that elements of `v` are pairwise distinct
def check_equality(v):
    for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
        assert (x == y) == (i == j)
        assert (x != y) == (i != j)

# Check that elements of `v` are ordered
def check_less_greater(v):
    for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
        assert (x < y) == (i < j)
        assert (x > y) == (i > j)

def test_Indices():
    all_ind = [Indices()]
    assert len(all_ind[-1]) == 0
    assert str(all_ind[-1]) == ""
    assert [i for i in all_ind[-1]] == []
    assert all_ind[-1].indices == []

    for i in (0, 1, "xxx", "yyy"):
        all_ind.append(Indices([i]))
        assert len(all_ind[-1]) == 1
        assert str(all_ind[-1]) == str(i)
        assert [n for n in all_ind[-1]] == [i]
        assert all_ind[-1].indices == [i]

    for i, j in product((0, 1, "xxx", "yyy"), (0, 1, "xxx", "yyy")):
        all_ind.append(Indices([i, j]))
        assert len(all_ind[-1]) == 2
        assert str(all_ind[-1]) == "%s,%s" % (i,j)
        assert [n for n in all_ind[-1]] == [i, j]
        assert all_ind[-1].indices == [i, j]

    check_equality(all_ind)
    check_less_greater(all_ind)

# FIXME
#class Derived2(Base):
#    def __init__(self, a, c):
#      self.c = c

#def test_Base():
b = Base(4)
b2 = b.clone()

assert b.a == 4
assert b2.a == 4
assert id(b) != id(b2)
assert b.virtual_method(3) == 12
assert b2.virtual_method(3) == 12

b3 = b2.clone()
assert b3.a == 4
assert id(b) != id(b3)
assert b3.virtual_method(3) == 12

#assert b.a == []

#d2 = Derived2(3, 4);
#assert d2.a == []


