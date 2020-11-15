#
# This file is part of libcommute, a C++11/14/17 header-only library allowing
# to manipulate polynomial expressions with quantum-mechanical operators.
#
# Copyright (C) 2016-2020 Igor Krivenko <igor.s.krivenko@gmail.com>
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
        all_ind.append(Indices(i))
        assert len(all_ind[-1]) == 1
        assert str(all_ind[-1]) == str(i)
        assert [n for n in all_ind[-1]] == [i]
        assert all_ind[-1].indices == [i]

    for i, j in product((0, 1, "xxx", "yyy"), (0, 1, "xxx", "yyy")):
        all_ind.append(Indices(i, j))
        assert len(all_ind[-1]) == 2
        assert str(all_ind[-1]) == "%s,%s" % (i,j)
        assert [n for n in all_ind[-1]] == [i, j]
        assert all_ind[-1].indices == [i, j]

    check_equality(all_ind)
    check_less_greater(all_ind)

def test_LinearFunctionGen():
    lf0 = LinearFunctionGen(5.0)
    #TODO: test the other constructor

