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
from pycommute.expression import Indices


class TestIndices(TestCase):

    # Check that elements of `v` are pairwise distinct
    def check_equality(self, v):
        for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
            self.assertEqual((x == y), (i == j))
            self.assertEqual((x != y), (i != j))

    # Check that elements of `v` are ordered
    def check_less_greater(self, v):
        for (i, x), (j, y) in product(enumerate(v), enumerate(v)):
            self.assertEqual((x < y), (i < j))
            self.assertEqual((x > y), (i > j))

    def test_Indices(self):
        all_ind = [Indices()]
        self.assertEqual(len(all_ind[-1]), 0)
        self.assertEqual(str(all_ind[-1]), "")
        self.assertEqual([i for i in all_ind[-1]], [])
        self.assertEqual(all_ind[-1].indices, [])

        for i in (0, 1, "xxx", "yyy"):
            all_ind.append(Indices(i))
            self.assertEqual(len(all_ind[-1]), 1)
            self.assertEqual(str(all_ind[-1]), str(i))
            self.assertEqual([n for n in all_ind[-1]], [i])
            self.assertEqual(all_ind[-1].indices, [i])

        for i, j in product((0, 1, "xxx", "yyy"), (0, 1, "xxx", "yyy")):
            all_ind.append(Indices(i, j))
            self.assertEqual(len(all_ind[-1]), 2)
            self.assertEqual(str(all_ind[-1]), "%s,%s" % (i, j))
            self.assertEqual([n for n in all_ind[-1]], [i, j])
            self.assertEqual(all_ind[-1].indices, [i, j])

        self.check_equality(all_ind)
        self.check_less_greater(all_ind)
