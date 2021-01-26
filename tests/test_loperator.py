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

from pycommute.expression import *
from pycommute.loperator import *

import numpy as np
from numpy.testing import assert_equal

class TestLOperator(TestCase):

    def test_empty(self):
        expr0 = ExpressionR()
        hs = HilbertSpace(expr0)
        lop = LOperatorR(expr0, hs)

        sv = np.array([], dtype = float)
        assert_equal(lop * sv, sv)
        dst = np.array([], dtype = float)
        lop(sv, dst)
        assert_equal(sv, dst)

    def test_LOperatorR(self):
        expr1 = 3*c_dag("dn")
        expr2 = 3*c("up")
        expr = 2*expr1 - 2*expr2

        hs = HilbertSpace(expr)
        lop1 = LOperatorR(expr1, hs)
        lop2 = LOperatorR(expr2, hs)
        lop = LOperatorR(expr, hs)

        src = np.array([1, 1, 1, 1])
        dst_real = np.zeros((4,), dtype = float)
        dst_complex = np.zeros((4,), dtype = complex)

        assert_equal(lop1 * src, np.array([0, 3, 0, 3]))
        lop1(src, dst_real)
        assert_equal(dst_real, np.array([0, 3, 0, 3]))
        lop1(src, dst_complex)
        assert_equal(dst_complex, np.array([0, 3, 0, 3], dtype = complex))

        assert_equal(lop2 * src, np.array([3, -3, 0, 0]))
        lop2(src, dst_real)
        assert_equal(dst_real, np.array([3, -3, 0, 0]))
        lop2(src, dst_complex)
        assert_equal(dst_complex, np.array([3, -3, 0, 0], dtype = complex))

        assert_equal(lop * src, np.array([-6, 12, 0, 6]))
        lop(src, dst_real)
        assert_equal(dst_real, np.array([-6, 12, 0, 6]))
        lop(src, dst_complex)
        assert_equal(dst_complex, np.array([-6, 12, 0, 6], dtype = complex))

        src_complex = 1j * np.array([1, 1, 1, 1])
        assert_equal(lop * src_complex, np.array([-6j, 12j, 0, 6j]))
        lop(src_complex, dst_complex)
        assert_equal(dst_complex, np.array([-6j, 12j, 0, 6j]))

        with self.assertRaisesRegex(
          RuntimeError,
          "^State vector must be a 1-dimensional array$"):
          lop * np.zeros((3, 3, 3))
        with self.assertRaisesRegex(
          RuntimeError,
          "^Source state vector must be a 1-dimensional array$"):
          lop(np.zeros((3, 3, 3)), np.zeros((5,)))
        with self.assertRaisesRegex(
          RuntimeError,
          "^Destination state vector must be a 1-dimensional array$"):
          lop(np.zeros((5,)), np.zeros((3, 3, 3)))

    def test_LOperatorC(self):
        expr1 = make_complex(3*c_dag("dn"))
        expr2 = make_complex(3*c("up"))
        expr = 2j*expr1 - 2j*expr2

        hs = HilbertSpace(expr)
        lop1 = LOperatorC(expr1, hs)
        lop2 = LOperatorC(expr2, hs)
        lop = LOperatorC(expr, hs)

        src = np.array([1, 1, 1, 1])
        dst = np.zeros((4,), dtype = complex)

        assert_equal(lop1 * src, np.array([0, 3, 0, 3], dtype = complex))
        lop1(src, dst)
        assert_equal(dst, np.array([0, 3, 0, 3], dtype = complex))

        assert_equal(lop2 * src, np.array([3, -3, 0, 0], dtype = complex))
        lop2(src, dst)
        assert_equal(dst, np.array([3, -3, 0, 0], dtype = complex))

        assert_equal(lop * src, np.array([-6j, 12j, 0, 6j]))
        lop(src, dst)
        assert_equal(dst, np.array([-6j, 12j, 0, 6j]))

        src_complex = 1j * np.array([1, 1, 1, 1])
        assert_equal(lop * src_complex, np.array([6, -12, 0, -6]))
        lop(src_complex, dst)
        assert_equal(dst, np.array([6, -12, 0, -6]))

        with self.assertRaisesRegex(
          RuntimeError,
          "^State vector must be a 1-dimensional array$"):
          lop * np.zeros((3, 3, 3))
        with self.assertRaisesRegex(
          RuntimeError,
          "^Source state vector must be a 1-dimensional array$"):
          lop(np.zeros((3, 3, 3)), np.zeros((5,), dtype = complex))
        with self.assertRaisesRegex(
          RuntimeError,
          "^Destination state vector must be a 1-dimensional array$"):
          lop(np.zeros((5,)), np.zeros((3, 3, 3), dtype = complex))
