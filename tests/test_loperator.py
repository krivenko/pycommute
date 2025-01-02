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

from pycommute.expression import (ExpressionR, c, c_dag, make_complex)
from pycommute.loperator import (HilbertSpace, LOperatorR, LOperatorC)

import numpy as np
from numpy.testing import assert_equal


class TestLOperator(TestCase):

    def test_empty(self):
        expr0 = ExpressionR()
        hs = HilbertSpace(expr0)
        lop = LOperatorR(expr0, hs)

        sv = np.array([], dtype=float)
        assert_equal(lop * sv, sv)
        dst = np.array([], dtype=float)
        lop(sv, dst)
        assert_equal(sv, dst)

    def test_LOperatorR(self):
        expr1 = 3 * c_dag("dn")
        expr2 = 3 * c("up")
        expr = 2 * expr1 - 2 * expr2

        hs = HilbertSpace(expr)
        lop1 = LOperatorR(expr1, hs)
        lop2 = LOperatorR(expr2, hs)
        lop = LOperatorR(expr, hs)

        src = np.array([1, 1, 1, 1])
        dst_real = np.zeros((4,), dtype=float)
        dst_complex = np.zeros((4,), dtype=complex)

        assert_equal(lop1 * src, np.array([0, 3, 0, 3]))
        lop1(src, dst_real)
        assert_equal(dst_real, np.array([0, 3, 0, 3]))
        lop1(src, dst_complex)
        assert_equal(dst_complex, np.array([0, 3, 0, 3], dtype=complex))

        assert_equal(lop2 * src, np.array([3, -3, 0, 0]))
        lop2(src, dst_real)
        assert_equal(dst_real, np.array([3, -3, 0, 0]))
        lop2(src, dst_complex)
        assert_equal(dst_complex, np.array([3, -3, 0, 0], dtype=complex))

        assert_equal(lop * src, np.array([-6, 12, 0, 6]))
        lop(src, dst_real)
        assert_equal(dst_real, np.array([-6, 12, 0, 6]))
        lop(src, dst_complex)
        assert_equal(dst_complex, np.array([-6, 12, 0, 6], dtype=complex))

        src_complex = 1j * np.array([1, 1, 1, 1])
        assert_equal(lop * src_complex, np.array([-6j, 12j, 0, 6j]))
        lop(src_complex, dst_complex)
        assert_equal(dst_complex, np.array([-6j, 12j, 0, 6j]))

        with self.assertRaisesRegex(
            RuntimeError,
            "^State vector must be a 1-dimensional array$"
        ):
            lop * np.zeros((3, 3, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            "^Source state vector must be a 1-dimensional array$"
        ):
            lop(np.zeros((3, 3, 3)), np.zeros((5,)))
        with self.assertRaisesRegex(
            RuntimeError,
            "^Destination state vector must be a 1-dimensional array$"
        ):
            lop(np.zeros((5,)), np.zeros((3, 3, 3)))

    def test_LOperatorC(self):
        expr1 = make_complex(3 * c_dag("dn"))
        expr2 = make_complex(3 * c("up"))
        expr = 2j * expr1 - 2j * expr2

        hs = HilbertSpace(expr)
        lop1 = LOperatorC(expr1, hs)
        lop2 = LOperatorC(expr2, hs)
        lop = LOperatorC(expr, hs)

        src = np.array([1, 1, 1, 1])
        dst = np.zeros((4,), dtype=complex)

        assert_equal(lop1 * src, np.array([0, 3, 0, 3], dtype=complex))
        lop1(src, dst)
        assert_equal(dst, np.array([0, 3, 0, 3], dtype=complex))

        assert_equal(lop2 * src, np.array([3, -3, 0, 0], dtype=complex))
        lop2(src, dst)
        assert_equal(dst, np.array([3, -3, 0, 0], dtype=complex))

        assert_equal(lop * src, np.array([-6j, 12j, 0, 6j]))
        lop(src, dst)
        assert_equal(dst, np.array([-6j, 12j, 0, 6j]))

        src_complex = 1j * np.array([1, 1, 1, 1])
        assert_equal(lop * src_complex, np.array([6, -12, 0, -6]))
        lop(src_complex, dst)
        assert_equal(dst, np.array([6, -12, 0, -6]))

        with self.assertRaisesRegex(
            RuntimeError,
            "^State vector must be a 1-dimensional array$"
        ):
            lop * np.zeros((3, 3, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            "^Source state vector must be a 1-dimensional array$"
        ):
            lop(np.zeros((3, 3, 3)), np.zeros((5,), dtype=complex))
        with self.assertRaisesRegex(
            RuntimeError,
            "^Destination state vector must be a 1-dimensional array$"
        ):
            lop(np.zeros((5,)), np.zeros((3, 3, 3), dtype=complex))

    def test_overload_selection(self):
        expr = 6 * c_dag("dn") - 6 * c("up")
        hs = HilbertSpace(expr)
        lop = LOperatorR(expr, hs)

        src_int = np.array([1, 1, 1, 1], dtype=int)
        src_real = np.array([1, 1, 1, 1], dtype=float)
        src_complex = np.array([1, 1, 1, 1], dtype=complex)

        ref_real = np.array([-6, 12, 0, 6], dtype=float)
        ref_complex = np.array([-6, 12, 0, 6], dtype=complex)

        self.assertEqual((lop * src_int).dtype, np.float64)
        self.assertEqual((lop * src_real).dtype, np.float64)
        self.assertEqual((lop * src_complex).dtype, np.complex128)

        dst_int = np.zeros(4, dtype=int)
        dst_real = np.zeros(4, dtype=float)
        dst_complex = np.zeros(4, dtype=complex)

        self.assertRaises(TypeError, lop, src_int, dst_int)
        self.assertRaises(TypeError, lop, src_real, dst_int)
        self.assertRaises(TypeError, lop, src_complex, dst_int)
        self.assertRaises(TypeError, lop, src_complex, dst_real)

        lop(src_int, dst_real)
        assert_equal(dst_real, ref_real)
        lop(src_int, dst_complex)
        assert_equal(dst_complex, ref_complex)
        lop(src_real, dst_real)
        assert_equal(dst_real, ref_real)
        lop(src_real, dst_complex)
        assert_equal(dst_complex, ref_complex)
        lop(src_complex, dst_complex)
        assert_equal(dst_complex, ref_complex)

        expr = 6j * c_dag("dn") - 6j * c("up")
        hs = HilbertSpace(expr)
        lop = LOperatorC(expr, hs)

        self.assertEqual((lop * src_int).dtype, np.complex128)
        self.assertEqual((lop * src_real).dtype, np.complex128)
        self.assertEqual((lop * src_complex).dtype, np.complex128)

        ref_complex = np.array([-6j, 12j, 0, 6j], dtype=complex)

        self.assertRaises(TypeError, lop, src_int, dst_int)
        self.assertRaises(TypeError, lop, src_real, dst_int)
        self.assertRaises(TypeError, lop, src_complex, dst_int)
        self.assertRaises(TypeError, lop, src_int, dst_real)
        self.assertRaises(TypeError, lop, src_real, dst_real)
        self.assertRaises(TypeError, lop, src_complex, dst_real)

        lop(src_int, dst_complex)
        assert_equal(dst_complex, ref_complex)
        lop(src_real, dst_complex)
        assert_equal(dst_complex, ref_complex)
        lop(src_complex, dst_complex)
        assert_equal(dst_complex, ref_complex)

    def test_strided_arrays(self):
        expr = 6 * c_dag("dn") - 6 * c("up")
        hs = HilbertSpace(expr)
        lop = LOperatorR(expr, hs)

        src_real = 999 * np.ones((10,), dtype=float)
        src_real[3:10:2] = 1
        src_complex = np.array(src_real, dtype=complex)

        assert_equal(lop * src_real[3:10:2],
                     np.array([-6, 12, 0, 6], dtype=float))
        assert_equal(lop * src_complex[3:10:2],
                     np.array([-6, 12, 0, 6], dtype=complex))

        dst_real = 777 * np.ones((10,), dtype=float)
        dst_complex = np.array(dst_real, dtype=complex)

        ref_real = np.array([777, 777, -6, 777, 12, 777, 0, 777, 6, 777],
                            dtype=float)
        ref_complex = np.array(ref_real, dtype=complex)

        lop(src_real[3:10:2], dst_real[2:9:2])
        assert_equal(dst_real, ref_real)
        lop(src_real[3:10:2], dst_complex[2:9:2])
        assert_equal(dst_complex, ref_complex)
        lop(src_complex[3:10:2], dst_complex[2:9:2])
        assert_equal(dst_complex, ref_complex)

        expr = 6j * c_dag("dn") - 6j * c("up")
        hs = HilbertSpace(expr)
        lop = LOperatorC(expr, hs)

        ref_complex = np.array([777, 777, -6j, 777, 12j, 777, 0, 777, 6j, 777],
                               dtype=complex)

        lop(src_real[3:10:2], dst_complex[2:9:2])
        assert_equal(dst_complex, ref_complex)
        lop(src_complex[3:10:2], dst_complex[2:9:2])
        assert_equal(dst_complex, ref_complex)
