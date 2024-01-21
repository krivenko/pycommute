/*******************************************************************************
 *
 * This file is part of pycommute, Python bindings for the libcommute C++
 * quantum operator algebra library.
 *
 * Copyright (C) 2020-2024 Igor Krivenko <igor.s.krivenko@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 ******************************************************************************/

#ifndef PYCOMMUTE_NUMPY_STATE_VECTORS_HPP_
#define PYCOMMUTE_NUMPY_STATE_VECTORS_HPP_

#include <libcommute/scalar_traits.hpp>
#include <libcommute/loperator/state_vector.hpp>

#include <pybind11/numpy.h>

#include <algorithm>

//
// Implementation of the StateVector interface for NumPy types
//

namespace libcommute {

//
// pybind11::array_t
//

template<typename ScalarType, int ExtraFlags>
struct element_type<pybind11::array_t<ScalarType, ExtraFlags>> {
  using type = ScalarType;
};

template<typename ScalarType, int ExtraFlags>
inline ScalarType const& get_element(
  pybind11::array_t<ScalarType, ExtraFlags> const& sv,
  sv_index_type n) {
  return *sv.data(n);
}

template<typename ScalarType, int ExtraFlags, typename T>
inline void update_add_element(pybind11::array_t<ScalarType, ExtraFlags> & sv,
                               sv_index_type n,
                               T value) {
  *sv.mutable_data(n) += value;
}

template<typename ScalarType, int ExtraFlags>
inline void set_zeros(pybind11::array_t<ScalarType, ExtraFlags> & sv) {
  auto z = scalar_traits<ScalarType>::make_const(0);
  auto d = sv.template mutable_unchecked<1>();
  sv_index_type size = d.shape(0);
  for(sv_index_type n = 0; n < size; ++n)
    d(n) = z;
}

template<typename ScalarType, int ExtraFlags, typename Functor>
inline void foreach(pybind11::array_t<ScalarType, ExtraFlags> const& sv,
                    Functor&& f) {
  auto d = sv.template unchecked<1>();
  sv_index_type size = d.shape(0);
  for(sv_index_type n = 0; n < size; ++n) {
    auto const& a = d(n);
    if(scalar_traits<ScalarType>::is_zero(a))
      continue;
    else
      f(n, a);
  }
}

//
// Single column view of a 2D NumPy array
//

template<typename ScalarType>
struct column_view {

  pybind11::array_t<ScalarType, pybind11::array::f_style> & array;
  pybind11::ssize_t n_rows;
  ScalarType* data_ptr;

  column_view(pybind11::array_t<ScalarType, pybind11::array::f_style> & array,
              pybind11::ssize_t n_rows) :
    array(array), n_rows(n_rows), data_ptr(array.mutable_data(0, 0)) {}

  void set_column(pybind11::ssize_t column) {
    data_ptr = array.mutable_data(0, column);
  }
};

//
// Implementation of the StateVector interface for column_view
//

template<typename ScalarType>
struct element_type<column_view<ScalarType>> { using type = ScalarType; };

template<typename ScalarType>
inline ScalarType const& get_element(
  column_view<ScalarType> const& sv,
  sv_index_type n) {
  return *(sv.data_ptr + n);
}

template<typename ScalarType, typename T>
inline void update_add_element(column_view<ScalarType> & sv,
                               sv_index_type n,
                               T value) {
  *(sv.data_ptr + n) += value;
}

template<typename ScalarType>
inline void set_zeros(column_view<ScalarType> & sv) {
  auto z = scalar_traits<ScalarType>::make_const(0);
  std::fill(sv.data_ptr, sv.data_ptr + sv.n_rows, z);
}

template<typename ScalarType, typename Functor>
inline void foreach(column_view<ScalarType> const& sv, Functor&& f) {
  for(sv_index_type n = 0; n < sv.n_rows; ++n) {
    auto const& a = *(sv.data_ptr + n);
    if(scalar_traits<ScalarType>::is_zero(a))
      continue;
    else
      f(n, a);
  }
}

}

#endif
