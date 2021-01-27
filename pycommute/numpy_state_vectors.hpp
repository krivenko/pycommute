/*******************************************************************************
 *
 * This file is part of pycommute, Python bindings for the libcommute C++
 * quantum operator algebra library.
 *
 * Copyright (C) 2020-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
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

}

#endif
