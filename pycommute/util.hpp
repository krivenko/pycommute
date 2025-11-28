/*******************************************************************************
 *
 * This file is part of pycommute, Python bindings for the libcommute C++
 * quantum operator algebra library.
 *
 * Copyright (C) 2020-2025 Igor Krivenko
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 ******************************************************************************/

#ifndef PYCOMMUTE_UTIL_HPP_
#define PYCOMMUTE_UTIL_HPP_

#include <cmath>
#include <string>
#include <type_traits>

using dcomplex = std::complex<double>;

//
// Names of the two scalar types used in pycommute
//

template<typename ScalarType> std::string scalar_type_name() {
  static_assert(std::is_same_v<ScalarType, double> ||
                std::is_same_v<ScalarType, std::complex<double>>
  );
  if constexpr(std::is_same_v<ScalarType, double>)
    return std::string("real");
  else
    return std::string("complex");
}

#endif
