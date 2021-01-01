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

#ifndef PYCOMMUTE_PYBIND11_WORKAROUNDS_HPP_
#define PYCOMMUTE_PYBIND11_WORKAROUNDS_HPP_

#include <pybind11/pybind11.h>

//
// Workarounds for some pybind11 issues
//

// pybind11 issue #2516
//
// Workaround by Yannick Jadoul:
// https://github.com/pybind/pybind11/issues/2033#issuecomment-703170432
#undef PYBIND11_OVERRIDE_IMPL
#define PYBIND11_OVERRIDE_IMPL(ret_type, cname, name, ...)                     \
do {                                                                           \
  pybind11::gil_scoped_acquire gil;                                            \
  pybind11::function override =                                                \
        pybind11::get_override(static_cast<const cname *>(this), name);        \
  if (override) {                                                              \
    auto o = override.operator()<py::return_value_policy::reference>           \
      (__VA_ARGS__);                                                           \
    if (pybind11::detail::cast_is_temporary_value_reference<ret_type>::value) {\
            static pybind11::detail::override_caster_t<ret_type> caster;       \
            return pybind11::detail::cast_ref<ret_type>(std::move(o), caster); \
    }                                                                          \
    else return pybind11::detail::cast_safe<ret_type>(std::move(o));           \
  }                                                                            \
} while (false)

#endif
