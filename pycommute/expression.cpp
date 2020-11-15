/*******************************************************************************
 *
 * This file is part of libcommute, a C++11/14/17 header-only library allowing
 * to manipulate polynomial expressions with quantum-mechanical operators.
 *
 * Copyright (C) 2016-2020 Igor Krivenko <igor.s.krivenko@gmail.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 ******************************************************************************/

#include "pybind11_workarounds.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <libcommute/expression/expression.hpp>
#include <libcommute/expression/dyn_indices.hpp>

#include <cassert>
#include <string>
#include <sstream>
#include <tuple>

using namespace libcommute;
namespace py = pybind11;

//
// Just because std::to_string(std::string) is not part of STL ...
//

template<typename T>
std::string to_string(T&& x) { using std::to_string; return to_string(x); }
std::string to_string(std::string const& x) { return x; }

//
// Some commonly used type abbreviations
//

using dynamic_indices::dyn_indices;
using gen_type = generator<dyn_indices>;

//
// Helper classes for abstract base generator<dyn_indices>
//

class py_generator : public gen_type {

  dyn_indices init(py::args args) {
    dyn_indices::indices_t v;
    v.reserve(args.size());
    for(auto const& a : args)
      v.emplace_back(a.cast<std::variant<int, std::string>>());
    return dyn_indices(std::move(v));
  }

  public:

  using gen_type::gen_type;

  py_generator(py::args args) : gen_type(std::move(init(args))) {}

  int algebra_id() const override {
    PYBIND11_OVERRIDE_PURE(int, gen_type, algebra_id, );
  }

  std::unique_ptr<gen_type> clone() const override {
    // Generators are immutable, so one should use multiple references instead
    // of creating deep copies in Python.
    assert(false);
    return nullptr;
  }

  double swap_with(gen_type const& g2, linear_function_t & f) const override {
    PYBIND11_OVERRIDE_PURE(double, gen_type, swap_with, g2, f);
  }

  bool simplify_prod(gen_type const& g2, linear_function_t & f) const override {
    PYBIND11_OVERRIDE(bool, gen_type, simplify_prod, g2, f);
  }

  bool reduce_power(int power, linear_function_t & f) const override {
    PYBIND11_OVERRIDE(bool, gen_type, reduce_power, power, f);
  }

  bool equal(gen_type const& g) const override {
    PYBIND11_OVERRIDE(bool, gen_type, equal, g);
  }

  bool less(gen_type const& g) const override {
    PYBIND11_OVERRIDE(bool, gen_type, less, g);
  }

  bool greater(gen_type const& g) const override {
    PYBIND11_OVERRIDE(bool, gen_type, greater, g);
  }
};

class py_generator_publicist : public gen_type {
public:
  using gen_type::equal;
  using gen_type::less;
  using gen_type::greater;
};

//
// 'expression' Python module
//

PYBIND11_MODULE(expression, m) {

  m.doc() = "Polynomial expressions involving quantum-mechanical operators "
            "and manipulations with them";

  //
  // dynamic_indices::dyn_indices
  //

  py::class_<dyn_indices>(m, "Indices",
    "Mixed sequence of integer/string indices"
  )
  .def(py::init([](py::args args) {
      dyn_indices::indices_t v;
      v.reserve(args.size());
      for(auto const& a : args)
        v.emplace_back(a.cast<std::variant<int, std::string>>());
      return std::make_unique<dyn_indices>(std::move(v));
    })
   )
  .def("__len__", &dyn_indices::size, "Index sequence length")
  .def(py::self == py::self)
  .def(py::self != py::self)
  .def(py::self < py::self)
  .def(py::self > py::self)
  .def_property_readonly("indices",
                         &dyn_indices::operator dyn_indices::indices_t const&)
  .def("__repr__", [](dyn_indices const& indices) {
    auto const& ind = static_cast<dyn_indices::indices_t const&>(indices);
    const size_t N = ind.size();
    std::string s;
    for(size_t i = 0; i < N; ++i) {
      std::visit([&s](auto const& x) { s += to_string(x); }, ind[i]);
      if(i + 1 < N) s += ",";
    }
    return s;
  })
  .def("__iter__", [](const dyn_indices &indices) {
      auto const& ind = static_cast<dyn_indices::indices_t const&>(indices);
      return py::make_iterator(ind.begin(), ind.end());
    },
    py::keep_alive<0, 1>()
  );

  //
  // generator::linear_function_t
  //

  py::class_<gen_type::linear_function_t>(m, "LinearFunctionGen",
    "Linear combination of algebra generators plus a constant term"
  )
  .def(py::init<double>(), "Construct a constant")
  .def(py::init([](
    double const_term,
    std::vector<std::pair<const gen_type*, double>> const& terms) {
      std::vector<std::pair<std::unique_ptr<gen_type>, double>> terms_;
      terms_.reserve(terms.size());
      for(auto const& t : terms)
        terms_.emplace_back(t.first->clone(), t.second);
      return std::make_unique<gen_type::linear_function_t>(
        const_term,
        std::move(terms_)
      );
    }),
    "Construct from a constant term and a list of coefficient/generator pairs"
  );

  //
  // generator<dyn_indices>
  //

  py::class_<gen_type, py_generator>(m, "Generator",
    "Abstract algebra generator"
  )
  // Algebra ID
  .def("algebra_id", &gen_type::algebra_id)
  // Product transformation methods
  .def("swap_with", &gen_type::swap_with)
  .def("simplify_prod", &gen_type::simplify_prod)
  .def("reduce_power", &gen_type::reduce_power)
  // Comparison methods
  .def("equal", &py_generator_publicist::equal)
  .def("less", &py_generator_publicist::less)
  .def("greater", &py_generator_publicist::greater)
  // Comparison operators
  .def("__eq__",
       [](gen_type const& g1, gen_type const& g2){ return g1 == g2; },
       py::is_operator()
  )
  .def("__ne__",
       [](gen_type const& g1, gen_type const& g2){ return g1 != g2; },
       py::is_operator()
  )
  .def("__lt__",
       [](gen_type const& g1, gen_type const& g2){ return g1 < g2; },
       py::is_operator()
  )
  .def("__gt__",
       [](gen_type const& g1, gen_type const& g2){ return g1 > g2; },
       py::is_operator()
  )
  // String representation
  .def("__repr__",
       [](gen_type const& g) {
         std::ostringstream ss; ss << g; return ss.str();
       }
  );
}
