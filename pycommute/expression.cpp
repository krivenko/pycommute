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

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <libcommute/expression/expression.hpp>
#include <libcommute/expression/dyn_indices.hpp>

#include <string>

using namespace libcommute;
namespace py = pybind11;

template<typename T>
std::string to_string(T&& x) { using std::to_string; return to_string(x); }
std::string to_string(std::string const& x) { return x; }

//// FIXME!
class Base {
protected:

  int a;

public:

  Base(int a) : a(a) {
    std::cout << "Base created at " << this << std::endl;
  }
  virtual ~Base() {
    std::cout << "Base destroyed at " << this << std::endl;
  }

  virtual std::unique_ptr<Base> clone() const = 0;
  virtual double virtual_method(double x) const { return a*x; }

  int get_a() const { return a; }
};

class Derived : public Base {
protected:
  int b;

public:

  Derived(int a, int b) : Base(a), b(b) {}
  virtual ~Derived() = default;

  virtual std::unique_ptr<Base> clone() const override {
    return std::make_unique<Derived>(a, b);
  }
  virtual double virtual_method(double x) const override { return Base::a*x + b; }
};

/// END OF FIXME!


PYBIND11_MODULE(expression, m) {

  m.doc() = "Polynomial expressions involving quantum-mechanical operators "
            "and manipulations with them";

  //
  // dynamic_indices::dyn_indices
  //

  using dynamic_indices::dyn_indices;

  py::class_<dyn_indices>(m, "Indices",
    "Mixed sequence of integer/string indices"
  )
  .def(py::init<>(), "Construct an index sequence of zero length")
  .def(py::init<dyn_indices::indices_t>(), "Construct an index sequence")
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
      if(i < N-1) s += ",";
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
  // generator
  //

  using gen_type = generator<dyn_indices>;

  // TODO
  // Helper class for generator<dyn_indices>
  /*
  class generator_trampoline : public gen_type {
    public:
    using gen_type::gen_type;
    int algebra_id() const override {
      PYBIND11_OVERLOAD_PURE(int, gen_type, algebra_id, );
    }
    std::unique_ptr<generator> clone() const override {
      PYBIND11_OVERLOAD_PURE(std::unique_ptr<gen_type>, gen_type, clone, );
    }
    double commute(gen_type const& g2, linear_function_t & f) const override {
      PYBIND11_OVERLOAD_PURE(double, gen_type, commute, g2, f);
    }
    // TODO
  };*/

  //py::class_<gen_type>(m, "Generator",
  //  "Abstract algebra generator"
  //);
  //.def(py::init<dyn_indices>());
  //.def_property_readonly("indices", &gen_type::indices);
  //.def(py::self == py::self);
  //.def(py::self != py::self)
  //.def(py::self < py::self)
  //.def(py::self > py::self);

  class PyBase : public Base {
    public:

    using Base::Base;

    virtual std::unique_ptr<Base> clone() const override {
      // Workaround for pybind11 issue #1962
      /*py::gil_scoped_acquire gil;
      py::function overload =
        py::get_overload(static_cast<const Base*>(this), "clone");
      if(overload) {
        auto o = overload();
        return py::detail::cast_safe<std::unique_ptr<Base>>(std::move(o));
      }
      py::pybind11_fail("Tried to call pure virtual function Base::clone");
      */
      // FIXME
      //PYBIND11_OVERLOAD_PURE(std::unique_ptr<Base>, Base, clone, );
    }
    virtual double virtual_method(double x) const override {
      PYBIND11_OVERLOAD(double, Base, virtual_method, x);
    }
  };

  // FIXME
  py::class_<Base, PyBase>(m, "Base")
  .def(py::init<int>())
  .def_property_readonly("a", &Base::get_a)
  .def("clone", &Base::clone);
}
