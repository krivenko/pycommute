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
#include <pybind11/complex.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <libcommute/expression/expression.hpp>
#include <libcommute/expression/dyn_indices.hpp>
#include <libcommute/expression/generator_fermion.hpp>
#include <libcommute/expression/generator_boson.hpp>
#include <libcommute/expression/generator_spin.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
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
// Use operator<< to create a string representation
//

template<typename T> std::string print(T const& obj) {
  std::ostringstream ss;
  ss << obj;
  return ss.str();
}

//
// Some commonly used type shorthands
//

using dynamic_indices::dyn_indices;
using gen_type = generator<dyn_indices>;
using mon_type = monomial<dyn_indices>;

//
// Helper classes for abstract base generator<dyn_indices>
//

class gen_type_trampoline : public gen_type {

  dyn_indices init(py::args args) {
    dyn_indices::indices_t v;
    v.reserve(args.size());
    for(auto const& a : args)
      v.emplace_back(a.cast<std::variant<int, std::string>>());
    return dyn_indices(std::move(v));
  }

  public:

  using gen_type::gen_type;

  gen_type_trampoline(py::args args) : gen_type(std::move(init(args))) {}

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

  void conj(linear_function_t & f) const override {
    PYBIND11_OVERRIDE(void, gen_type, conj, f);
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

class gen_type_publicist : public gen_type {
public:
  using gen_type::equal;
  using gen_type::less;
  using gen_type::greater;
};

//
// Convert Python positional arguments to dyn_indices::indices_t
//

dyn_indices::indices_t args2indices_t(py::args args) {
  dyn_indices::indices_t v;
  v.reserve(args.size());
  for(auto const& a : args)
    v.emplace_back(a.cast<std::variant<int, std::string>>());
  return v;
}

//
// Wrap expression<ScalarType, dyn_indices>
//

template<typename ScalarType>
auto register_expression(py::module_ & m,
                         std::string const& class_name,
                         std::string const& docstring) {
  using expr_t = expression<ScalarType, dyn_indices>;
  py::class_<expr_t> c(m, class_name.c_str(), docstring.c_str());

  using monomials_map_t = typename expr_t::monomials_map_t;

  // Constructors
  c
  .def(py::init<>(), "Construct a zero expression.")
  .def(py::init<ScalarType const&>(),
       "Construct a constant expression.",
       py::arg("x"))
  .def(py::init<ScalarType const&, mon_type>(),
       "Construct an expression with one monomial, x*m.",
       py::arg("x"), py::arg("m")
  )
  // Accessors
  .def("__len__", &expr_t::size, "Number of monomials in this expression.")
  .def_property_readonly(
    "monomials",
    static_cast<monomials_map_t const&(expr_t::*)() const>(
      &expr_t::get_monomials
    ),
    "List of monomials."
  )
  .def("clear", &expr_t::clear, "Reset expression to zero")
  // Homogeneous arithmetic
  .def(py::self == py::self)
  .def(py::self != py::self)
  .def(py::self + py::self)
  .def(py::self - py::self)
  .def(py::self * py::self)
  // Compound assignments
  .def(py::self += py::self)
  .def(py::self -= py::self)
  .def(py::self *= py::self)
  // Unary minus
  .def(-py::self)
  // Arithmetic involving constants
  .def(py::self + ScalarType{})
  .def(ScalarType{} + py::self)
  .def(py::self - ScalarType{})
  .def(ScalarType{} - py::self)
  .def(py::self * ScalarType{})
  .def(ScalarType{} * py::self)
  // Compound assignments from constants
  .def(py::self += ScalarType{})
  .def(py::self -= ScalarType{})
  .def(py::self *= ScalarType{})
  // String representation
  .def("__repr__", &print<expr_t>)
  // Iterator over monomials
  .def("__iter__", [](const expr_t &e) {
      return py::make_iterator(e.begin(), e.end());
    },
    py::keep_alive<0, 1>()
  );

  // Hermitian conjugate
  m.def("conj",
        [](expr_t const& e) { return conj(e); },
        "Hermitian conjugate",
        py::arg("expr")
  );

  // transform()
  using f_t = std::function<ScalarType(mon_type const& m, ScalarType coeff)>;
  m.def("transform",
        [](expr_t const& expr, f_t const& f) { return transform(expr, f); },
        R"eol(
Apply function 'f' to all monomial/coefficient pairs and replace the
coefficients with values returned by the function.)eol",
    py::arg("expr"), py::arg("f")
  );

  return c;
}

//
// 'expression' Python module
//

PYBIND11_MODULE(expression, m) {

  m.doc() = "Polynomial expressions involving quantum-mechanical operators "
            "and manipulations with them";

  //
  // dynamic_indices::dyn_indices
  //
  // The 'Indices' objects are not the same thing as Python tuples, because
  // they follow a different ordering rule. Unlike with the Python tuples, two
  // index sequences I1 and I2 always compare as I1 < I2 if len(I1) < len(I2).
  //

  py::class_<dyn_indices>(m, "Indices",
    "Mixed sequence of integer/string indices"
  )
  .def(py::init([](py::args args) {
      return std::make_unique<dyn_indices>(args2indices_t(args));
    }),
    "Construct an index sequence from positional integer/string arguments."
   )
  .def("__len__", &dyn_indices::size, "Index sequence length")
  .def(py::self == py::self)
  .def(py::self != py::self)
  .def(py::self < py::self)
  .def(py::self > py::self)
  .def_property_readonly("indices",
                         &dyn_indices::operator dyn_indices::indices_t const&,
                         "Index sequence as a list of integers and strings"
                        )
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
  // Wrap generator<dyn_indices> early on so that pybind11 knows about this type
  // at the point where generator::linear_function_t is wrapped.
  //

  py::class_<gen_type, gen_type_trampoline> Generator(m, "Generator",
    "Abstract algebra generator"
  );

  //
  // generator::linear_function_t
  //

  auto copy_terms = [](auto const& terms) {
    std::vector<std::pair<std::unique_ptr<gen_type>, double>> res;
    res.reserve(terms.size());
    for(auto const& t : terms)
      res.emplace_back(t.first->clone(), t.second);
    return res;
  };

  py::class_<gen_type::linear_function_t>(m, "LinearFunctionGen",
    "Linear combination of algebra generators plus a constant term"
  )
  .def(py::init<>(), "Construct a function that is identically zero.")
  .def(py::init<double>(), "Construct a constant.", py::arg("const_term"))
  .def(py::init([&](
    double const_term,
    std::vector<std::pair<const gen_type*, double>> const& terms) {
      return std::make_unique<gen_type::linear_function_t>(
        const_term,
        std::move(copy_terms(terms))
      );
    }),
    "Construct from a constant term and a list of coefficient/generator pairs.",
    py::arg("const_term"),
    py::arg("terms")
  )
  .def_readwrite("const_term",
                 &gen_type::linear_function_t::const_term,
                 "Constant term.")
  .def_property("terms",
    [&](gen_type::linear_function_t const& f) { return copy_terms(f.terms); },
    [&](gen_type::linear_function_t & f,
       std::vector<std::pair<const gen_type*, double>> const& terms) {
       f.terms = std::move(copy_terms(terms));
    },
    "List of pairs of algebra generators and their respective coefficients."
  )
  .def_property_readonly("vanishing", &gen_type::linear_function_t::vanishing,
    "Is this linear function identically zero?"
  );

  //
  // Algebra IDs
  //

  m.attr("FERMION") = fermion;
  m.attr("BOSON") = boson;
  m.attr("SPIN") = spin;

  //
  // generator<dyn_indices>
  //

  Generator
  // Algebra ID
  .def_property_readonly("algebra_id",
                         &gen_type::algebra_id,
                         "ID of the algebra this generator belongs to."
                        )
  // Tuple of indices
  .def_property_readonly("indices", [](gen_type const& g){
    return std::get<0>(g.indices());
    },
    "Indices carried by this generator."
  )
  // Product transformation methods
  .def("swap_with", &gen_type::swap_with, R"eol(
Given a pair of generators g1 = 'self' and g2 such that g1 > g2, swap_with()
must signal what transformation g1 * g2 -> c * g2 * g1 + f(g) should be applied
to the product g1 * g2 to put it into the canonical order.
swap_with() returns the constant 'c' and writes the linear function f(g) into
'f'. 'c' is allowed to be zero.

This method should be overridden in derived classes.)eol",
    py::arg("g2"),
    py::arg("f")
  )
  .def("simplify_prod", &gen_type::simplify_prod, R"eol(
Given a pair of generators g1 = 'self' and g2 such that g1 * g2 is in the
canonical order (g1 <= g2), optionally apply a simplifying transformation
g1 * g2 -> f(g). If a simplification is actually possible, simplify_prod()
must return True and write the linear function f(g) into 'f'.
Otherwise return False.

This method should be overridden in derived classes.)eol",
    py::arg("g2"),
    py::arg("f")
  )
  .def("reduce_power", &gen_type::reduce_power, R"eol(
Given a generator g1 = 'self' and a power > 2, optionally apply a simplifying
transformation g1^power -> f(g). If a simplification is actually possible,
reduce_power() must return True and write the linear function f(g) into 'f'.
Otherwise return False.

N.B. Simplifications for power = 2 must be carried out by simplify_prod().

This method should be overridden in derived classes.)eol",
    py::arg("power"), py::arg("f")
  )
  // Comparison methods
  .def("equal", &gen_type_publicist::equal, R"eol(
Determine whether two generators 'self' and 'g' belonging to the same algebra
are equal.

This method should be overridden in derived classes.)eol",
    py::arg("g")
  )
  .def("less", &gen_type_publicist::less, R"eol(
Determine whether two generators 'self' and 'g' belonging to the same algebra
satisfy self < g.

This method should be overridden in derived classes.)eol",
    py::arg("g")
  )
  .def("greater", &gen_type_publicist::greater, R"eol(
Determine whether two generators 'self' and 'g' belonging to the same algebra
satisfy self > g.

This method should be overridden in derived classes.)eol",
    py::arg("g")
  )
  // Hermitian conjugate
  .def("conj", &gen_type::conj, R"eol(
Return the Hermitian conjugate of this generator as a linear function of
generators via 'f'.

This method should be overridden in derived classes.)eol",
    py::arg("f")
  )
  // Comparison operators
  .def("__eq__",
       [](gen_type const& g1, gen_type const& g2){ return g1 == g2; },
       py::is_operator(),
    py::arg("g2")
  )
  .def("__ne__",
       [](gen_type const& g1, gen_type const& g2){ return g1 != g2; },
       py::is_operator(),
       py::arg("g2")
  )
  .def("__lt__",
       [](gen_type const& g1, gen_type const& g2){ return g1 < g2; },
       py::is_operator(),
       py::arg("g2")
  )
  .def("__gt__",
       [](gen_type const& g1, gen_type const& g2){ return g1 > g2; },
       py::is_operator(),
       py::arg("g2")
  )
  // String representation
  .def("__repr__", &print<gen_type>);

  // Swap generators of potentially different algebras
  m.def("swap_with", &swap_with<dyn_indices>, R"eol(
Check if 'g1' and 'g2' belong to the same algebra and call g1.swap_with(g2, f)
accordingly. Generators of different algebras always commute, and for such
generators swap_with() returns 1 and sets 'f' to the trivial function.)eol",
    py::arg("g1"), py::arg("g2"), py::arg("f")
  );

  //
  // generator_fermion<dyn_indices>
  //

  py::class_<generator_fermion<dyn_indices>, gen_type>(m, "GeneratorFermion",
    "Generator of the fermionic algebra"
  )
  .def(py::init<bool, dyn_indices const&>(), R"eol(
Construct a creation ('dagger' = True) or annihilation ('dagger' = False)
fermionic operator with given 'indices'.)eol",
    py::arg("dagger"), py::arg("indices")
  )
  .def_property_readonly("dagger",
                         &generator_fermion<dyn_indices>::dagger,
                         "Is this generator a creation operator?"
                        );

  m.def("make_fermion", [](bool dagger, py::args args) {
    return generator_fermion<dyn_indices>(dagger, args2indices_t(args));
  },
    R"eol(
Make a creation ('dagger' = True) or annihilation ('dagger' = False) fermionic
operator with indices passed as positional arguments.)eol",
    py::arg("dagger")
  );

  //
  // generator_boson<dyn_indices>
  //

  py::class_<generator_boson<dyn_indices>, gen_type>(m, "GeneratorBoson",
    "Generator of the bosonic algebra"
  )
  .def(py::init<bool, dyn_indices const&>(), R"eol(
Construct a creation ('dagger' = True) or annihilation ('dagger' = False)
bosonic operator with given 'indices'.)eol",
    py::arg("dagger"), py::arg("indices"))
  .def_property_readonly("dagger",
                         &generator_boson<dyn_indices>::dagger,
                         "Is this generator a creation operator?");

  m.def("make_boson", [](bool dagger, py::args args) {
    return generator_boson<dyn_indices>(dagger, args2indices_t(args));
  },
    R"eol(
Make a creation ('dagger' = True) or annihilation ('dagger' = False) bosonic
operator with indices passed as positional arguments.)eol",
    py::arg("dagger")
  );

  //
  // generator_spin<dyn_indices>
  //

  py::enum_<spin_component>(m, "SpinComponent",
                            "Spin operator component, S_+, S_- or S_z.")
    .value("PLUS",
           spin_component::plus,
           "Label for the spin raising operators S_+ = S_x + i S_y"
          )
    .value("MINUS",
           spin_component::minus,
           "Label for the spin lowering operators S_- = S_x - i S_y"
          )
    .value("Z",
           spin_component::z,
           "Label for the 3rd spin projection operators S_z"
          );

  py::class_<generator_spin<dyn_indices>, gen_type>(m, "GeneratorSpin",
    "Generator of the spin algebra"
  )
  .def(py::init<spin_component, dyn_indices const&>(), R"eol(
Construct a spin-1/2 operator corresponding to the spin component 'c' and
carrying given 'indices'.)eol",
    py::arg("c"), py::arg("indices")
  )
  .def(py::init<double, spin_component, dyn_indices const&>(), R"eol(
Construct an operator for a general spin S = 'spin' corresponding to the spin
component 'c' and carrying given 'indices'.)eol",
    py::arg("spin"), py::arg("c"), py::arg("indices")
  )
  .def_property_readonly("multiplicity",
                         &generator_spin<dyn_indices>::multiplicity, R"eol(
Multiplicity 2S+1 of the spin algebra this generator belongs to.)eol"
  )
  .def_property_readonly("spin", &generator_spin<dyn_indices>::spin, R"eol(
Spin S of the algebra this generator belongs to.)eol"
  )
  .def_property_readonly("component",
                         &generator_spin<dyn_indices>::component, R"eol(
Whether this generator S_+, S_- or S_z?)eol"
  );

  m.def("make_spin", [](spin_component c, py::args args) {
    return generator_spin<dyn_indices>(c, args2indices_t(args));
  }, R"eol(
Make a spin-1/2 operator corresponding to the spin component 'c' and carrying
indices passed as positional arguments.)eol",
    py::arg("c")
);
  m.def("make_spin", [](double spin, spin_component c, py::args args) {
    return generator_spin<dyn_indices>(spin, c, args2indices_t(args));
  }, R"eol(
Make an operator for a general spin S = 'spin' corresponding to the spin
component 'c' and carrying indices passed as positional arguments.)eol",
    py::arg("spin"), py::arg("c")
  );

  //
  // Generator type checks
  //

  m.def("is_fermion",
        &is_fermion<dyn_indices>,
        "Does 'g' belong to the fermionic algebra?",
        py::arg("g")
       );
  m.def("is_boson",
        &is_boson<dyn_indices>,
        "Does 'g' belong to the bosonic algebra?",
        py::arg("g")
       );
  m.def("is_spin",
        &is_spin<dyn_indices>,
        "Does 'g' belong to a spin algebra?",
        py::arg("g")
       );

  //
  // monomial<dyn_indices>
  //

  py::class_<mon_type>(m, "Monomial",
                       "Monomial: a product of algebra generators")
  .def(py::init<>(), "Construct an identity monomial.")
  .def(py::init<std::vector<gen_type*>>(),
    "Construct from a list of algebra generators."
  )
  .def("__len__", &mon_type::size, "Number of generators in this monomial.")
  // Ordering
  .def(py::self == py::self)
  .def(py::self != py::self)
  .def(py::self < py::self)
  .def(py::self > py::self)
  .def_property_readonly("is_ordered", &mon_type::is_ordered,
                         "Is this monomial canonically ordered?")
  // String representation
  .def("__repr__", &print<mon_type>)
  // Individual generator access
  .def("__getitem__", [](mon_type const& m, std::size_t n) -> gen_type const& {
      if(n >= m.size())
        throw py::index_error();
      return m[n];
    },
    "Get a generator in the monomial by its index.",
    py::return_value_policy::reference
  )
  .def("__getitem__", [](mon_type const& m, py::slice slice) {
      std::size_t start, stop, step, slicelength;
      if(!slice.compute(m.size(), &start, &stop, &step, &slicelength))
        throw py::error_already_set();
      mon_type res;
      for(std::size_t n = start; n < stop; n += step)
        res.append(m[n]);
      return res;
    },
    "Get a slice of the monomial."
  )
  .def("__contains__", [](mon_type const& m, gen_type const& g) {
    return std::find(m.begin(), m.end(), g) != m.end();
  }
  )
  // Swap two generators
  .def("swap_generators", [](mon_type & m, std::size_t n1, std::size_t n2) {
      if(n1 >= m.size() || n2 >= m.size())
        throw py::index_error();
      m.swap_generators(n1, n2);
    },
    "Swap generators at positions 'n1' and 'n2'.",
    py::arg("n1"), py::arg("n2")
  )
  // Iterators
  .def("__iter__", [](mon_type const& m) {
      return py::make_iterator(m.begin(), m.end());
    },
    py::keep_alive<0, 1>()
  )
  .def("__reverse__", [](mon_type const& m) {
      return py::make_iterator(m.rbegin(), m.rend());
    },
    py::keep_alive<0, 1>()
  )
  // Concatenation
  .def("__mul__", [](mon_type const& m, gen_type const& g) {
      return concatenate(m, g);
    }
  )
  .def("__rmul__", [](mon_type const& m, gen_type const& g) {
      return concatenate(g, m);
    }
  )
  .def("__mul__", [](mon_type const& m1, mon_type const& m2) {
      return concatenate(m1, m2);
    }
  );

  //
  // expression<double, dyn_indices>
  //
  // and
  //
  // expression<std::complex<double>, dyn_indices>
  //

  auto expr_r = register_expression<double>(m,
    "ExpressionR",
    "Polynomial in quantum-mechanical operators with real coefficients"
  );
  auto expr_c = register_expression<std::complex<double>>(m,
    "ExpressionC",
    "Polynomial in quantum-mechanical operators with complex coefficients"
  );

  expr_c.def(py::init<expression<double, dyn_indices> const&>(),
    "Construct from a real expression by complexifying its coefficients.",
    py::arg("expr")
  );

  // Heterogeneous arithmetic
  using dynamic_indices::expr_real;
  using dynamic_indices::expr_complex;
  expr_r
  .def("__add__", [](expr_real const& e1, expr_complex const& e2) {
    return e1 + e2;
  }, py::is_operator())
  .def("__sub__", [](expr_real const& e1, expr_complex const& e2) {
    return e1 - e2;
  }, py::is_operator())
  .def("__mul__", [](expr_real const& e1, expr_complex const& e2) {
    return e1 * e2;
  }, py::is_operator());
  expr_c
  .def("__add__", [](expr_complex const& e1, expr_real const& e2) {
    return e1 + e2;
  }, py::is_operator())
  .def("__sub__", [](expr_complex const& e1, expr_real const& e2) {
    return e1 - e2;
  }, py::is_operator())
  .def("__mul__", [](expr_complex const& e1, expr_real const& e2) {
    return e1 * e2;
  }, py::is_operator());
  // Compound assignments real -> complex
  expr_c
  .def(py::self += dynamic_indices::expr_real{})
  .def(py::self -= dynamic_indices::expr_real{})
  .def(py::self *= dynamic_indices::expr_real{});
  // Arithmetic involving constants
  expr_r
  .def(py::self + std::complex<double>{})
  .def(std::complex<double>{} + py::self)
  .def(py::self - std::complex<double>{})
  .def(std::complex<double>{} - py::self)
  .def(py::self * std::complex<double>{})
  .def(std::complex<double>{} * py::self);
  expr_c
  .def(py::self + double{})
  .def(double{} + py::self)
  .def(py::self - double{})
  .def(double{} - py::self)
  .def(py::self * double{})
  .def(double{} * py::self);
  // Compound assignments from real constants
  expr_c
  .def(py::self += double{})
  .def(py::self -= double{})
  .def(py::self *= double{});

  // transform() from real to complex and vice versa.
  m.def("transform",
        [](dynamic_indices::expr_real const& expr, std::function<
             std::complex<double>(mon_type const& m, double coeff)
           > const& f)
        { return transform(expr, f); },
        R"eol(
Apply function 'f' to all monomial/coefficient pairs and replace the
coefficients with values returned by the function.)eol",
    py::arg("expr"), py::arg("f")
  );
  m.def("transform",
        [](dynamic_indices::expr_complex const& expr, std::function<
            double(mon_type const& m, std::complex<double> coeff)
           > const& f)
        { return transform(expr, f); },
        R"eol(
Apply function 'f' to all monomial/coefficient pairs and replace the
coefficients with values returned by the function.)eol",
    py::arg("expr"), py::arg("f")
  );

  // TODO: hc
  // TODO: factories
}
