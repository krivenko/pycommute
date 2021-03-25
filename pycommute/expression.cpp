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
#include <libcommute/expression/factories.hpp>
#include <libcommute/expression/factories_dyn.hpp>
#include <libcommute/expression/hc.hpp>

#include <algorithm>
#include <cassert>
#include <complex>
#include <functional>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>

using namespace libcommute;
namespace py = pybind11;

//
// Just because std::to_string(std::string) is not part of STL ...
//

template<typename T>
std::string to_string(T&& x) { using std::to_string; return to_string(x); }
std::string to_string(std::string const& x) { return x; }

//
// Some commonly used type shorthands
//

using dynamic_indices::dyn_indices;
using gen_type = generator<dyn_indices>;
using mon_type = monomial<dyn_indices>;

////////////////////////////////////////////////////////////////////////////////

//
// Convert Python positional arguments into dyn_indices::indices_t
//

dyn_indices::indices_t args2indices_t(py::args args) {
  dyn_indices::indices_t v;
  v.reserve(args.size());
  for(auto const& a : args)
    v.emplace_back(a.cast<std::variant<int, std::string>>());
  return v;
}

//
// Register dynamic_indices::dyn_indices
//
// The 'Indices' objects are not the same thing as Python tuples because
// they follow a different ordering rule. Unlike with the Python tuples, two
// index sequences I1 and I2 always compare as I1 < I2 if len(I1) < len(I2).
//

void register_dyn_indices(py::module_ & m) {
  py::class_<dyn_indices>(m, "Indices",
    "Mixed sequence of integer/string indices"
  )
  .def(py::init([](py::args args) {
      return std::make_unique<dyn_indices>(args2indices_t(args));
    }),
    "Construct an index sequence from positional integer/string arguments."
   )
  .def("__len__", &dyn_indices::size, "Index sequence length.")
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
  .def("__getitem__", [](dyn_indices const& indices, std::size_t n)
      -> std::variant<int, std::string> const& {
      if(n >= indices.size())
        throw py::index_error();
      auto const& ind = static_cast<dyn_indices::indices_t const&>(indices);
      return ind[n];
    },
    "Individual index access.",
    py::return_value_policy::reference
  )
  .def("__iter__", [](const dyn_indices &indices) {
      auto const& ind = static_cast<dyn_indices::indices_t const&>(indices);
      return py::make_iterator(ind.begin(), ind.end());
    },
    py::keep_alive<0, 1>()
  );
}

////////////////////////////////////////////////////////////////////////////////

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

  gen_type_trampoline(py::args args) : gen_type(init(args)) {}

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
};

//
// Register generator<dyn_indices>
//

void register_generator(py::module_ & m) {

  py::class_<gen_type, gen_type_trampoline>(m, "Generator",
    "Abstract algebra generator"
  )
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
  .def("__repr__", [ss = std::ostringstream()](gen_type const& g) mutable {
    ss.str(std::string());
    ss.clear();
    ss << g;
    return ss.str();
  });
}

////////////////////////////////////////////////////////////////////////////////

//
// Register generator_fermion<dyn_indices>
//

void register_generator_fermion(py::module_ & m) {

  py::class_<generator_fermion<dyn_indices>, gen_type>(m, "GeneratorFermion",
    "Generator of the fermionic algebra"
  )
  .def(py::init<bool, dyn_indices const&>(),
R"=(
Construct a creation or annihilation fermionic operator with given indices.

:param dagger: ``True``/``False`` for a creation/annihilation operator
                respectively.
:param indices: Index sequence of the creation/annihilation operator.
)=",
    py::arg("dagger"), py::arg("indices")
  )
  .def_property_readonly("dagger",
                         &generator_fermion<dyn_indices>::dagger,
                         "Is this generator a creation operator?"
                        );

  m.def("make_fermion", [](bool dagger, py::args args) {
    return generator_fermion<dyn_indices>(dagger, args2indices_t(args));
  },
R"=(
Make a creation or annihilation fermionic operator with indices passed as
positional arguments.

:param dagger: ``True``/``False`` for a creation/annihilation operator
                respectively.
:param *args: Indices of the creation/annihilation operator.
)=",
    py::arg("dagger")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register generator_boson<dyn_indices>
//

void register_generator_boson(py::module_ & m) {

  py::class_<generator_boson<dyn_indices>, gen_type>(m, "GeneratorBoson",
    "Generator of the bosonic algebra"
  )
  .def(py::init<bool, dyn_indices const&>(),
R"=(
Construct a creation or annihilation bosonic operator with given indices.

:param dagger: ``True``/``False`` for a creation/annihilation operator
                respectively.
:param indices: Index sequence of the creation/annihilation operator.
)=",
    py::arg("dagger"), py::arg("indices"))
  .def_property_readonly("dagger",
                         &generator_boson<dyn_indices>::dagger,
                         "Is this generator a creation operator?");

  m.def("make_boson", [](bool dagger, py::args args) {
    return generator_boson<dyn_indices>(dagger, args2indices_t(args));
  },
R"=(
Make a creation or annihilation bosonic operator with indices passed as
positional arguments.

:param dagger: ``True``/``False`` for a creation/annihilation operator
                respectively.
:param *args: Indices of the creation/annihilation operator.
)=",
    py::arg("dagger")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register generator_spin<dyn_indices>
//

void register_generator_spin(py::module_ & m) {

  py::enum_<spin_component>(m, "SpinComponent",
    "Enumeration: spin operator component, :math:`S_+`, :math:`S_-` "
    "or :math:`S_z`.")
    .value("PLUS",
           spin_component::plus,
           "Label for the spin raising operators :math:`S_+ = S_x + i S_y`"
          )
    .value("MINUS",
           spin_component::minus,
           "Label for the spin lowering operators :math:`S_- = S_x - i S_y`"
          )
    .value("Z",
           spin_component::z,
           "Label for the 3rd spin projection operators :math:`S_z`"
          );

  py::class_<generator_spin<dyn_indices>, gen_type>(m, "GeneratorSpin",
    "Generator of the spin algebra"
  )
  .def(py::init<spin_component, dyn_indices const&>(),
R"=(
Construct a spin-1/2 operator corresponding to a single spin component and
carrying given indices.

:param c: Which spin component to construct, one of :py:class:`SpinComponent`
          values.
:param indices: Index sequence of the spin component operator.

)=",
    py::arg("c"), py::arg("indices")
  )
  .def(py::init<double, spin_component, dyn_indices const&>(),
R"=(
Construct an operator for a general spin :math:`S` corresponding to a single
spin component and carrying given indices.

:param spin: Integer or half-integer value of spin :math:`S`.
:param c: Which spin component to construct, one of :py:class:`SpinComponent`
          values.
:param indices: Index sequence of the spin component operator.
)=",
    py::arg("spin"), py::arg("c"), py::arg("indices")
  )
  .def_property_readonly("multiplicity",
                         &generator_spin<dyn_indices>::multiplicity, R"=(
Multiplicity :math:`2S+1` of the spin algebra this generator belongs to.)="
  )
  .def_property_readonly("spin", &generator_spin<dyn_indices>::spin, R"=(
Spin :math:`S` of the algebra this generator belongs to.)="
  )
  .def_property_readonly("component",
                         &generator_spin<dyn_indices>::component, R"=(
Whether this generator :math:`S_+`, :math:`S_-` or :math:`S_z`?)="
  );

  m.def("make_spin", [](spin_component c, py::args args) {
    return generator_spin<dyn_indices>(c, args2indices_t(args));
  },
R"=(
Make a spin-1/2 operator corresponding to a single spin component and carrying
indices passed as positional arguments.

:param c: Which spin component to construct, one of :py:class:`SpinComponent`
          values.
:param *args: Indices of the operator.
)=",
    py::arg("c")
);
  m.def("make_spin", [](double spin, spin_component c, py::args args) {
    return generator_spin<dyn_indices>(spin, c, args2indices_t(args));
  },
R"=(
Make an operator for a general spin :math:`S` corresponding to a single spin
component and carrying indices passed as positional arguments.

:param spin: Integer or half-integer value of spin :math:`S`.
:param c: Which spin component to construct, one of :py:class:`SpinComponent`
          values.
:param *args: Indices of the operator.
)=",
    py::arg("spin"), py::arg("c")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register monomial<dyn_indices>
//

void register_monomial(py::module_ & m) {
  py::class_<mon_type>(m, "Monomial",
                       "Monomial: a product of algebra generators")
  .def(py::init<>(),
       "Construct an identity monomial (a product of zero generators).")
  .def(py::init<std::vector<gen_type*> const&>(),
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
  .def("__repr__", [ss = std::ostringstream()](mon_type const& m) mutable {
    ss.str(std::string());
    ss.clear();
    ss << m;
    return ss.str();
  })
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
R"=(
Swap two generators at given positions within the monomial.

:param n1: Position of the first generator to be swapped.
:param n2: Position of the second generator to be swapped.
)=",
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
    }, py::is_operator()
  )
  .def("__rmul__", [](mon_type const& m, gen_type const& g) {
      return concatenate(g, m);
    }, py::is_operator()
  )
  .def("__mul__", [](mon_type const& m1, mon_type const& m2) {
      return concatenate(m1, m2);
    }, py::is_operator()
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register expression<ScalarType, dyn_indices>
//

template<typename ScalarType>
auto register_expression(py::module_ & m,
                         std::string const& class_name,
                         std::string const& docstring) {
  using expr_t = expression<ScalarType, dyn_indices>;
  py::class_<expr_t> c(m, class_name.c_str(), docstring.c_str());

  // Constructors
  c
  .def(py::init<>(), "Construct a zero expression.")
  .def(py::init<ScalarType const&>(),
R"=(
Construct a constant expression equal to :math:`x`.

:param x: Constant :math:`x`.
)=",
       py::arg("x"))
  .def(py::init<ScalarType const&, mon_type>(),
R"=(
Construct an expression with one monomial, :math:`x \cdot m`.

:param x: Coefficient :math:`x`.
:param m: Monomial :math:`m`.
)=",
       py::arg("x"), py::arg("m")
  )
  // Accessors
  .def("__len__", &expr_t::size, "Number of monomials in this expression.")
  .def("clear", &expr_t::clear, "Reset expression to zero.")
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
  .def("__repr__", [ss = std::ostringstream()](expr_t const& e) mutable {
    ss.str(std::string());
    ss.clear();
    ss << e;
    return ss.str();
  })
  // Iterator over monomials
  .def("__iter__", [](const expr_t &e) {
      return py::make_iterator(e.get_monomials().begin(),
                               e.get_monomials().end());
    },
    py::keep_alive<0, 1>()
  );

  // Hermitian conjugate
  m.def("conj",
        [](expr_t const& expr) { return conj(expr); },
R"=(
Hermitian conjugate.

:param expr: Expression to conjugate.
)=",
        py::arg("expr")
  );

  // transform()
  using f_t = std::function<ScalarType(mon_type const& m, ScalarType coeff)>;

  std::string f_doc_text = std::is_same_v<ScalarType, double> ?
R"=(
:param f: Transformation function. It must take two arguments, one of type
          :py:class:`Monomial` and one :py:class:`float` argument.
)=" :
R"=(
:param f: Transformation function. It must take two arguments, one of type
          :py:class:`Monomial` and one :py:class:`complex` argument.
)=";

  m.def("transform",
        [](expr_t const& expr, f_t const& f) { return transform(expr, f); },
        (std::string(
R"=(
Apply a given function to all monomial/coefficient pairs and replace the
coefficients with values returned by the function. Zero return values will
result in respective monomials being discarded.

:param expr: Expression to be transformed.
)=") + f_doc_text).c_str(),
    py::arg("expr"), py::arg("f")
  );

  return c;
}

////////////////////////////////////////////////////////////////////////////////

//
// Register interactions between
//
// expression<double, dyn_indices>
//
// and
//
// expression<std::complex<double>, dyn_indices>
//

template<typename ExprR, typename ExprC>
void register_expr_mixed_real_complex(py::module_ & m,
                                      ExprR & expr_r,
                                      ExprC & expr_c) {
  using dynamic_indices::expr_real;
  using dynamic_indices::expr_complex;

  // Heterogeneous arithmetic
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
R"=(
Apply a given function to all monomial/coefficient pairs and replace the
coefficients with values returned by the function. Zero return values will
result in respective monomials being discarded.

:param expr: Expression to be transformed.
:param f: Transformation function. It must take two arguments, one of type
          :py:class:`Monomial` and one :py:class:`float` argument.
)=",
    py::arg("expr"), py::arg("f")
  );
  m.def("transform",
        [](dynamic_indices::expr_complex const& expr, std::function<
            double(mon_type const& m, std::complex<double> coeff)
           > const& f)
        { return transform(expr, f); },
R"=(
Apply a given function to all monomial/coefficient pairs and replace the
coefficients with values returned by the function. Zero return values will
result in respective monomials being discarded.

:param expr: Expression to be transformed.
:param f: Transformation function. It must take two arguments, one of type
          :py:class:`Monomial` and one :py:class:`complex` argument.
)=",
    py::arg("expr"), py::arg("f")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register factory functions
//

void register_factories(py::module_ & m) {
  using dynamic_indices::expr_real;
  using dynamic_indices::expr_complex;

  // Fermions
  m
  .def("c_dag", [](py::args args) -> expr_real {
      return static_indices::c_dag(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a fermionic creation operator with indices passed as positional
arguments.

:param *args: Indices of the operator.
)="
  )
  .def("c", [](py::args args) -> expr_real {
      return static_indices::c(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a fermionic annihilation operator with indices passed as positional
arguments.

:param *args: Indices of the operator.
)="
  )
  .def("n", [](py::args args) -> expr_real {
      return static_indices::n(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a fermionic particle number operator with indices passed as positional
arguments.

:param *args: Indices of the operator.
)="
  );
  // Bosons
  m
  .def("a_dag", [](py::args args) -> expr_real {
      return static_indices::a_dag(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a bosonic creation operator with indices passed as positional arguments.

:param *args: Indices of the operator.
)="
  )
  .def("a", [](py::args args) -> expr_real {
      return static_indices::a(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a bosonic annihilation operator with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  );

  // Spin 1/2
  m
  .def("S_p", [&](py::args args) -> expr_real {
      return static_indices::S_p(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a spin-1/2 raising operator :math:`S_+` with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  )
  .def("S_m", [](py::args args) -> expr_real {
      return static_indices::S_m(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a spin-1/2 lowering operator :math:`S_-` with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  )
  .def("S_x", [](py::args args) -> expr_complex {
      return static_indices::S_x(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a spin-1/2 x-projection operator :math:`S_x` with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  )
  .def("S_y", [](py::args args) -> expr_complex {
      return static_indices::S_y(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a spin-1/2 y-projection operator :math:`S_y` with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  )
  .def("S_z", [](py::args args) -> expr_real {
      return static_indices::S_z(dyn_indices(args2indices_t(args)));
    },
R"=(
Returns a spin-1/2 z-projection operator :math:`S_z` with indices passed as
positional arguments.

:param *args: Indices of the operator.
)="
  );
  // Arbitrary spin

  // FIXME: This lambda is a temporary solution.
  // It should be replaced by py::arg("spin") = 0.5 preceded by py::kw_only()
  // as soon as that option can be used together with py::args.
  auto extract_spin_arg = [](py::kwargs const& kwargs) -> double {
    if(kwargs.size() == 1 && kwargs.contains("spin"))
      return kwargs["spin"].cast<double>();
    else
      throw std::invalid_argument("Unexpected keyword argument");
  };

  auto valudate_spin = [](double spin) {
    if(2*spin != int(spin*2))
      throw std::invalid_argument(
        "Spin must be either integer or half-integer"
      );
  };

  m
  .def("S_p", [=](py::args args, py::kwargs kwargs) -> expr_real {
      double spin = extract_spin_arg(kwargs);
      valudate_spin(spin);
      return expr_real(1.0, mon_type(
        static_indices::make_spin(spin,
                                  spin_component::plus,
                                  dyn_indices(args2indices_t(args)))
      ));
    },
R"=(
Returns a general spin raising operator :math:`S_+` with indices passed as
positional arguments.

:param spin: Spin :math:`S`.
:param *args: Indices of the operator.
)="
  )
  .def("S_m", [=](py::args args, py::kwargs kwargs) -> expr_real {
      double spin = extract_spin_arg(kwargs);
      valudate_spin(spin);
      return expr_real(1.0, mon_type(
        static_indices::make_spin(spin,
                                  spin_component::minus,
                                  dyn_indices(args2indices_t(args)))
      ));
    },
R"=(
Returns a general spin lowering operator :math:`S_-` with indices passed as
positional arguments.

:param spin: Spin :math:`S`.
:param *args: Indices of the operator.
)="
  )
  .def("S_x", [=](py::args args, py::kwargs kwargs) -> expr_complex {
      double spin = extract_spin_arg(kwargs);
      valudate_spin(spin);
      auto indices = dyn_indices(args2indices_t(args));

      using spin_component::plus;
      using spin_component::minus;

      return 0.5 * (
        expr_real(1.0, mon_type(static_indices::make_spin(spin, plus, indices)
        )) +
        expr_real(1.0, mon_type(static_indices::make_spin(spin, minus, indices)
        ))
      );
    },
R"=(
Returns a general spin :math:`S` x-projection operator :math:`S_x` with indices
passed as positional arguments.

:param spin: Spin :math:`S`.
:param *args: Indices of the operator.
)="
  )
  .def("S_y", [=](py::args args, py::kwargs kwargs) -> expr_complex {
      double spin = extract_spin_arg(kwargs);
      valudate_spin(spin);
      auto indices = dyn_indices(args2indices_t(args));

      using spin_component::plus;
      using spin_component::minus;
      using namespace std::complex_literals;

      return -0.5i * (
        expr_real(1.0, mon_type(static_indices::make_spin(spin, plus, indices)
        )) -
        expr_real(1.0, mon_type(static_indices::make_spin(spin, minus, indices)
        ))
      );
    },
R"=(
Returns a general spin :math:`S` y-projection operator :math:`S_y` with indices
passed as positional arguments.

:param spin: Spin :math:`S`.
:param *args: Indices of the operator.
)="
  )
  .def("S_z", [=](py::args args, py::kwargs kwargs) -> expr_real {
      double spin = extract_spin_arg(kwargs);
      valudate_spin(spin);
      return expr_real(1.0, mon_type(
        static_indices::make_spin(spin, spin_component::z,
                                  dyn_indices(args2indices_t(args)))
      ));
    },
R"=(
Returns a general spin :math:`S` z-projection operator :math:`S_z` with indices
passed as positional arguments.

:param spin: Spin :math:`S`.
:param *args: Indices of the operator.
)="
  );

  //
  // make_complex()
  //

  m.def("make_complex", &dynamic_indices::make_complex,
R"=(
Make a complex expression out of a real one.

:param expr: Real expression to be complexified.
)=",
    py::arg("expr")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register 'hc'
//

void register_hc(py::module_ & m) {
  using dynamic_indices::expr_real;
  using dynamic_indices::expr_complex;

  py::class_<std::remove_const_t<decltype(hc)>>(m, "HC",
R"=(
A placeholder type that adds/subtracts the Hermitian conjugate
to/from an expression. There is a module-level constant of this type called
:py:data:`hc`, which allows to mimic the :math:`\pm H.c.` notation in
expressions.
)="
  )
  .def("__radd__", [](decltype(hc), expr_real const& e){ return e + hc; },
       py::is_operator())
  .def("__radd__", [](decltype(hc), expr_complex const& e){ return e + hc; },
       py::is_operator())
  .def("__rsub__", [](decltype(hc), expr_real const& e){ return e - hc; },
       py::is_operator())
  .def("__rsub__", [](decltype(hc), expr_complex const& e){ return e - hc; },
       py::is_operator());

  m.attr("hc") = hc;
}

////////////////////////////////////////////////////////////////////////////////

//
// 'expression' Python module
//

PYBIND11_MODULE(expression, m) {

  m.doc() = "Polynomial expressions involving quantum-mechanical operators "
            "and manipulations with them";

  register_dyn_indices(m);

  //
  // Algebra IDs
  //

  m.attr("FERMION") = fermion;
  m.attr("BOSON") = boson;
  m.attr("SPIN") = spin;

  register_generator(m);

  register_generator_fermion(m);
  register_generator_boson(m);
  register_generator_spin(m);

  register_monomial(m);

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

  register_expr_mixed_real_complex(m, expr_r, expr_c);

  register_factories(m);
  register_hc(m);
}
