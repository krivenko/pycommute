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
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "numpy_state_vectors.hpp"

#include <libcommute/expression/dyn_indices.hpp>
#include <libcommute/loperator/loperator.hpp>
#include <libcommute/loperator/elementary_space_fermion.hpp>
#include <libcommute/loperator/elementary_space_boson.hpp>
#include <libcommute/loperator/elementary_space_spin.hpp>
#include <libcommute/loperator/es_constructor.hpp>
#include <libcommute/loperator/space_partition.hpp>
#include <libcommute/loperator/mapped_basis_view.hpp>

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using namespace libcommute;
namespace py = pybind11;

//
// Some commonly used type shorthands
//

using dcomplex = std::complex<double>;

using dynamic_indices::dyn_indices;
using es_type = elementary_space<dyn_indices>;
using hs_type = hilbert_space<dyn_indices>;

template<typename ScalarType>
using lop_type = loperator<ScalarType, fermion, boson, spin>;

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

////////////////////////////////////////////////////////////////////////////////

//
// Names of the two scalar types used in this module
//

template<typename ScalarType>
std::string scalar_type_name() {
  static_assert(std::is_same_v<ScalarType, double> ||
                std::is_same_v<ScalarType, dcomplex>
  );
  if constexpr(std::is_same_v<ScalarType, double>)
    return "real";
  else
    return "complex";
}

////////////////////////////////////////////////////////////////////////////////

//
// Helper classes for abstract base elementary_space<dyn_indices>
//

class es_type_trampoline : public es_type {

  dyn_indices init(py::args args) {
    dyn_indices::indices_t v;
    v.reserve(args.size());
    for(auto const& a : args)
      v.emplace_back(a.cast<std::variant<int, std::string>>());
    return dyn_indices(std::move(v));
  }

  public:

  using es_type::es_type;

  es_type_trampoline(py::args args) : es_type(init(args)) {}

  int algebra_id() const override {
    PYBIND11_OVERRIDE_PURE(int, es_type, algebra_id, );
  }

  std::unique_ptr<es_type> clone() const override {
    // elementary spaces are immutable, so one should use multiple references
    // instead of creating deep copies in Python.
    assert(false);
    return nullptr;
  }

  int n_bits() const override {
    PYBIND11_OVERRIDE(int, es_type, n_bits, );
  }
};

//
// Register elementary_space<dyn_indices>
//

void register_elementary_space(py::module_ & m) {

  py::class_<es_type, es_type_trampoline>(m, "ElementarySpace",
    "Hilbert space corresponding to one quantum degree of freedom"
  )
  // Algebra ID
  .def_property_readonly(
    "algebra_id",
    &es_type::algebra_id,
    "ID of the algebra this elementary space is associated with."
  )
  // The minimal number of binary digits needed to represent any state
  // in this elementary space
  .def_property_readonly(
    "n_bits",
    &es_type::n_bits,
    "The minimal number of binary digits needed to represent any state "
    "in this elementary space."
  )
  // Tuple of indices
  .def_property_readonly("indices", [](es_type const& es){
    return std::get<0>(es.indices());
    },
    "Indices carried by this elementary space."
  )
  // Comparison operators
  .def("__eq__",
       [](es_type const& es1, es_type const& es2){ return es1 == es2; },
       py::is_operator(),
    py::arg("es2")
  )
  .def("__ne__",
       [](es_type const& es1, es_type const& es2){ return es1 != es2; },
       py::is_operator(),
       py::arg("es2")
  )
  .def("__lt__",
       [](es_type const& es1, es_type const& es2){ return es1 < es2; },
       py::is_operator(),
       py::arg("es2")
  )
  .def("__gt__",
       [](es_type const& es1, es_type const& es2){ return es1 > es2; },
       py::is_operator(),
       py::arg("es2")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register elementary_space_fermion<dyn_indices>
//

void register_elementary_space_fermion(py::module_ & m) {

  py::class_<elementary_space_fermion<dyn_indices>, es_type>(
    m,
    "ESpaceFermion",
    "Elementary space generated by one fermionic degree of freedom"
  )
  .def(py::init<dyn_indices const&>(),
R"=(
Construct a 2-dimensional elementary space a fermionic creation/annihilation
operator acts in.

:param indices: Index sequence of the corresponding creation/annihilation operator.
)=",
    py::arg("indices")
  );

  m.def("make_space_fermion", [](py::args args) {
    return elementary_space_fermion<dyn_indices>(args2indices_t(args));
  },
R"=(
Make a fermionic elementary space with indices passed as positional arguments.

:param *args: Indices of the corresponding creation/annihilation operator.
)="
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register elementary_space_boson<dyn_indices>
//

void register_elementary_space_boson(py::module_ & m) {

  py::class_<elementary_space_boson<dyn_indices>, es_type>(
    m,
    "ESpaceBoson",
    "Truncated :math:`2^n`-dimensional elementary space generated by one "
    "bosonic degree of freedom"
  )
  .def(py::init<int, dyn_indices const&>(),
R"=(
Construct a :math:`2^\text{n_bits}`-dimensional elementary space a bosonic
creation/annihilation operator acts in.

:param n_bits: Base-2 logarithm of the required space dimension.
:param indices: Index sequence of the creation/annihilation operator.
)=",
    py::arg("n_bits"), py::arg("indices"));

  m.def("make_space_boson", [](int n_bits, py::args args) {
    return elementary_space_boson<dyn_indices>(n_bits, args2indices_t(args));
  },
R"=(
Make a bosonic elementary space with indices passed as positional arguments.

:param n_bits: Base-2 logarithm of the required space dimension.
:param *args: Indices of the corresponding creation/annihilation operator.
)=",
    py::arg("n_bits")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register register_elementary_space_spin<dyn_indices>
//

void register_elementary_space_spin(py::module_ & m) {

  py::class_<elementary_space_spin<dyn_indices>, es_type>(
    m,
    "ESpaceSpin",
    "Elementary space generated by one spin degree of freedom"
  )
  .def(py::init<double, dyn_indices const&>(),
R"=(
Construct an elementary space spin-:math:`S` operators act in.

:param spin: Integer or half-integer value of spin :math:`S`.
:param indices: Index sequence of the corresponding spin operator.
)=",
    py::arg("spin"), py::arg("indices")
  );

  m.def("make_space_spin", [](double spin, py::args args) {
    return elementary_space_spin<dyn_indices>(spin, args2indices_t(args));
  },
R"=(
Make a spin elementary space with indices passed as positional arguments.

:param spin: Integer or half-integer value of spin :math:`S`.
:param *args: Indices of the corresponding spin operator.
)=",
    py::arg("spin")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register hilbert_space<dyn_indices>
//

void register_hilbert_space(py::module_ & m) {

  using dynamic_indices::expr_real;
  using dynamic_indices::expr_complex;

  py::class_<hs_type>(m, "HilbertSpace",
    "Hilbert space as a direct product of elementary spaces"
  )
  .def(py::init<>(), "Construct an empty Hilbert space.")
  .def(py::init<std::vector<es_type*> const&>(),
    "Construct from a list to elementary spaces.",
    py::arg("elementary_spaces")
  )
  .def(py::init(
    [](expr_real const& expr, int bits_per_boson) {
      return hs_type(expr, boson_es_constructor(bits_per_boson));
    }),
R"=(
Inspect a real polynomial expression and collect elementary spaces associated to
every algebra generator found in the expression.

:param expr: Polynomial expression to inspect.
:param bits_per_boson: Base-2 logarithm of the dimension of every bosonic
       elementary space to be constructed.
)=",
    py::arg("expr"), py::arg("bits_per_boson") = 1
  )
  .def(py::init(
    [](expr_complex const& expr, int bits_per_boson) {
      return hs_type(expr, boson_es_constructor(bits_per_boson));
    }),
R"=(
Inspect a complex polynomial expression and collect elementary spaces associated
to every algebra generator found in the expression.

:param expr: Polynomial expression to inspect.
:param bits_per_boson: Base-2 logarithm of the dimension of every bosonic
       elementary space to be constructed.
)=",
    py::arg("expr"), py::arg("bits_per_boson") = 1
  )
  .def(py::self == py::self, py::arg("hs"))
  .def(py::self != py::self, py::arg("hs"))
  .def("add",
    &hs_type::add,
R"=(
Add a new elementary space into the direct product.

:param es: Elementary space to add.
)=",
    py::arg("es")
  )
  .def("__contains__",
    &hs_type::has,
    "Is a given elementary space part of the direct product?",
    py::arg("es")
  )
  .def("index",
    &hs_type::index,
    "Position of a given elementary space in the product.",
    py::arg("es")
  )
  .def("__len__",
    &hs_type::size,
    "Number of elementary spaces in the direct product."
  )
  .def_property_readonly("dim",
    &hs_type::dim,
    "Dimension of this Hilbert space, :math:`2^\\text{total_n_bits}`."
  )
  .def("bit_range",
    &hs_type::bit_range,
R"=(
Bit range spanned by a given elementary space.

:param es: Elementary space.
)=",
    py::arg("es")
  )
  .def("algebra_bit_range",
    &hs_type::algebra_bit_range,
R"=(
Bit range spanned by a given algebra ID.

:param algebra_id: Algebra ID.
)=",
  py::arg("algebra_id")
  )
  .def_property_readonly("total_n_bits",
    &hs_type::total_n_bits,
R"=(The minimal number of binary digits needed to represent any state in this
Hilbert space.)="
  )
  .def("basis_state_index",
    &hs_type::basis_state_index,
R"=(
Returns index of the product basis state, which decomposes over bases of
the elementary spaces as
:math:`|0\rangle |0\rangle \ldots |0\rangle |n\rangle_\text{es} |0\rangle
\ldots |0\rangle`.

:param es: Elementary space corresponding to the arbitrary index in
           the decomposition.
:param n: Index of the basis state within the selected elementary space.
)=", py::arg("es"), py::arg("n")
  );

  // foreach()
  using f_t = std::function<void(sv_index_type n)>;

  m.def("foreach",
        [](hs_type const& hs, f_t const& f) { return foreach(hs, f); },
R"=(
Apply a given functor to all basis state indices in a Hilbert space.

:param hs: Hilbert space in question.
:param f: Functor to be applied.
)=",
    py::arg("hs"), py::arg("f")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register loperator<ScalarType, fermion, boson, spin>::operator()()
//

template<typename ScalarType, typename SrcScalarType, typename DstScalarType>
void register_loperator_act(py::class_<lop_type<ScalarType>> & lop) {

  std::string src_vector_text = scalar_type_name<SrcScalarType>();
  std::string dst_vector_text = scalar_type_name<DstScalarType>();

  auto docstring = "\nAct on a " + src_vector_text +
    " state vector and write the result into another " + dst_vector_text +
    " state vector.\n" +
    R"=(
:param src: Source state vector.
:param dst: Destination state vector.
)=";

  lop.def("__call__",
    [](lop_type<ScalarType> const& op,
       py::array_t<SrcScalarType, 0> src,
       py::array_t<DstScalarType, 0> dst
      ) {
      py::buffer_info src_buf = src.request();
      py::buffer_info dst_buf = dst.request();
      if(src_buf.ndim != 1)
        throw std::runtime_error(
          "Source state vector must be a 1-dimensional array"
        );
      if(dst_buf.ndim != 1)
        throw std::runtime_error(
          "Destination state vector must be a 1-dimensional array"
        );
      op(src, dst);
    },
    docstring.c_str(),
    py::arg("src"), py::arg("dst").noconvert()
  );
}

//
// Register loperator<ScalarType, fermion, boson, spin>::operator*()
//

template<typename ScalarType, typename StateScalarType>
void register_loperator_mul(py::class_<lop_type<ScalarType>> & lop) {
  using dst_scalar_type = mul_type<ScalarType, StateScalarType>;

  auto docstring = "Act on a " + scalar_type_name<StateScalarType>() +
    " state vector and return the resulting vector.";

  lop.def("__mul__",
    [](lop_type<ScalarType> const& op, py::array_t<StateScalarType, 0> sv) ->
      py::array_t<dst_scalar_type> {
      py::buffer_info sv_buf = sv.request();
      if(sv_buf.ndim != 1)
        throw std::runtime_error("State vector must be a 1-dimensional array");
      auto dst = py::array_t<dst_scalar_type>(sv_buf.size);
      op(sv, dst);
      return dst;
    },
    docstring.c_str(),
    py::is_operator(), py::arg("sv")
  );
}

//
// Register loperator<ScalarType, fermion, boson, spin>
//

template<typename ScalarType>
py::class_<lop_type<ScalarType>> register_loperator(
  py::module_ & m,
  std::string const& class_name,
  std::string const& docstring
) {
  py::class_<lop_type<ScalarType>> lop(m,
                                       class_name.c_str(),
                                       docstring.c_str());

  // Constructor
  lop.def(
    py::init<expression<ScalarType, dyn_indices> const&, hs_type const&>(),
R"=(
Construct the linear operator representing action of a given polynomial
expression on a Hilbert space.

:param expr: Source polynomial expression.
:param hs: Hilbert space the linear operator acts on.
)=",
    py::arg("expr"), py::arg("hs")
  );

  // Multiplication (action on a state vector)
  register_loperator_mul<ScalarType, double>(lop);
  register_loperator_mul<ScalarType, dcomplex>(lop);

  // Call operator (in-place action on a state vector)
  if constexpr(std::is_same_v<ScalarType, double>)
    register_loperator_act<ScalarType, double, double>(lop);
  register_loperator_act<ScalarType, double, dcomplex>(lop);
  register_loperator_act<ScalarType, dcomplex, dcomplex>(lop);

  return lop;
}

////////////////////////////////////////////////////////////////////////////////

//
// Register make_space_partition()
//

template<typename ScalarType>
void register_make_space_partition(py::module_ & m) {
  m.def(
    "make_space_partition",
    [](lop_type<ScalarType> const& h,
       hs_type const& hs,
       bool store_matrix_elements)
      -> std::pair<space_partition, matrix_elements_map<ScalarType>>
    {
      if(store_matrix_elements) {
        matrix_elements_map<ScalarType> matrix_elements;
        return {space_partition(h, hs, matrix_elements),
                std::move(matrix_elements)};
      } else
        return {space_partition(h, hs), matrix_elements_map<ScalarType>{}};
    },
R"=(
Constructs a partition of a finite-dimensional Hilbert space into a direct sum
of invariant subspaces of a given Hermitian operator.

This function can optionally collect non-vanishing matrix elements
:math:`H_{ij}` of the Hermitian operator and return them in a form of
a dictionary ``{(i, j) : H_ij}``.

:param h: Hermitian operator (Hamiltonian) :math:`\hat H` used to partition
          the space.
:param hs: Hilbert space to partition.
:param store_matrix_elements: Collect the non-vanishing matrix elements of
                              :py:data:`h`.
:return: A tuple containing the constructed :py:class:`SpacePartition` object
         and a dictionary with the collected matrix elements (an empty
         dictionary when ``store_matrix_elements = False``).
)=",
    py::arg("h"), py::arg("hs"), py::arg("store_matrix_elements") = true
  );
}

//
// Register space_partition::merge_subspaces()
//

template<typename ScalarType>
void register_merge_subspaces(py::class_<space_partition> & sp) {
  sp.def("merge_subspaces",
  &space_partition::merge_subspaces<hs_type, ScalarType, fermion, boson, spin>,
R"=(
Merge some of the invariant subspaces to ensure that a given operator
:math:`\hat O` and its Hermitian conjugate :math:`\hat O^\dagger` generate only
one-to-one connections between the subspaces.

This function can optionally collect non-vanishing matrix elements
of :math:`\hat O` and :math:`\hat O^\dagger` and return them in a form of
dictionaries ``{(i, j) : O_ij}``.

:param od: Operator :math:`\hat O^\dagger`.
:param o: Operator :math:`\hat O`.
:param hs: Hilbert space used to construct the original space partition.
:param store_matrix_elements: Collect the non-vanishing matrix elements of
                              :py:data:`od` and :py:data:`o`.
:return: A tuple containing dictionaries with the collected matrix elements
         of :math:`\hat O^\dagger` and :math:`\hat O` (empty dictionaries when
         ``store_matrix_elements = False``).
)=",
    py::arg("od"), py::arg("o"), py::arg("hs"),
    py::arg("store_matrix_elements") = true
  );
}

//
// Register space_partition
//

void register_space_partition(py::module_ & m) {
  py::class_<space_partition> sp(
    m,
    "SpacePartition",
R"=(Partition of a Hilbert space into a set of disjoint subspaces invariant
under action of a given Hermitian operator (Hamiltonian).

For a detailed description of the algorithm see
`Computer Physics Communications 200, March 2016, 274-284
<http://dx.doi.org/10.1016/j.cpc.2015.10.023>`_ (section 4.2).
)="
  );

  sp.def(py::init<lop_type<double> const&, hs_type const&>(),
R"=(
Partition a finite-dimensional Hilbert space into a direct sum of invariant
subspaces of a Hermitian operator.

:param h: Hermitian operator (Hamiltonian) used to partition the space.
:param hs: Hilbert space to partition.
)=",
    py::arg("h"), py::arg("hs")
  )
  .def(py::init<lop_type<dcomplex> const&, hs_type const&>(),
R"=(
Partition a finite-dimensional Hilbert space into a direct sum of invariant
subspaces of a Hermitian operator.

:param h: Hermitian operator (Hamiltonian) used to partition the space.
:param hs: Hilbert space to partition.
)=",
    py::arg("h"), py::arg("hs")
  )
  .def_property_readonly(
    "dim",
     &space_partition::dim,
   "Dimension of the original Hilbert space used to construct this partition."
  )
  .def_property_readonly(
    "n_subspaces",
    &space_partition::n_subspaces,
    "Number of invariant subspaces in this partition."
  )
  .def(
    "__getitem__",
    [](space_partition const& partition, sv_index_type index) {
      if(index >= partition.dim())
        throw std::out_of_range("Unexpected basis state index " +
                                std::to_string(index));
      return partition[index];
    },
    "Find what invariant subspace a given basis state belongs to.",
    py::arg("basis_state_index")
  );

  register_merge_subspaces<double>(sp);
  register_merge_subspaces<dcomplex>(sp);

  register_make_space_partition<double>(m);
  register_make_space_partition<dcomplex>(m);

  using f_t = std::function<void(sv_index_type n, sv_index_type sp_index)>;

  m.def("foreach",
        [](space_partition const& sp, f_t const& f) { return foreach(sp, f); },
R"=(
Apply a given functor to all basis states in a given space partition.
The functor must take two arguments, index of the basis state and index of
the subspace this basis state belongs to.

:param sp: Space partition in question.
:param f: Functor to be applied.
)=",
    py::arg("sp"), py::arg("f")
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register mapped_basis_view()
//

template<typename ScalarType>
void register_mapped_basis_view(py::module_ & m, std::string const& classname) {

  std::string docstring = "This object is a view of a " +
                          scalar_type_name<ScalarType>() +
R"=( state vector (one-dimensional NumPy array) that performs index translation
according to a predefined map. It is accepted by methods of linear operator
objects :py:func:`LOperatorR.__call__()` and :py:func:`LOperatorC.__call__()`.

:py:class:`)=" + classname + R"=(` can be used in situations where a linear
operator is known to act only within a subspace of a full Hilbert space, and
it is desirable to store vector components only within this particular subspace.
The relevant components are then stored in a NumPy array, while the view object
translates indices of basis states from the full Hilbert space to the smaller
subspace.
)=";

  py::class_<mapped_basis_view<py::array_t<ScalarType, 0>, false>>(
    m,
    classname.c_str(),
    docstring.c_str()
  );
}

//
// Register action of linear operators on MappedBasisView objects
//

template<typename SrcScalarType, typename DstScalarType, typename LOpType>
void register_loperator_call_mbv(py::class_<LOpType> & lop) {

  using scalar_t = typename LOpType::scalar_type;
  using src_mbv_t = mapped_basis_view<py::array_t<SrcScalarType, 0>, false>;
  using dst_mbv_t = mapped_basis_view<py::array_t<DstScalarType, 0>, false>;

  std::string src_vector_text = scalar_type_name<SrcScalarType>();
  std::string dst_vector_text = scalar_type_name<DstScalarType>();

  auto docstring = "\nAct on a mapped view of a " + src_vector_text +
    " state vector and write the result through a view of another " +
    dst_vector_text +
    " state vector.\n" +
    R"=(
:param src: View of the source state vector.
:param dst: View of the destination state vector.
)=";

  lop.def("__call__",
    [](lop_type<scalar_t> const& op, src_mbv_t const& src, dst_mbv_t & dst) {
      op(src, dst);
    },
    docstring.c_str(),
    py::arg("src").noconvert(), py::arg("dst").noconvert()
  );
}

////////////////////////////////////////////////////////////////////////////////

//
// Register basis_mapper
//

void register_basis_mapper(py::module_ & m) {
  py::class_<basis_mapper>(
    m,
    "BasisMapper",
    "Factory class for :py:class:`MappedBasisViewR` and "
    ":py:class:`MappedBasisViewC` objects"
  )
  .def(py::init<std::vector<sv_index_type> const&>(),
R"=(
Build a mapping from a list of basis state indices to their positions within
the list.

:param basis_state_indices: List of the basis state indices.
)=",
    py::arg("basis_state_indices")
  )
  .def(py::init<lop_type<double> const&, hs_type const&>(),
R"=(
Build a mapping from a set of all basis states contributing to
:math:`\hat O|0\rangle`, where :math:`|0\rangle` is the product basis state
corresponding to zero index in each elementary space.

.. math::

  |0\rangle = |0\rangle_1 |0\rangle_2 |0\rangle_3 \ldots.

:param O: Real-valued linear operator :math:`\hat O`.
:param hs: Hilbert space :math:`\hat O` acts in.
)=",
    py::arg("O"), py::arg("hs")
  )
  .def(py::init<lop_type<dcomplex> const&, hs_type const&>(),
R"=(
Build a mapping from a set of all basis states contributing to
:math:`\hat O|0\rangle`, where :math:`|0\rangle` is the product basis state
corresponding to zero index in each elementary space.

.. math::

  |0\rangle = |0\rangle_1 |0\rangle_2 |0\rangle_3 \ldots.

:param O: Complex-valued linear operator :math:`\hat O`.
:param hs: Hilbert space :math:`\hat O` acts in.
)=",
    py::arg("O"), py::arg("hs")
  )
  .def(py::init<std::vector<lop_type<double>> const&, hs_type const&, int>(),
R"=(
Given a list of operators
:math:`\{\hat O_1, \hat O_2, \hat O_3, \ldots, \hat O_M\}`, build a mapping
including all basis states that contribute to all states
:math:`\hat O_1^{n_1} \hat O_2^{n_2} \ldots \hat O_M^{n_M} |0\rangle`.

:math:`|0\rangle` is the product basis state corresponding to zero index
in each elementary space,

.. math::

  |\hat 0\rangle = |0\rangle_1 |0\rangle_2 |0\rangle_3 \ldots,

and the non-negative integers :math:`n_m` satisfy :math:`\sum_{m=1}^M n_m = N`.
Mapped values are assigned continuously but without any specific order.

:param O_list: List of real-valued linear operators :math:`\{\hat O_m\}`.
:param hs: Hilbert space operators :math:`\hat O_m` act in.
:param N: Total power :math:`N`.
)=",
    py::arg("O_list"), py::arg("hs"), py::arg("N")
  )
  .def(py::init<std::vector<lop_type<dcomplex>> const&, hs_type const&, int>(),
R"=(
Given a list of operators
:math:`\{\hat O_1, \hat O_2, \hat O_3, \ldots, \hat O_M\}`, build a mapping
including all basis states that contribute to all states
:math:`\hat O_1^{n_1} \hat O_2^{n_2} \ldots \hat O_M^{n_M} |0\rangle`.

:math:`|0\rangle` is the product basis state corresponding to zero index
in each elementary space,

.. math::

  |\hat 0\rangle = |0\rangle_1 |0\rangle_2 |0\rangle_3 \ldots,

and the non-negative integers :math:`n_m` satisfy :math:`\sum_{m=1}^M n_m = N`.
Mapped values are assigned continuously but without any specific order.

:param O_list: List of complex-valued linear operators :math:`\{\hat O_m\}`.
:param hs: Hilbert space operators :math:`\hat O_m` act in.
:param N: Total power :math:`N`.
)=",
    py::arg("O_list"), py::arg("hs"), py::arg("N")
  )
  .def("__call__", &basis_mapper::make_view_no_ref<py::array_t<double, 0>>,
R"=(
Make a basis mapping view of a real state vector (1-dimensional NumPy array).

:param sv: The state vector to make the view of.
)=",
    py::arg("sv"), py::keep_alive<1, 0>()
  )
  .def("__call__", &basis_mapper::make_view_no_ref<py::array_t<dcomplex, 0>>,
R"=(
Make a basis mapping view of a complex state vector (1-dimensional NumPy array).

:param sv: The state vector to make the view of.
)=",
    py::arg("sv"), py::keep_alive<1, 0>()
  )
  .def("__len__",
       &basis_mapper::size,
       "Number of basis states in the mapping."
  )
  .def_property_readonly(
    "map",
     &basis_mapper::map,
    "Direct access to the mapping as Dict[int, int]."
  )
  .def_property_readonly(
    "inverse_map",
    &basis_mapper::inverse_map,
    "Direct access to the inverse mapping as Dict[int, int]. (slow!)"
   );
}

////////////////////////////////////////////////////////////////////////////////

//
// 'loperator' Python module
//

PYBIND11_MODULE(loperator, m) {

  m.doc() = "Linear operators in finite-dimensional Hilbert spaces";

  register_elementary_space(m);

  register_elementary_space_fermion(m);
  register_elementary_space_boson(m);
  register_elementary_space_spin(m);

  register_hilbert_space(m);

  auto lop_real = register_loperator<double>(
    m, "LOperatorR", "Real-valued linear operator"
  );
  auto lop_complex = register_loperator<dcomplex>(
    m, "LOperatorC", "Complex-valued linear operator"
  );

  register_space_partition(m);

  register_mapped_basis_view<double>(m, "MappedBasisViewR");
  register_mapped_basis_view<dcomplex>(m, "MappedBasisViewC");

  register_loperator_call_mbv<double, double>(lop_real);
  register_loperator_call_mbv<double, dcomplex>(lop_real);
  register_loperator_call_mbv<dcomplex, dcomplex>(lop_real);
  register_loperator_call_mbv<double, dcomplex>(lop_complex);
  register_loperator_call_mbv<dcomplex, dcomplex>(lop_complex);

  register_basis_mapper(m);
}
