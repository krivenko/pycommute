#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2019-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .expression import (
    FERMION, BOSON,
    ExpressionR, ExpressionC,
    c, c_dag, n, a, a_dag, S_p, S_m, S_x, S_y, S_z
)

from typing import Union, Tuple, Sequence
from itertools import product
from math import sqrt
import numpy as np

IndicesType = Union[int, str, Tuple[Union[int, str], ...]]


def _make_default_indices(indices, N, i_min=0):
    if indices is None:
        return list(map(lambda i: (i,), range(i_min, N)))
    else:
        assert len(indices) == N
        return [((i,) if isinstance(i, (int, str)) else i) for i in indices]


def _make_default_indices_with_spin(indices, N, spin_name, i_min=0):
    if indices is None:
        return list(map(lambda i: (i, spin_name), range(i_min, N)))
    else:
        assert len(indices) == N
        return [((i,) if isinstance(i, (int, str)) else i) for i in indices]


# Wigner 3-j symbols
def _three_j_symbol(j1, m1, j2, m2, j3, m3):
    if not (m1 + m2 + m3 == 0
            and -j1 <= m1 <= j1
            and -j2 <= m2 <= j2
            and -j3 <= m3 <= j3
            and abs(j1 - j2) <= j3 <= j1 + j2):
        return 0

    fact = np.math.factorial

    three_j_sym = -1.0 if (j1 - j2 - m3) % 2 else 1.0
    three_j_sym *= sqrt(fact(j1 + j2 - j3)
                        * fact(j1 - j2 + j3)
                        * fact(-j1 + j2 + j3)
                        / fact(j1 + j2 + j3 + 1))
    three_j_sym *= sqrt(fact(j1 - m1)
                        * fact(j1 + m1)
                        * fact(j2 - m2)
                        * fact(j2 + m2)
                        * fact(j3 - m3)
                        * fact(j3 + m3))

    t_min = max(j2 - j3 - m1, j1 - j3 + m2, 0)
    t_max = min(j1 - m1, j2 + m2, j1 + j2 - j3)

    t_sum = 0
    for t in np.arange(t_min, t_max + 1):
        t_sum += (-1.0 if t % 2 else 1.0) \
            / (fact(t) * fact(j3 - j2 + m1 + t)
                * fact(j3 - j1 - m2 + t)
                * fact(j1 + j2 - j3 - t)
                * fact(j1 - m1 - t)
                * fact(j2 + m2 - t))

    three_j_sym *= t_sum
    return three_j_sym


##############################
## Quantum many-body theory ##
##############################

def tight_binding(t: np.ndarray,
                  *,
                  indices: Sequence[IndicesType] = None,
                  statistics: int = FERMION
                  ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a tight-binding lattice Hamiltonian from a matrix of hopping
    elements :math:`t`.

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} t_{ij} \hat O^\dagger_{i} \hat O_{j}.

    By default, the corresponding lattice is a collection of sites
    with integer indices :math:`i = 0, \ldots, N-1`. Operators
    :math:`\hat O^\dagger_i` and :math:`\hat O_j` can be fermionic or bosonic
    creation/annihilation operators.
    The on-site energies are given by the diagonal elements of the hopping
    matrix :math:`t_{ij}`.

    :param t: An :math:`N\times N` matrix of hopping elements :math:`t_{ij}`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param statistics: Statistics of the particles in question, either
                       :attr:`pycommute.expression.FERMION` or
                       :attr:`pycommute.expression.BOSON`.
    :return: Tight-binding Hamiltonian :math:`\hat H`.
    """
    assert t.ndim == 2
    N = t.shape[0]
    assert t.shape == (N, N)
    assert statistics in (FERMION, BOSON)

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(t) else ExpressionR()

    O, O_dag = (c, c_dag) if statistics == FERMION else (a, a_dag)

    with np.nditer(t, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            H += x * O_dag(*ind_i) * O(*ind_j)

    return H


def dispersion(eps: np.ndarray,
               *,
               indices: Sequence[IndicesType] = None,
               statistics: int = FERMION
               ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an energy dispersion term of a system of many fermions or bosons
    from a list of energy levels.

    .. math::

        \hat H = \sum_{i=0}^{N-1} \varepsilon_i \hat O^\dagger_{i} \hat O_{i}.

    By default, individual degrees of freedom carry integer indices
    :math:`i = 0, \ldots, N-1`. Operators :math:`\hat O^\dagger_i` and
    :math:`\hat O_j` can be fermionic or bosonic creation/annihilation
    operators.

    :param eps: A length-:math:`N` list of energy levels
                :math:`\varepsilon_i`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param statistics: Statistics of the particles in question, either
                       :attr:`pycommute.expression.FERMION` or
                       :attr:`pycommute.expression.BOSON`.
    :return: Dispersion term :math:`\hat H`.
    """
    assert eps.ndim == 1
    N = eps.shape[0]
    assert statistics in (FERMION, BOSON)

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(eps) else ExpressionR()

    O, O_dag = (c, c_dag) if statistics == FERMION else (a, a_dag)

    with np.nditer(eps, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices[it.index]
            H += x * O_dag(*ind) * O(*ind)

    return H


def pairing(delta: np.ndarray,
            *,
            indices: Sequence[IndicesType] = None):
    r"""
    Constructs a pairing Hamiltonian from a matrix of pairing parameters
    :math:`\Delta`.

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} (
            \Delta_{ij} \hat c_i \hat c_j -
            \Delta^*_{ij} \hat c^\dagger_i \hat c^\dagger_j).

    :param delta: An :math:`N\times N` matrix of the pairing parameters
                   :math:`\Delta_{ij}`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :return: Pairing Hamiltonian :math:`\hat H`.
    """
    assert delta.ndim == 2
    N = delta.shape[0]
    assert delta.shape == (N, N)

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(delta) else ExpressionR()

    with np.nditer(delta, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            H += x * c(*ind_i) * c(*ind_j) \
                - np.conj(x) * c_dag(*ind_i) * c_dag(*ind_j)

    return H


def zeeman(b: np.ndarray,
           *,
           indices_up: Sequence[IndicesType] = None,
           indices_dn: Sequence[IndicesType] = None
           ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a Zeeman coupling term describing a system of :math:`N` electrons
    (spin-1/2 fermions) in an external magnetic field,

    .. math::

        \hat H = 2 \sum_{i=0}^{N-1} \hat{\mathbf{S}}_i \cdot \mathbf{B}_i
               = \sum_{i=0}^{N-1} \sum_{\sigma,\sigma'=\uparrow,\downarrow}
                 (\boldsymbol{\tau}_{\sigma\sigma'} \cdot \mathbf{B}_i)
                 \hat c^\dagger_{i,\sigma} \hat c_{i,\sigma'}.

    The pre-factor 2 is the spin Landé factor, while the Bohr magneton and the
    Planck constant are set to unity. :math:`\boldsymbol{\tau}` is a vector of
    Pauli matrices, and operators
    :math:`\hat c^\dagger_{i,\sigma}`/:math:`\hat c_{i,\sigma}`
    create/annihilate electrons at site :math:`i` with the 3-rd spin projection
    :math:`\sigma`.

    :param b: One of the following:

        - An :math:`N\times 3` matrix, whose rows are the local magnetic
          field vectors :math:`\mathbf{b}_i = \{b^x_i, b^y_i, b^z_i\}`.
        - A length-:math:`N` vector of 3-rd projections :math:`b^z_i`.
          It is assumed that :math:`b^x_i = b^y_i = 0` in this case.

    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.

    :return: Zeeman coupling term :math:`\hat H`.
    """
    N = b.shape[0]
    assert b.shape == (N,) or b.shape == (N, 3)

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")

    is_complex = np.iscomplexobj(b) or (b.ndim == 2)
    H = ExpressionC() if is_complex else ExpressionR()

    if b.ndim == 1:  # Only the b^z component
        with np.nditer(b, flags=['c_index']) as it:
            for x in it:
                if x == 0:
                    continue
                ind_up, ind_dn = indices_up[it.index], indices_dn[it.index]
                H += x * (n(*ind_up) - n(*ind_dn))
    else:
        for i, h_i in enumerate(b):
            ind_up = indices_up[i]
            ind_dn = indices_dn[i]
            H += h_i[0] * (c_dag(*ind_dn) * c(*ind_up)
                           + c_dag(*ind_up) * c(*ind_dn))
            H += 1j * h_i[1] * (c_dag(*ind_dn) * c(*ind_up)
                                - c_dag(*ind_up) * c(*ind_dn))
            H += h_i[2] * (n(*ind_up) - n(*ind_dn))

    return H


def hubbard_int(
    U: np.ndarray,
    *,
    indices_up: Sequence[IndicesType] = None,
    indices_dn: Sequence[IndicesType] = None
) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an interaction term of the Fermi-Hubbard model defined on
    a finite lattice (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \sum_{i=0}^{N-1} U_i \hat n_{i,\uparrow} \hat n_{i,\downarrow}.

    :param U: A length-:math:`N` vector of Hubbard interaction parameters
              :math:`U_i`.
    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.

    :return: Interaction term :math:`\hat H`.
    """
    assert U.ndim == 1
    N = U.shape[0]

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")

    H = ExpressionC() if np.iscomplexobj(U) else ExpressionR()

    with np.nditer(U, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_up, ind_dn = indices_up[it.index], indices_dn[it.index]
            H += x * n(*ind_up) * n(*ind_dn)

    return H


def bose_hubbard_int(U: np.ndarray,
                     *,
                     indices: Sequence[IndicesType] = None
                     ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an interaction term of the Bose-Hubbard model defined on a finite
    lattice (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \sum_{i=0}^{N-1} U_i
                 \hat a^\dagger_i \hat a_i (\hat a^\dagger_i \hat a_i - 1).

    :param U: A length-:math:`N` vector of Hubbard interaction parameters
              :math:`U_i`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :return: Interaction term :math:`\hat H`.
    """
    assert U.ndim == 1
    N = U.shape[0]

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(U) else ExpressionR()

    with np.nditer(U, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            index = indices[it.index]
            H += x * a_dag(*index) * a(*index) * (a_dag(*index) * a(*index) - 1)

    return H


def extended_hubbard_int(
    V: np.ndarray,
    *,
    indices_up: Sequence[IndicesType] = None,
    indices_dn: Sequence[IndicesType] = None
) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an interaction term of the extended Fermi-Hubbard model defined
    on a finite lattice (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \frac{1}{2} \sum_{i,j=0}^{N-1} V_{ij} \hat n_i \hat n_j,

    where :math:`\hat n_i = \hat n_{i,\uparrow} + \hat n_{i,\downarrow}` is
    the total electron occupation number at site :math:`i`. The local part of
    the interaction is encoded in the diagonal matrix elements :math:`V_{ii}`.

    :param V: An :math:`N\times N` matrix of interaction parameters
              :math:`V_{ij}`.
    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.

    :return: Interaction term :math:`\hat H`.
    """
    assert V.ndim == 2
    N = V.shape[0]
    assert V.shape == (N, N)

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")

    H = ExpressionC() if np.iscomplexobj(V) else ExpressionR()

    with np.nditer(V, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i_up = indices_up[it.multi_index[0]]
            ind_i_dn = indices_dn[it.multi_index[0]]
            ind_j_up = indices_up[it.multi_index[1]]
            ind_j_dn = indices_dn[it.multi_index[1]]
            H += 0.5 * x * (n(*ind_i_up) + n(*ind_i_dn)) * \
                           (n(*ind_j_up) + n(*ind_j_dn))

    return H


def t_j_int(
    J: np.ndarray,
    *,
    indices_up: Sequence[IndicesType] = None,
    indices_dn: Sequence[IndicesType] = None
) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an interaction term of the t-J model defined on a finite lattice
    (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} J_{ij}
            \left(
                \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j
                - \frac{1}{4} \hat n_i \hat n_j
            \right),

    where

    .. math::

        \hat{\mathbf{S}}_i = \sum_{\sigma\sigma'}
            \frac{\boldsymbol{\tau}_{\sigma\sigma'}}{2}
            \hat c^\dagger_{i,\sigma} \hat c_{i,\sigma'}

    is the spin vector of the electron localized at site :math:`i`, and
    :math:`\hat n_i = \hat n_{i,\uparrow} + \hat n_{i,\downarrow}` is
    the total electron occupation number at site :math:`i`.

    :param J: An :math:`N\times N` matrix of coupling parameters
              :math:`J_{ij}`.
    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.
    :return: Interaction term :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")

    H = ExpressionC() if np.iscomplexobj(J) else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue

            ind_i_up = indices_up[it.multi_index[0]]
            ind_i_dn = indices_dn[it.multi_index[0]]
            ind_j_up = indices_up[it.multi_index[1]]
            ind_j_dn = indices_dn[it.multi_index[1]]

            spi = c_dag(*ind_i_up) * c(*ind_i_dn)
            smi = c_dag(*ind_i_dn) * c(*ind_i_up)
            szi = 0.5 * (n(*ind_i_up) - n(*ind_i_dn))
            spj = c_dag(*ind_j_up) * c(*ind_j_dn)
            smj = c_dag(*ind_j_dn) * c(*ind_j_up)
            szj = 0.5 * (n(*ind_j_up) - n(*ind_j_dn))

            H += x * (0.5 * (spi * smj + smi * spj) + szi * szj)
            H -= x * 0.25 * (n(*ind_i_up) + n(*ind_i_dn)) \
                          * (n(*ind_j_up) + n(*ind_j_dn))

    return H


def kondo_int(
    J: np.ndarray,
    *,
    indices_up: Sequence[IndicesType] = None,
    indices_dn: Sequence[IndicesType] = None,
    indices_spin: Sequence[IndicesType] = None,
    spin: float = 1 / 2
) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an interaction term of the Kondo lattice model defined on
    a finite collection of sites :math:`i = 0, \ldots, N-1`,

    .. math::

        \hat H = \sum_{i=0}^{N-1} J_i
                \hat{\mathbf{s}}_i \cdot \hat{\mathbf{S}}_i,

    where

    .. math::

        \hat{\mathbf{s}}_i = \sum_{\sigma\sigma'}
            \frac{\boldsymbol{\tau}_{\sigma\sigma'}}{2}
            \hat c^\dagger_{i,\sigma} \hat c_{i,\sigma'}

    is the spin vector of the electron localized at site :math:`i`.

    :param J: A length-:math:`N` vector of coupling constants :math:`J_i`.
    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators
                       :math:`\hat c^\dagger_{i,\uparrow}`/
                       :math:`\hat c_{i,\uparrow}`. By default, the spin-up
                       operators carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators
                       :math:`\hat c^\dagger_{i,\downarrow}`/
                       :math:`\hat c_{i,\downarrow}`. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.
    :param indices_spin: An optional list of :math:`N` (multi-)indices to label
                         the localized spin operators
                         :math:`\hat{\mathbf{S}}_i`. By default, the localized
                         spin operators carry indices ``0, 1, ...``.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Interaction term :math:`\hat H`.
    """
    assert J.ndim == 1
    N = J.shape[0]

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")
    indices_spin = _make_default_indices(indices_spin, N)

    H = ExpressionC() if np.iscomplexobj(J) else ExpressionR()

    with np.nditer(J, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_up = indices_up[it.index]
            ind_dn = indices_dn[it.index]
            ind_spin = indices_spin[it.index]
            sp = c_dag(*ind_up) * c(*ind_dn)
            sm = c_dag(*ind_dn) * c(*ind_up)
            sz = 0.5 * (n(*ind_up) - n(*ind_dn))
            H += x * (0.5 * (sp * S_m(*ind_spin, spin=spin)
                             + sm * S_p(*ind_spin, spin=spin))
                      + sz * S_z(*ind_spin, spin=spin))

    return H


def holstein_int(
    g: np.ndarray,
    *,
    indices_up: Sequence[IndicesType] = None,
    indices_dn: Sequence[IndicesType] = None,
    indices_boson: Sequence[IndicesType] = None
) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs an electron-phonon coupling term of the Holstein model defined on
    a finite collection of sites :math:`i = 0, \ldots, N-1`,

    .. math::

        \hat H = \sum_{i=0}^{N-1}\sum_\sigma
                g_i \hat n_{i,\sigma} (\hat a^\dagger_i + \hat a_i).

    :param g: A length-:math:`N` vector of coupling constants :math:`g_i`.
    :param indices_up: An optional list of :math:`N` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`N` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.
    :param indices_boson: An optional list of :math:`N` (multi-)indices to label
                          the localized boson (phonon) operators
                          :math:`\hat a_i`. By default, the localized
                          boson operators carry indices ``0, 1, ...``.
    :return: Electron-phonon coupling term :math:`\hat H`.
    """
    assert g.ndim == 1
    N = g.shape[0]

    indices_up = _make_default_indices_with_spin(indices_up, N, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, N, "dn")
    indices_boson = _make_default_indices(indices_boson, N)

    H = ExpressionC() if np.iscomplexobj(g) else ExpressionR()

    with np.nditer(g, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_up = indices_up[it.index]
            ind_dn = indices_dn[it.index]
            ind_boson = indices_boson[it.index]
            H += x * (n(*ind_up) + n(*ind_dn)) \
                * (a_dag(*ind_boson) + a(*ind_boson))

    return H


def quartic_int(U: np.ndarray,
                *,
                indices: Sequence[IndicesType] = None,
                statistics: int = FERMION
                ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a general 4-fermion or 4-boson interaction Hamiltonian

    .. math::

        \hat H = \frac{1}{2}\sum_{ijkl=0}^{N-1} U_{ijkl}
            \hat O^\dagger_i \hat O^\dagger_j \hat O_l \hat O_k.

    Operators :math:`\hat O^\dagger_i` and :math:`\hat O_j` can be fermionic
    or bosonic creation/annihilation operators.

    :param U: An :math:`N\times N\times N\times N` tensor of interaction matrix
              elements :math:`U_{ijkl}`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param statistics: Statistics of the particles in question, either
                       :attr:`pycommute.expression.FERMION` or
                       :attr:`pycommute.expression.BOSON`.
    :return: Interaction Hamiltonian :math:`\hat H`.
    """
    assert U.ndim == 4
    N = U.shape[0]
    assert U.shape == (N, N, N, N)
    assert statistics in (FERMION, BOSON)

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(U) else ExpressionR()

    O, O_dag = (c, c_dag) if statistics == FERMION else (a, a_dag)

    with np.nditer(U, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            ind_k = indices[it.multi_index[2]]
            ind_l = indices[it.multi_index[3]]
            H += 0.5 * x * O_dag(*ind_i) * O_dag(*ind_j) * O(*ind_l) * O(*ind_k)

    return H


def kanamori_int(M: int,
                 U: Union[float, complex],
                 J: Union[float, complex],
                 Up: Union[float, complex] = None,
                 Jx: Union[float, complex] = None,
                 Jp: Union[float, complex] = None,
                 *,
                 indices_up: Sequence[IndicesType] = None,
                 indices_dn: Sequence[IndicesType] = None
                 ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a generalized Kanamori multiorbital Hamiltonian with :math:`M`
    orbitals,

    .. math::

        \hat H = U \sum_{m=0}^{M-1} \hat n_{m,\uparrow} \hat n_{m,\downarrow}
            + U' \sum_{m\neq m'=0}^{M-1}
                \hat n_{m,\uparrow} \hat n_{m',\downarrow}
            + (U'-J) \sum_{m<m'}^{M-1} \sum_\sigma
                \hat n_{m,\sigma} \hat n_{m',\sigma}\\
            + J_X \sum_{m\neq m'=0}^{M-1}
                \hat d^\dagger_{m,\uparrow} \hat d^\dagger_{m',\downarrow}
                \hat d_{m,\downarrow} \hat d_{m',\uparrow}
            + J_P \sum_{m\neq m'=0}^{M-1}
                \hat d^\dagger_{m,\uparrow} \hat d^\dagger_{m,\downarrow}
                \hat d_{m',\downarrow} \hat d_{m',\uparrow}.

    This function will derive values of the optional parameters according to the
    following rules.

    - If only :math:`U` and :math:`J` are provided, then the rest of parameters
      will be defined according to :math:`U'=U-2J` and :math:`J_X=J_P=J` (this
      choice results in a rotationally invariant form of the model).
    - If :math:`U`, :math:`J` and :math:`U'` are provided, then
      :math:`J_X = J_P = J`.
    - If :math:`J_X` is provided, then :math:`J_P` must also be provided
      (and vice versa).

    :param M: Number of orbitals :math:`M`.
    :param U: Intra-orbital Coulomb repulsion :math:`U`.
    :param J: Hund coupling constant :math:`J`.
    :param Up: Inter-orbital Coulomb repulsion :math:`U'`.
    :param Jx: Spin-flip coupling constant :math:`J_X`.
    :param Jp: Pair-hopping coupling constant :math:`J_P`.
    :param indices_up: An optional list of :math:`M` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices ``(0, "up"), (1, "up"), ...``.
    :param indices_dn: An optional list of :math:`M` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices ``(0, "dn"), (1, "dn"), ...``.
    :return: Kanamori Hamiltonian :math:`\hat H`.
    """

    if Up is None:
        Up = U - 2 * J
    if (Jx is None) or (Jp is None):
        assert (Jx is None) and (Jp is None)
        Jx = Jp = J

    indices_up = _make_default_indices_with_spin(indices_up, M, "up")
    indices_dn = _make_default_indices_with_spin(indices_dn, M, "dn")

    is_complex = np.iscomplexobj(U) \
        or np.iscomplexobj(Up) \
        or np.iscomplexobj(J) \
        or np.iscomplexobj(Jx) \
        or np.iscomplexobj(Jp)

    H = ExpressionC() if is_complex else ExpressionR()

    orbs = range(M)

    # Intra-orbital interactions
    if U != 0:
        for m in orbs:
            H += U * n(*indices_up[m]) * n(*indices_dn[m])
    # Inter-orbital interactions, different spins
    if Up != 0:
        for m, mp in product(orbs, repeat=2):
            if m == mp:
                continue
            H += Up * n(*indices_up[m]) * n(*indices_dn[mp])
    # Inter-orbital interactions, equal spins
    if Up - J != 0:
        for m, mp in product(orbs, repeat=2):
            if m >= mp:
                continue
            H += (Up - J) * n(*indices_up[m]) * n(*indices_up[mp])
            H += (Up - J) * n(*indices_dn[m]) * n(*indices_dn[mp])
    # Spin flips
    if Jx != 0:
        for m, mp in product(orbs, repeat=2):
            if m == mp:
                continue
            H += Jx * c_dag(*indices_up[m]) * c_dag(*indices_dn[mp]) \
                    * c(*indices_dn[m]) * c(*indices_up[mp])
    # Pair hopping
    if Jp != 0:
        for m, mp in product(orbs, repeat=2):
            if m == mp:
                continue
            H += Jp * c_dag(*indices_up[m]) * c_dag(*indices_dn[m]) \
                    * c(*indices_dn[mp]) * c(*indices_up[mp])

    return H


def slater_int(F: np.ndarray,
               *,
               indices_up: Sequence[IndicesType] = None,
               indices_dn: Sequence[IndicesType] = None
               ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a :math:`(2L+1)`-orbital fully rotationally-invariant electron
    interaction Hamiltonian using Slater parametrization,

    .. math::

        \hat H = \frac{1}{2}
            \sum_{m_1, m_2, m_3, m_4 = -L}^{L} \sum_{\sigma, \sigma'}
            U^{\sigma, \sigma'}_{m_1, m_2, m_3, m_4}
            \hat c^\dagger_{m_1, \sigma} \hat c^\dagger_{m_2, \sigma'}
            \hat c_{m_4, \sigma'} \hat c_{m_3, \sigma}.

    The interaction tensor :math:`U` is a linear combination of :math:`L+1`
    radial integrals :math:`F_0, F_2, F_4, \ldots, F_{2L}` with coefficients
    given by the angular interaction matrix elements
    :math:`A_k(m_1, m_2, m_3, m_4)`,

    .. math::

        U^{\sigma, \sigma'}_{m_1, m_2, m_3, m_4} =
        \sum_{k=0}^{2L} F_k A_k(m_1, m_2, m_3, m_4).

    All odd-:math:`k` angular matrix elements vanish, while for the even
    :math:`k`

    .. math::

        \begin{multline}
        A_k(m_1, m_2, m_3, m_4) =\\= (2l+1)^2
        \begin{pmatrix}
            l & k & l \\
            0 & 0 & 0
        \end{pmatrix}^2
        \sum_{q=-k}^k (-1)^{m_1+m_2+q}
        \begin{pmatrix}
            l & k & l \\
         -m_1 & q & m_3
        \end{pmatrix}
        \begin{pmatrix}
            l & k  & l \\
         -m_2 & -q & m_4
        \end{pmatrix}.
        \end{multline}

    :param F: List of :math:`L+1` radial Slater integrals
              :math:`F_0, F_2, F_4, \ldots`.
    :param indices_up: An optional list of :math:`2L+1` (multi-)indices to label
                       the spin-up operators. By default, the spin-up operators
                       carry indices
                       ``(-L, "up"), (-L+1, "up"), ..., (L, "up")``.
    :param indices_dn: An optional list of :math:`2L+1` (multi-)indices to label
                       the spin-down operators. By default, the spin-down
                       operators carry indices
                       ``(-L, "dn"), (-L+1, "dn"), ..., (L, "dn")``.
    :return: Slater interaction Hamiltonian :math:`\hat H`.
    """
    assert F.ndim == 1
    L = F.shape[0] - 1

    indices_up = _make_default_indices_with_spin(indices_up,
                                                 2 * L + 1,
                                                 "up",
                                                 i_min=-L)
    indices_dn = _make_default_indices_with_spin(indices_dn,
                                                 2 * L + 1,
                                                 "dn",
                                                 i_min=-L)

    H = ExpressionC() if np.iscomplexobj(F) else ExpressionR()

    # Angular matrix element
    def A(k, m1, m2, m3, m4):
        res = 0
        for q in range(-k, k + 1):
            res += _three_j_symbol(L, -m1, k, q, L, m3) \
                * _three_j_symbol(L, -m2, k, -q, L, m4) \
                * (-1.0 if (m1 + q + m2) % 2 else 1.0)
        res *= (2 * L + 1)**2 * _three_j_symbol(L, 0, k, 0, L, 0) ** 2
        return res

    U = np.zeros((2 * L + 1,) * 4,
                 dtype=complex if np.iscomplexobj(F) else float)
    for i, f in enumerate(F):
        k = 2 * i
        for m1, m2, m3, m4 in product(range(-L, L + 1), repeat=4):
            U[m1 + L, m2 + L, m3 + L, m4 + L] += f * A(k, m1, m2, m3, m4)

    with np.nditer(U, flags=['multi_index']) as it:
        def take_ind(indices, c):
            return indices[it.multi_index[c]]

        for x in it:
            if x == 0:
                continue
            ind_up = list(map(lambda i: take_ind(indices_up, i), range(4)))
            ind_dn = list(map(lambda i: take_ind(indices_dn, i), range(4)))

            H += 0.5 * x * c_dag(*ind_up[0]) * c_dag(*ind_up[1]) \
                * c(*ind_up[3]) * c(*ind_up[2])
            H += 0.5 * x * c_dag(*ind_up[0]) * c_dag(*ind_dn[1]) \
                * c(*ind_dn[3]) * c(*ind_up[2])
            H += 0.5 * x * c_dag(*ind_dn[0]) * c_dag(*ind_up[1]) \
                * c(*ind_up[3]) * c(*ind_dn[2])
            H += 0.5 * x * c_dag(*ind_dn[0]) * c_dag(*ind_dn[1]) \
                * c(*ind_dn[3]) * c(*ind_dn[2])

    return H

#########################
## Spin lattice models ##
#########################


def ising(J: np.ndarray,
          h_l: np.ndarray = None,
          h_t: np.ndarray = None,
          *,
          indices: Sequence[IndicesType] = None,
          spin: float = 1 / 2
          ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs Hamiltonian of the quantum Ising model on a finite lattice
    (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = -\sum_{i,j = 0}^{N-1} J_{ij}
                 \hat{S}^z_i \hat{S}^z_j
                 - \sum_{i=0}^{N-1} (h^l_i \hat{S}^z_i + h^t_i \hat{S}^x_i).

    :param J: An :math:`N\times N` matrix of coupling constants :math:`J_{ij}`.
    :param h_l: A length-:math:`N` vector of the local longitudinal magnetic
                fields :math:`h^l_i`. By default, all magnetic fields are zero.
    :param h_t: A length-:math:`N` vector of the local transverse magnetic
                fields :math:`h^t_i`. By default, all magnetic fields are zero.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param spin: Spin of operators :math:`\hat{S}^\alpha_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    indices = _make_default_indices(indices, N)

    is_complex = np.iscomplexobj(J) or \
        np.iscomplexobj(h_l) or (h_t is not None)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            H += -x * S_z(*ind_i, spin=spin) * S_z(*ind_j, spin=spin)

    if h_l is not None:
        assert h_l.shape == (N,)
        for i, h_i in enumerate(h_l):
            ind = indices[i]
            H += -h_i * S_z(*ind, spin=spin)
    if h_t is not None:
        assert h_t.shape == (N,)
        for i, h_i in enumerate(h_t):
            ind = indices[i]
            H += -h_i * S_x(*ind, spin=spin)

    return H


def heisenberg(J: np.ndarray,
               h: np.ndarray = None,
               *,
               indices: Sequence[IndicesType] = None,
               spin: float = 1 / 2
               ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs Hamiltonian of the quantum Heisenberg model on a finite lattice
    (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = -\sum_{i,j = 0}^{N-1} J_{ij}
                 \hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j
                 - \sum_{i=0}^{N-1} \mathbf{h}_i \cdot \hat{\mathbf{S}}_i.

    :param J: An :math:`N\times N` matrix of Heisenberg coupling constants
              :math:`J_{ij}`.
    :param h: An :math:`N\times 3` matrix, whose rows are the local magnetic
              field vectors :math:`\mathbf{h}_i = \{h^x_i, h^y_i, h^z_i\}`.
              By default, all magnetic fields are zero.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    indices = _make_default_indices(indices, N)

    is_complex = np.iscomplexobj(J) or (h is not None)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            H += -x * (
                S_z(*ind_i, spin=spin) * S_z(*ind_j, spin=spin)
                + 0.5 * S_p(*ind_i, spin=spin) * S_m(*ind_j, spin=spin)
                + 0.5 * S_m(*ind_i, spin=spin) * S_p(*ind_j, spin=spin)
            )

    if h is not None:
        assert h.shape == (N, 3)
        for i, h_i in enumerate(h):
            ind = indices[i]
            H += -np.dot(h_i, (S_x(*ind, spin=spin),
                               S_y(*ind, spin=spin),
                               S_z(*ind, spin=spin)))

    return H


def anisotropic_heisenberg(J: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           h: np.ndarray = None,
                           *,
                           indices: Sequence[IndicesType] = None,
                           spin: float = 1 / 2
                           ) -> ExpressionC:
    r"""
    Constructs Hamiltonian of the anisotropic quantum Heisenberg model on
    a finite lattice (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = -\sum_{i,j = 0}^{N-1}\left[
                 J^x_{ij}\hat{S}^x_i \hat{S}^x_j +
                 J^y_{ij}\hat{S}^y_i \hat{S}^y_j +
                 J^z_{ij}\hat{S}^z_i \hat{S}^z_j
                 \right]
                 - \sum_{i=0}^{N-1} \mathbf{h}_i \cdot \hat{\mathbf{S}}_i.

    :param J: A triplet of :math:`N\times N` matrices of Heisenberg coupling
              constants :math:`(J^x_{ij}, J^y_{ij}, J^z_{ij})`.
    :param h: An :math:`N\times 3` matrix, whose rows are the local magnetic
              field vectors :math:`\mathbf{h}_i = \{h^x_i, h^y_i, h^z_i\}`.
              By default, all magnetic fields are zero.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert len(J) == 3
    Jx, Jy, Jz = J

    N = J[0].shape[0]
    assert Jx.shape == (N, N)
    assert Jy.shape == (N, N)
    assert Jz.shape == (N, N)

    indices = _make_default_indices(indices, N)

    H = ExpressionC()

    def add_J_terms(H, J, Sa):
        with np.nditer(J, flags=['multi_index']) as it:
            for x in it:
                if x == 0:
                    continue
                ind_i = indices[it.multi_index[0]]
                ind_j = indices[it.multi_index[1]]
                H += -x * Sa(*ind_i, spin=spin) * Sa(*ind_j, spin=spin)

    add_J_terms(H, Jx, S_x)
    add_J_terms(H, Jy, S_y)
    add_J_terms(H, Jz, S_z)

    if h is not None:
        assert h.shape == (N, 3)
        for i, h_i in enumerate(h):
            ind = indices[i]
            H += -np.dot(h_i, (S_x(*ind, spin=spin),
                               S_y(*ind, spin=spin),
                               S_z(*ind, spin=spin)))

    return H


def biquadratic_spin_int(J: np.ndarray,
                         *,
                         indices: Sequence[IndicesType] = None,
                         spin: float = 1
                         ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs a biquadratic spin interaction term on a finite lattice
    (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \sum_{i,j = 0}^{N-1} J_{ij}
                 \left(\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j\right)^2

    :param J: An :math:`N\times N` matrix of Heisenberg coupling constants
              :math:`J_{ij}`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1 by default.
    :return: Interaction term :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    indices = _make_default_indices(indices, N)

    H = ExpressionC() if np.iscomplexobj(J) else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_i = indices[it.multi_index[0]]
            ind_j = indices[it.multi_index[1]]
            SS = (S_z(*ind_i, spin=spin) * S_z(*ind_j, spin=spin)
                  + 0.5 * S_p(*ind_i, spin=spin) * S_m(*ind_j, spin=spin)
                  + 0.5 * S_m(*ind_i, spin=spin) * S_p(*ind_j, spin=spin))
            H += x * SS * SS

    return H


def dzyaloshinskii_moriya(D: np.ndarray,
                          *,
                          indices: Sequence[IndicesType] = None,
                          spin: float = 1 / 2
                          ) -> ExpressionC:
    r"""
    Constructs Dzyaloshinskii–Moriya interaction Hamiltonian on
    a finite lattice (collection of sites :math:`i = 0, \ldots, N-1`).

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} \mathbf{D}_{ij} \cdot
                 (\hat{\mathbf{S}}_i \times \hat{\mathbf{S}}_j).

    :param D: An :math:`N\times N \times 3` array, whose slices
              ``D[i,j,:]`` are vectors :math:`\mathbf{D}_{ij}`.
    :param indices: An optional list of :math:`N` (multi-)indices to be used
                    instead of the simple numeric indices :math:`i`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert D.ndim == 3
    N = D.shape[0]
    assert D.shape == (N, N, 3)

    indices = _make_default_indices(indices, N)

    H = ExpressionC()

    for (i, ind_i), (j, ind_j) in product(enumerate(indices),
                                          enumerate(indices)):
        D_vec = D[i, j, :]
        if np.all((D_vec == 0)):
            continue
        H += D_vec[0] * (S_y(*ind_i, spin=spin) * S_z(*ind_j, spin=spin)
                         - S_z(*ind_i, spin=spin) * S_y(*ind_j, spin=spin))
        H += D_vec[1] * (S_z(*ind_i, spin=spin) * S_x(*ind_j, spin=spin)
                         - S_x(*ind_i, spin=spin) * S_z(*ind_j, spin=spin))
        H += D_vec[2] * (S_x(*ind_i, spin=spin) * S_y(*ind_j, spin=spin)
                         - S_y(*ind_i, spin=spin) * S_x(*ind_j, spin=spin))

    return H

############################################
## Quantum optics and quantum dissipation ##
############################################


def spin_boson(eps: np.ndarray,
               delta: np.ndarray,
               omega: np.ndarray,
               lambda_: np.ndarray,
               *,
               indices_spin: Sequence[IndicesType] = None,
               indices_boson: Sequence[IndicesType] = None,
               spin: float = 1 / 2
               ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs Hamiltonian of the general multi-spin-boson model with :math:`N`
    spin degrees of freedom and :math:`M` bosonic modes,

    .. math::

        \hat H = \sum_{i=0}^{N-1}
            \left(
            -\epsilon_i \hat S^z_i
            +\Delta_i \hat S^x_i
            \right) +
            \sum_{m=0}^{M-1}
            \omega_m \hat a^\dagger_m \hat a_m +
            \sum_{i=0}^{N-1} \sum_{m=0}^{M-1}
            \hat S^z_i (\lambda_{im}\hat a^\dagger_m + \lambda^*_{im}\hat a_m).

    :param eps: A length-:math:`N` vector of energy biases :math:`\epsilon_i`.
    :param delta: A length-:math:`N` vector of tunneling amplitudes
                  :math:`\Delta_i`.
    :param omega: A length-:math:`M` vector of bosonic frequencies
                  :math:`\omega_m`.
    :param lambda_: An :math:`N \times M` matrix of spin-boson coupling
                    constants :math:`\lambda_{im}`.
    :param indices_spin:  An optional list of :math:`N` (multi-)indices to be
                          used instead of the simple numeric indices :math:`i`.
    :param indices_boson:  An optional list of :math:`M` (multi-)indices to be
                           used instead of the simple numeric indices :math:`m`.
    :param spin: Spin of operators :math:`\hat{S}^\alpha_i`, 1/2 by default.
    :return: Spin-boson Hamiltonian :math:`\hat H`.
    """
    assert eps.ndim == 1
    assert delta.ndim == 1
    assert omega.ndim == 1
    assert lambda_.ndim == 2
    N, M = lambda_.shape
    assert eps.shape == (N,)
    assert delta.shape == (N,)
    assert omega.shape == (M,)

    indices_spin = _make_default_indices(indices_spin, N)
    indices_boson = _make_default_indices(indices_boson, M)

    is_complex = np.iscomplexobj(eps) \
        or (not np.all((delta == 0))) \
        or np.iscomplexobj(omega) \
        or np.iscomplexobj(lambda_)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(eps, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_spin[it.index]
            H += -x * S_z(*ind, spin=spin)

    with np.nditer(delta, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_spin[it.index]
            H += x * S_x(*ind, spin=spin)

    with np.nditer(omega, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_boson[it.index]
            H += x * a_dag(*ind) * a(*ind)

    with np.nditer(lambda_, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_spin = indices_spin[it.multi_index[0]]
            ind_boson = indices_boson[it.multi_index[1]]
            H += S_z(*ind_spin, spin=spin) \
                * (x * a_dag(*ind_boson) + np.conj(x) * a(*ind_boson))

    return H


def rabi(eps: np.ndarray,
         omega: np.ndarray,
         g: np.ndarray,
         *,
         indices_atom: Sequence[IndicesType] = None,
         indices_boson: Sequence[IndicesType] = None,
         spin: float = 1 / 2
         ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs Hamiltonian of the general multi-mode multi-atom Rabi model with
    :math:`N` atomic degrees of freedom and :math:`M` cavity modes,

    .. math::

        \hat H = \sum_{i=0}^{N-1}
            \epsilon_i \hat S^z_i +
            \sum_{m=0}^{M-1}
            \omega_m \hat a^\dagger_m \hat a_m +
            \sum_{i=0}^{N-1} \sum_{m=0}^{M-1}
            g_{im} \hat S^x_i (\hat a^\dagger_m + \hat a_m).

    :param eps: A length-:math:`N` vector of atomic transition frequencies
                :math:`\epsilon_i`.
    :param omega: A length-:math:`M` vector of cavity frequencies
                  :math:`\omega_m`.
    :param g: An :math:`N \times M` matrix of atom-cavity coupling
              constants :math:`g_{im}`.
    :param indices_atom: An optional list of :math:`N` (multi-)indices to be
                         used instead of the simple numeric indices :math:`i`.
    :param indices_boson: An optional list of :math:`M` (multi-)indices to be
                          used instead of the simple numeric indices :math:`m`.
    :param spin: Pseudo-spin of operators :math:`\hat{S}^\alpha_i`,
                 1/2 by default (two-level atoms).
    :return: Spin-boson Hamiltonian :math:`\hat H`.
    """
    assert eps.ndim == 1
    assert omega.ndim == 1
    assert g.ndim == 2
    N, M = g.shape
    assert eps.shape == (N,)
    assert omega.shape == (M,)

    indices_atom = _make_default_indices(indices_atom, N)
    indices_boson = _make_default_indices(indices_boson, M)

    is_complex = np.iscomplexobj(eps) \
        or np.iscomplexobj(omega) \
        or np.iscomplexobj(g) \
        or (not np.all((g == 0)))
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(eps, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_atom[it.index]
            H += x * S_z(*ind, spin=spin)

    with np.nditer(omega, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_boson[it.index]
            H += x * a_dag(*ind) * a(*ind)

    with np.nditer(g, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_atom = indices_atom[it.multi_index[0]]
            ind_boson = indices_boson[it.multi_index[1]]
            H += x * S_x(*ind_atom, spin=spin) \
                * (a_dag(*ind_boson) + a(*ind_boson))

    return H


def jaynes_cummings(eps: np.ndarray,
                    omega: np.ndarray,
                    g: np.ndarray,
                    *,
                    indices_atom: Sequence[IndicesType] = None,
                    indices_boson: Sequence[IndicesType] = None,
                    spin: float = 1 / 2
                    ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Constructs Hamiltonian of the multi-mode multi-atom Jaynes-Cummings model
    (Tavis-Cummings model) with :math:`N` atomic degrees of freedom and
    :math:`M` cavity modes,

    .. math::

        \hat H = \sum_{i=0}^{N-1}
            \epsilon_i \hat S^z_i +
            \sum_{m=0}^{M-1}
            \omega_m \hat a^\dagger_m \hat a_m +
            \sum_{i=0}^{N-1} \sum_{m=0}^{M-1}
            (g_{im} \hat a^\dagger_m \hat S^-_i + g^*_{im} \hat a_m \hat S^+_i).

    :param eps: A length-:math:`N` vector of atomic transition frequencies
                :math:`\epsilon_i`.
    :param omega: A length-:math:`M` vector of cavity frequencies
                  :math:`\omega_m`.
    :param g: An :math:`N \times M` matrix of atom-cavity coupling
              constants :math:`g_{im}`.
    :param indices_atom: An optional list of :math:`N` (multi-)indices to be
                         used instead of the simple numeric indices :math:`i`.
    :param indices_boson: An optional list of :math:`M` (multi-)indices to be
                          used instead of the simple numeric indices :math:`m`.
    :param spin: Pseudo-spin of operators :math:`\hat{S}^\alpha_i`,
                 1/2 by default (two-level atoms).
    :return: Spin-boson Hamiltonian :math:`\hat H`.
    """
    assert eps.ndim == 1
    assert omega.ndim == 1
    assert g.ndim == 2
    N, M = g.shape
    assert eps.shape == (N,)
    assert omega.shape == (M,)

    indices_atom = _make_default_indices(indices_atom, N)
    indices_boson = _make_default_indices(indices_boson, M)

    is_complex = np.iscomplexobj(eps) \
        or np.iscomplexobj(omega) \
        or np.iscomplexobj(g)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(eps, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_atom[it.index]
            H += x * S_z(*ind, spin=spin)

    with np.nditer(omega, flags=['c_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind = indices_boson[it.index]
            H += x * a_dag(*ind) * a(*ind)

    with np.nditer(g, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue
            ind_atom = indices_atom[it.multi_index[0]]
            ind_boson = indices_boson[it.multi_index[1]]
            H += x * a_dag(*ind_boson) * S_m(*ind_atom, spin=spin) \
                + np.conj(x) * a(*ind_boson) * S_p(*ind_atom, spin=spin)

    return H
