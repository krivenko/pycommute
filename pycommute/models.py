#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2019-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

__all__ = ["tight_binding", "heisenberg"]

from .expression import (
    FERMION, BOSON,
    ExpressionR, ExpressionC,
    c, c_dag, a, a_dag, S_p, S_m, S_x, S_y, S_z
)

from typing import Union, Tuple, Sequence
from itertools import product
import numpy as np


IndicesType = Tuple[Union[int, str], ...]


def tight_binding(t: np.ndarray,
                  sites: Sequence[IndicesType] = None,
                  *,
                  statistics: int = FERMION
                  ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Make a tight-binding lattice Hamiltonian from a matrix of hopping
    elements :math:`t`.

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} t_{ij} \hat O^\dagger_{i} \hat O_{j}.

    By default, the corresponding lattice is a collection of sites
    with integer indices :math:`i = 0, \ldots, N-1`. Operators
    :math:`O^\dagger_i` and :math:`\hat O_j` can be fermionic or bosonic
    creation/annihilation operators.
    The on-site energies are given by the diagonal elements of the hopping
    matrix :math:`t_{ij}`.

    :param t: An :math:`N\times N` matrix of hopping elements :math:`t_{ij}`.
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`.
                  Dimensions of :obj:`t` must agree with
                  the length of :obj:`sites`.
    :param statistics: Statistics of the particles in question, either
                       :attr:`pycommute.expression.FERMION` or
                       :attr:`pycommute.expression.BOSON`.
    """
    assert t.ndim == 2
    N = t.shape[0]
    assert t.shape == (N, N)
    assert statistics in (FERMION, BOSON)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    H = ExpressionC() if np.iscomplexobj(t) else ExpressionR()

    O, O_dag = (c, c_dag) if statistics == FERMION else (a, a_dag)

    with np.nditer(t, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue

            site_i = sites[it.multi_index[0]]
            site_j = sites[it.multi_index[1]]

            H += x * O_dag(*site_i) * O(*site_j)

    return H


# TODO: dispersion(eps: np.ndarray (1D), ...)


def ising(J: np.ndarray,
          h_l: np.ndarray = None,
          h_t: np.ndarray = None,
          sites: Sequence[IndicesType] = None,
          *,
          spin: float = 1 / 2
          ) -> ExpressionR:
    r"""
    Make Hamiltonian of the quantum Ising model on a finite lattice
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
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`. Dimensions of :obj:`J` and
                  :obj:`h_l`/:obj:`h_t` must agree with the length
                  of :obj:`sites`.
    :param spin: Spin of operators :math:`\hat{S}^\alpha_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    is_complex = np.iscomplexobj(J) or \
        np.iscomplexobj(h_l) or (h_t is not None)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue

            site_i = sites[it.multi_index[0]]
            site_j = sites[it.multi_index[1]]

            H += -x * S_z(*site_i, spin=spin) * S_z(*site_j, spin=spin)

    if h_l is not None:
        assert h_l.shape == (N,)
        for i, h_i in enumerate(h_l):
            site = sites[i]
            H += -h_i * S_z(*site, spin=spin)
    if h_t is not None:
        assert h_t.shape == (N,)
        for i, h_i in enumerate(h_t):
            site = sites[i]
            H += -h_i * S_x(*site, spin=spin)

    return H


def heisenberg(J: np.ndarray,
               h: np.ndarray = None,
               *,
               sites: Sequence[IndicesType] = None,
               spin: float = 1 / 2
               ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Make Hamiltonian of the quantum Heisenberg model on a finite lattice
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
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`. Dimensions of :obj:`J` and
                  :obj:`h` must agree with the length of :obj:`sites`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    is_complex = np.iscomplexobj(J) or (h is not None)
    H = ExpressionC() if is_complex else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue

            site_i = sites[it.multi_index[0]]
            site_j = sites[it.multi_index[1]]

            H += -x * (
                S_z(*site_i, spin=spin) * S_z(*site_j, spin=spin)
                + 0.5 * S_p(*site_i, spin=spin) * S_p(*site_j, spin=spin)
                + 0.5 * S_m(*site_i, spin=spin) * S_m(*site_j, spin=spin)
            )

    if h is not None:
        assert h.shape == (N, 3)
        for i, h_i in enumerate(h):
            site = sites[i]
            H += -np.dot(h_i, (S_x(*site, spin=spin),
                               S_y(*site, spin=spin),
                               S_z(*site, spin=spin)))

    return H


def anisotropic_heisenberg(J: Tuple[np.ndarray, np.ndarray, np.ndarray],
                           h: np.ndarray = None,
                           *,
                           sites: Sequence[IndicesType] = None,
                           spin: float = 1 / 2
                           ) -> ExpressionC:
    r"""
    Make Hamiltonian of the anisotropic quantum Heisenberg model on
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
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`. Dimensions of components of
                  :obj:`J` and of :obj:`h` must agree with the length of
                  :obj:`sites`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert len(J) == 3
    Jx, Jy, Jz = J

    N = J[0].shape[0]
    assert Jx.shape == (N, N)
    assert Jy.shape == (N, N)
    assert Jz.shape == (N, N)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    H = ExpressionC()

    def add_J_terms(H, J, Sa):
        with np.nditer(J, flags=['multi_index']) as it:
            for x in it:
                if x == 0:
                    continue
                site_i = sites[it.multi_index[0]]
                site_j = sites[it.multi_index[1]]
                H += -x * Sa(*site_i, spin=spin) * Sa(*site_j, spin=spin)

    add_J_terms(H, Jx, S_x)
    add_J_terms(H, Jy, S_y)
    add_J_terms(H, Jz, S_z)

    if h is not None:
        assert h.shape == (N, 3)
        for i, h_i in enumerate(h):
            site = sites[i]
            H += -np.dot(h_i, (S_x(*site, spin=spin),
                               S_y(*site, spin=spin),
                               S_z(*site, spin=spin)))

    return H


def biquadratic_spin_int(J: np.ndarray,
                         *,
                         sites: Sequence[IndicesType] = None,
                         spin: float = 1
                         ) -> Union[ExpressionR, ExpressionC]:
    r"""
    Make a biquadratic spin interaction term on a finite lattice
    (collection of sites :math:`i = 0, \ldots, N-1`),

    .. math::

        \hat H = \sum_{i,j = 0}^{N-1} J_{ij}
                 \left(\hat{\mathbf{S}}_i \cdot \hat{\mathbf{S}}_j\right)^2

    :param J: An :math:`N\times N` matrix of Heisenberg coupling constants
              :math:`J_{ij}`.
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`. Dimensions of :obj:`J`
                  must agree with the length of :obj:`sites`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1 by default.
    :return: Interaction term :math:`\hat H`.
    """
    assert J.ndim == 2
    N = J.shape[0]
    assert J.shape == (N, N)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    H = ExpressionC() if np.iscomplexobj(J) else ExpressionR()

    with np.nditer(J, flags=['multi_index']) as it:
        for x in it:
            if x == 0:
                continue

            site_i = sites[it.multi_index[0]]
            site_j = sites[it.multi_index[1]]

            SS = (S_z(*site_i, spin=spin) * S_z(*site_j, spin=spin)
                  + 0.5 * S_p(*site_i, spin=spin) * S_p(*site_j, spin=spin)
                  + 0.5 * S_m(*site_i, spin=spin) * S_m(*site_j, spin=spin))
            H += x * SS * SS

    return H


def dzyaloshinskii_moriya(D: np.ndarray,
                          *,
                          sites: Sequence[IndicesType] = None,
                          spin: float = 1 / 2
                          ) -> ExpressionC:
    r"""
    Make Dzyaloshinskii–Moriya interaction Hamiltonian on
    a finite lattice (collection of sites :math:`i = 0, \ldots, N-1`).

    .. math::

        \hat H = \sum_{i,j=0}^{N-1} \mathbf{D}_{ij} \cdot
                 (\hat{\mathbf{S}}_i \times \hat{\mathbf{S}}_j).

    :param D: An :math:`N\times N \times 3` array, whose slices
              ``D[i,j,:]`` are vectors :math:`\mathbf{D}_{ij}`.
    :param sites: An optional list of site names to be used instead of the
                  simple numeric indices :math:`i`. Dimensions of :obj:`D`
                  must agree with the length of :obj:`sites`.
    :param spin: Spin of operators :math:`\hat{\mathbf{S}}_i`, 1/2 by default.
    :return: Hamiltonian :math:`\hat H`.
    """
    assert D.ndim == 3
    N = D.shape[0]
    assert D.shape == (N, N, 3)

    if sites is None:
        sites = list(map(lambda i: (i,), range(N)))
    else:
        assert len(sites) == N

    H = ExpressionC()

    for (i, site_i), (j, site_j) in product(enumerate(sites),
                                            enumerate(sites)):
        D_vec = D[i, j, :]
        if (D_vec == 0).all():
            continue

        H += D_vec[0] * (S_y(*site_i, spin=spin) * S_z(*site_j, spin=spin)
                         - S_z(*site_i, spin=spin) * S_y(*site_j, spin=spin))
        H += D_vec[1] * (S_z(*site_i, spin=spin) * S_x(*site_j, spin=spin)
                         - S_x(*site_i, spin=spin) * S_z(*site_j, spin=spin))
        H += D_vec[2] * (S_x(*site_i, spin=spin) * S_y(*site_j, spin=spin)
                         - S_y(*site_i, spin=spin) * S_x(*site_j, spin=spin))

    return H
