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


def tight_binding(matrix_elements: np.ndarray,
                  indices: Sequence[IndicesType],
                  statistics: int = FERMION
                  ) -> Union[ExpressionR, ExpressionC]:
    """
    TODO
    """
    assert matrix_elements.ndim == 2, "TODO"
    assert matrix_elements.shape[0] == len(indices), "TODO"
    assert matrix_elements.shape[1] == len(indices), "TODO"

    assert statistics in (FERMION, BOSON), "TODO"

    is_complex = np.iscomplexobj(matrix_elements.dtype)
    H = ExpressionC() if is_complex else ExpressionR()

    O, O_dag = (c, c_dag) if statistics == FERMION else (a, a_dag)
    for (i1, ind1), (i2, ind2) in product(enumerate(indices),
                                          enumerate(indices)):
        H += matrix_elements[i1, i2] * O_dag(*ind1) * O(ind2)

    return H


def ising(J: np.array,
          h_l: np.array = None,
          h_t: np.array = None,
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
    assert N == J.shape[1]

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


def heisenberg(J: np.array,
               h: np.array = None,
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
    assert N == J.shape[1]

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


def anisotropic_heisenberg(J: Tuple[np.array, np.array, np.array],
                           h: np.array = None,
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
