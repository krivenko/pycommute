#
# This file is part of pycommute, Python bindings for the libcommute C++
# quantum operator algebra library.
#
# Copyright (C) 2019-2021 Igor Krivenko <igor.s.krivenko@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Union, Tuple, Dict, Sequence, Callable, Any
from itertools import product
import numpy as np


IndicesType = Tuple[Union[int, str], ...]


class Lattice:
    """
    Lattice objects describe (undirected) graphs made of named nodes.

    The adjacency matrix of a graph is stored as a sum of matrices of
    spanning subgraphs. Each of these subgraphs include all nodes of the full
    graph and a physically distinct type of edges. For example, for a
    two-dimensional square lattice two adjacency matrices are stored,
    one describing the edges between the nearest neighbors, and one for the
    next-nearest neighbors.
    """

    def __init__(self,
                 node_names: Sequence[Tuple[Union[int, str], ...]],
                 subgraph_matrices: Dict[str, np.ndarray]
                 ):
        """
        TODO
        """
        assert len(subgraph_matrices) > 0
        shapes = [subgraph_matrices[k].shape for k in subgraph_matrices]
        assert len(shapes[0]) == 2 and shapes[0][0] == shapes[0][1]
        assert all(s == shapes[0] for s in shapes[1:])

        self.node_names = list(node_names)
        self.subgraph_matrices = subgraph_matrices

    def adjacency_matrix(self, weights: Dict[str, Any]):
        """
        TODO
        """
        return sum(weights[w] * self.subgraph_matrices[w] for w in weights)


def hypercubic_lattice(shape: Tuple[int, ...],
                       index_translator:
                       Callable[[Tuple[int, ...]], IndicesType] = None,
                       periodic: Union[bool, Tuple[bool, ...]] = True
                       ) -> Lattice:
    """
    TODO
    Basis vectors: (1,0,0,...), (0,1,0,...), (0,0,1,...)
    """
    ndim = len(shape)
    assert ndim > 0
    nodes = list(product(*map(range, shape)))
    n_nodes = len(nodes)

    # Nearest neighbor, next-nearest neighbor and so on ...
    subgraph_names = ['N' * (d + 2) for d in range(ndim)]

    mats = {n: np.zeros((n_nodes, n_nodes), dtype=int) for n in subgraph_names}
    per = (periodic,) * len(shape) if isinstance(periodic, bool) else periodic
    assert len(per) == ndim, "'periodic' must have the same length as 'shape'"

    def wrap_node_indices(node):
        return tuple((i % s if p else i) for i, s, p in zip(node, shape, per))

    for d in range(1, ndim + 1):
        mat = mats['N' * (d + 1)]
        stencils = [p for p in product((0,-1,1), repeat=ndim)
                    if sum(np.abs(p)) == d]

        for n1, node1 in enumerate(nodes):
            for st in stencils:
                node2 = wrap_node_indices(np.add(node1, st))
                if any(not (0 <= i < s) for i, s in zip(node2, shape)):
                    continue
                n2 = np.ravel_multi_index(node2, shape)
                mat[n1, n2] += 1

    it = index_translator if (index_translator is not None) else (lambda i: i)
    return Lattice(map(it, nodes), mats)

def chain(N: int,
          index_translator: Callable[[int], Union[int, str]] = None,
          periodic: bool = True
          ) -> Lattice:
    """
    TODO
    Basis vector: (1)
    """
    if index_translator is not None:
        def it(indices: Tuple[int, ...]) -> IndicesType:
            return (index_translator(indices[0]),)
    else:
        it = None
    return hypercubic_lattice((N,), it, periodic)

def square_lattice(
    Nx: int, Ny: int,
    index_translator: Callable[[Tuple[int, int]], IndicesType] = None,
    periodic: bool = True
    ) -> Lattice:
    """
    TODO
    Basis vectors: (1,0), (0,1)

    """
    return hypercubic_lattice((Nx, Ny), index_translator, periodic)

def cubic_lattice(
    Nx: int, Ny: int, Nz: int,
    index_translator: Callable[[Tuple[int, int, int]], IndicesType] = None,
    periodic: bool = True
    ) -> Lattice:
    """
    TODO
    Basis vectors: (1,0,0), (0,1,0), (0,0,1)
    """
    return hypercubic_lattice((Nx, Ny, Nz), index_translator, periodic)

def triangular_lattice(
    cluster: str,
    index_translator: Callable[[Tuple[int, int]], IndicesType] = None,
    **kwargs
    ) -> Lattice:
    """
    TODO
    Basis vectors: u1 = (1,0), u2 = (1/2, \sqrt{3}/2)

    Phys. Rev. B 50, 10048 (1994)
    """

    if cluster == "triangle":
        # Triangle spanned by T1 = l*u1 and T2 = l*u2
        l = kwargs['l']
        nodes = [(i, j) for i in range(l + 1) for j in range(l - i + 1)]
        def node_on_cluster(i, j):
            return (i, j) if ((0 <= j <= l) and (0 <= i <= l - j)) else None
    elif cluster == "parallelogram":
        # Parallelogram spanned by T1 = l*u1 and T2 = m*u2
        # This cluster keeps all symmetries of the infinite lattice at l = m.
        l, m = kwargs['l'], kwargs['m']
        periodic = kwargs.get('periodic', True)
        nodes = list(product(range(l + 1), range(m + 1)))
        def node_on_cluster(i, j):
            if periodic:
                return (i % (l + 1), j % (m + 1))
            else:
                return (i, j) if ((0 <= i <= l) and (0 <= j <= m)) else None
    elif cluster == "hexagon":
        # Hexagonal cluster which keeps all symmetries of the infinite lattice.
        # Periodicity vectors T1 = l*(u1 + u2) and T2 = -l*u1 + 2l*u2
        l = kwargs['l']
        periodic = kwargs.get('periodic', True)
        i_max = 3 * l - 2
        nodes = [(i, j) for i in range(i_max + 1) for j in range(i_max + 1 - i)
                 if (i + j >= l - 1 and i <= 2 * l - 1 and j <= 2 * l - 1)]
        def on_hexagon(i, j):
            return (i >= 0 and j >= 0 and
                    i + j >= l - 1 and i + j <= i_max and
                    i <= 2 * l - 1 and j <= 2 * l - 1)

        if periodic:
            T = ((2, 2), (-2, -2), (-2, 4), (2, -4), (4, -2), (-4, 2))
            reduced_nodes = {}
            for node in nodes:
                reduced_nodes[node] = node
                for t in T:
                    shifted_node = tuple(np.add(node, t))
                    if not on_hexagon(*shifted_node):
                        reduced_nodes[shifted_node] = node

            def node_on_cluster(i, j):
                return reduced_nodes.get((i, j), None)

        else:
            def node_on_cluster(i, j):
                return (i, j) if on_hexagon(i, j) else None
    else:
        raise ValueError(f"Unknown cluster shape '{cluster}'")

    n_nodes = len(nodes)

    stencils_d = {}
    stencils_d['NN'] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    stencils_d['NNN'] = [(1, 1), (-1, -1), (2, -1), (-2, 1), (-1, 2), (1, -2)]

    mats = {n: np.zeros((n_nodes, n_nodes), dtype=int)
            for n in stencils_d.keys()}

    for subgraph_name in stencils_d.keys():
        stencils = stencils_d[subgraph_name]
        mat = mats[subgraph_name]
        for n1, node1 in enumerate(nodes):
            for st in stencils:
                node2 = node_on_cluster(*np.add(node1, st))
                if node2 is not None:
                    mat[n1, nodes.index(node2)] += 1

    it = index_translator if (index_translator is not None) else (lambda i: i)
    return Lattice(map(it, nodes), mats)

# TODO: Lieb lattice
# TODO: Shastry–Sutherland lattice
# TODO: BCC lattice
# TODO: FCC lattice
# TODO: Hexagonal lattice
# TODO: Kagome lattice
