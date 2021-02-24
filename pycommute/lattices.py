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
        sum(weights[w] * self.subgraph_matrices[w] for w in weights)


def hypercubic_lattice(shape: Tuple[int, ...],
                       index_translator:
                       Callable[[Tuple[int, ...]], IndicesType] = None,
                       periodic: Union[bool, Tuple[bool, ...]] = True
                       ) -> Lattice:
    """
    TODO
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
