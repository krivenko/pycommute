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
    c, c_dag, a, a_dag
)

from typing import Union, Tuple, Sequence, Callable
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

# TODO: Su-Schrieffer–Heeger model
# TODO: Kitaev chain
# TODO: Spin models
# TODO: Interaction term for bosons
# TODO: Interaction terms for fermions: Hubbard, Hubbard-Kanamori and Slater
# TODO: Interaction term: Hubbard-Holstein
