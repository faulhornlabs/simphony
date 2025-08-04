# This code is part of Simphony.
#
# Copyright 2025 Qutility @ Faulhorn Labs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions and classes."""

from .config import unp

import numpy as np
import jax.numpy as jnp
from typing import Optional, Union, List

from .exceptions import SimphonyError

COMPONENTS = ['x', 'y', 'z']

uarray = Union[np.ndarray, jnp.ndarray]


def tensorprod(*args) -> uarray:
    """Return the Kronecker tensor product of arrays. It extends the functionality of the ``(jax.)numpy.kron()``
    function.

    Args:
        *args: Input arrays to compute the Kronecker tensor product.

    Returns:
        Kronecker tensor product of the input arrays

    """

    args = [unp.array(arg) for arg in args]

    if len(args) == 1:
        return args[0]
    elif len(args) == 2:
        a = args[0]
        b = args[1]
        return unp.kron(a, b)
    else:
        a = args[0]
        b = args[1]
        rest = args[2:]
        return tensorprod(tensorprod(a, b), *rest)

def _product_with_diagonal_from_left(diagonal, matrix) -> uarray:
    return diagonal[:, unp.newaxis] * matrix

def _product_with_diagonal_from_right(matrix, diagonal) -> uarray:
    return matrix * diagonal[unp.newaxis, :]


def average_gate_fidelity_from_unitaries(unitary_1: uarray,
                                         unitary_2: uarray) -> Union[float, uarray]:
    """Return the average gate fidelity between two unitary matrices.

    Args:
        unitary_1: The first unitary matrix, representing the ideal or target quantum operation.
        unitary_2: The second unitary matrix, representing the implemented or actual quantum operation. It could be a
            multidimensional array where the last two dimensions correspond to the unitary matrices themselves.

    Returns:
        The average gate fidelity between the two unitaries. If ``unitary_2`` is multidimensional, it is an array.

    Raises:
        SimphonyError: If the provided matrices are not of the same shape or are not square matrices.

    Note:
        The formula used for calculating the average gate fidelity :math:`F` is:

        .. math::
            F = \\frac{| \\text{Tr}(U_1^{\\dagger} U_2) |^2 / d + 1}{d + 1},

        where :math:`d` is the dimension of the matrices.

    """

    if unitary_1.ndim != 2 and unitary_1.shape[0] != unitary_1.shape[1]:
        raise SimphonyError('unitary_1 must be a square matrix.')

    if unitary_2.ndim >= 2 and unitary_2.shape[-1] != unitary_2.shape[-2]:
        raise SimphonyError('unitary_2 must be a square matrix in the last two dimensions.')

    if unitary_1.shape[-1] != unitary_2.shape[-1]:
        raise SimphonyError('Matrices must have the same shape in the last two dimensions.')

    d = unitary_1.shape[0]
    process_fidelity = unp.abs(unp.einsum('ij,...ji->...', unitary_1.conjugate().transpose(), unitary_2)) ** 2 / d ** 2

    return (d * process_fidelity + 1) / (d + 1)


def average_gate_fidelity_from_process_matrices(process_matrix_1: uarray,
                                                process_matrix_2: uarray) -> Union[float, uarray]:
    """Return the average gate fidelity between two process matrices.

    Args:
        process_matrix_1: The first process matrix, representing the ideal or target quantum operation.
        process_matrix_2: The second process matrix, representing the implemented or actual quantum operation. It
            could be a multidimensional array where the last two dimensions correspond to the process matrices
            themselves.

    Returns:
        The average gate fidelity between the two process matrices. If ``process_matrix_2`` is multidimensional, it is
        an array.

    Raises:
        SimphonyError: If the provided matrices are not of the same shape, are not square, or do not have
            dimensions that are a power of 4.

    Note:
        The formula used for calculating the average gate fidelity :math:`F` is:

        .. math::
            F = \\frac{d \\cdot \\text{Tr}(\\chi_1 \\chi_2) + 1}{d + 1},

        where :math:`d` is the dimension of the matrices.


    """

    if process_matrix_1.ndim != 2 and process_matrix_1.shape[0] != process_matrix_1.shape[1]:
        raise SimphonyError('process_matrix_1 must be a square matrix.')

    if process_matrix_2.ndim >= 2 and process_matrix_2.shape[-1] != process_matrix_2.shape[-2]:
        raise SimphonyError('process_matrix_2 must be a square matrix in the last two dimensions.')

    if process_matrix_1.shape[-1] != process_matrix_2.shape[-1]:
        raise SimphonyError('Matrices must have the same shape in the last two dimensions.')

    d_process_matrix = process_matrix_1.shape[0]
    if not (d_process_matrix > 0 and (d_process_matrix & (d_process_matrix - 1) == 0) and (d_process_matrix % 3 == 1)):
        raise SimphonyError('Dimension of the process matrices must be a power of 4')

    d = np.sqrt(d_process_matrix)

    process_fidelity = unp.einsum('ij,...ji->...', process_matrix_1, process_matrix_2).real

    return (d * process_fidelity + 1) / (d + 1)


def is_unitary(matrix: uarray) -> Union[bool, uarray]:
    """Check if a given matrix is unitary.

    Args:
        matrix: A square matrix to be checked for unitarity.

    Raises:
        SimphonyError: If the provided matrix is not 2-dimensional or not square.

    """

    if matrix.ndim != 2:
        raise SimphonyError('Not a 2-dimensional matrix provided.')

    if matrix.shape[0] != matrix.shape[1]:
        raise SimphonyError('Not a square matrix provided.')

    return unp.allclose(unp.eye(matrix.shape[0]), matrix.conjugate().transpose() @ matrix)


def leakage_from_process_matrix(matrix: uarray) -> Union[float, uarray]:
    """Return the leakage from the qubit subspace.

    Leakage is defined as the probability that a quantum operation results in the system leaving the qubit subspace.
    For an ideal process matrix, the trace of the process matrix should equal 1, indicating no leakage.

    Args:
        matrix: The process matrix representing the quantum operation.

    Returns:
        The leakage from the qubit subspace.

    Raises:
        SimphonyError: If the process matrix is not square.

    Notes:
        The leakage :math:`L` is calculated as:

        .. math::
            L = 1 - \\text{Tr}(\\chi),

        where :math:`\\chi` is the process matrix.

    """

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SimphonyError("The process matrix must be square.")

    return 1 - unp.real(unp.trace(matrix))

def leakage_from_time_evolution_operator(matrix: uarray) -> Union[float, uarray]:
    """Compute the leakage from the qubit subspace of a quantum operation.

    Leakage is defined as the probability that a quantum operation causes the system to evolve outside the qubit
    subspace. For a unitary operator, the leakage is zero. For a non-unitary (e.g., projected) operator describing
    dynamics within the qubit subspace, the leakage is expected to be less than 1.

    Args:
        matrix: A square complex matrix representing the quantum operation. It could be a multidimensional array where
            the last two dimensions correspond to the time evolution operator themselves.

    Returns:
        Leakage from the qubit subspace.

    Raises:
        SimphonyError: If the input matrix is not square in the last two dimensions.

    Notes:
        The leakage :math:`L` is calculated as:

        .. math::

            L = 1 - \\frac{1}{d} \\text{Tr}(U^\\dagger U),

        where :math:`d` is the dimension of the matrix :math:`U`, and :math:`U^\\dagger` is the Hermitian conjugate of
        :math:`U`.

    """

    if len(matrix.shape) < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise SimphonyError("The matrix must be square.")

    d = matrix.shape[-1]
    UdagU = unp.einsum('...ij,...ij->...', matrix.conjugate(), matrix)

    return 1 - (unp.real(UdagU) / d)


def partial_trace(matrix: uarray,
                  sub_dims: List[int],
                  keep: list) -> uarray:
    """Compute the partial trace of a matrix. The matrix may be a multidimensional array, with the partial trace taken
    over the last two dimensions.

    Parameters:
        matrix: The input matrix to compute the partial trace of.
        sub_dims: The dimensions of the subsystems.
        keep: Indices of the subsystems to keep.

    Returns:
        The resulting matrix after tracing out the specified subsystems.

    Raises:
        SimphonyError: If the matrix does not have at least two dimensions, is not square in its last two dimensions, or
            if the subdimensions do not match the matrix size.
    """

    if len(matrix.shape) < 2:
        raise SimphonyError("Input matrix must have at least 2 dimensions.")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise SimphonyError("Input matrix must be square.")
    if unp.prod(unp.array(sub_dims)) != matrix.shape[-1]:
        raise SimphonyError("Subdimensions do not match matrix size.")

    rest_dims = list(matrix.shape[:-2])
    dim = len(sub_dims)
    trace = list(set(range(dim)) - set(keep))

    dim_keep = unp.prod(unp.array([sub_dims[i] for i in keep], dtype=int))
    dim_trace = unp.prod(unp.array([sub_dims[i] for i in trace], dtype=int))

    traced_matrix = matrix.copy()
    traced_matrix = traced_matrix.reshape(rest_dims + sub_dims + sub_dims)

    permute_axes = (
            list(range(len(rest_dims))) +
            [len(rest_dims) + i for i in keep] +
            [len(rest_dims) + i for i in trace] +
            [len(rest_dims) + dim + i for i in keep] +
            [len(rest_dims) + dim + i for i in trace]
    )
    traced_matrix = unp.transpose(traced_matrix, permute_axes)

    reshaped_dims = rest_dims + [dim_keep, dim_trace, dim_keep, dim_trace]
    traced_matrix = traced_matrix.reshape(reshaped_dims)

    traced_matrix = unp.einsum('...ikjk->...ij', traced_matrix)

    return traced_matrix


def fill_array_by_idx(array: np.ndarray,
                      dimension: int,
                      idxs: List[int]
                      ) -> np.ndarray:
    """Extend a square array using specified indices.

    Args:
        array: Input array to be extended.
        dimension: Dimension of the new square array.
        idxs: A list of indices indicating positions in the input array to be filled.

    Returns:
        Extended square array.

    """

    # Create an extended array with zeros
    extended_array = np.zeros(shape=(dimension, dimension), dtype=array.dtype)

    # Copy values from the input array to the extended array at specified indices
    extended_array[np.ix_(idxs, idxs)] = array

    return extended_array


class Components1D:
    """1D components (vector) class.

    Initialize the components by a vector. Attributes :attr:`x`, :attr:`y` and :attr:`z` can be used to set the components.

    Args:
        vec: The initial vector. Defaults to ``None``, resulted in a zero vector.

    """

    def __init__(self,
                 vec: Optional[List[float]] = None):
        if vec is None:
            self._vec = [0, 0, 0]
        else:
            self._vec = vec

        for idx, component in enumerate(COMPONENTS):
            def fget(self, idx=idx):
                return self._vec[idx]

            def fset(self, value, idx=idx):
                self._vec[idx] = value

            docstring = f"The {component}-component of the vector."
            setattr(type(self), component, property(fget, fset, doc=docstring))

        # extra attribute to store the identity operator of a spin vector-operator
        self.i = None

    def __repr__(self):
        vec_str = np.array2string(np.array(self.vec), separator=', ')
        return f'{type(self).__name__}(vec={vec_str})'

    @property
    def vec(self):
        """Vector representation of the components."""
        return self._vec

    @vec.setter
    def vec(self, vec):
        self._vec = vec


class Components2D:
    """2D components (tensor) class.

    Initialize the components by a tensor. Attributes :attr:`xx`, :attr:`xy`, ..., :attr:`zz` can be used to set the
    components.

    Args:
        tensor: The initial tensor. Defaults to ``None``, results in a zero tensor.

    """

    def __init__(self,
                 tensor: Optional[List[List[float]]] = None):

        if tensor is None:
            self._tensor = [[0 for _ in COMPONENTS] for _ in COMPONENTS]
        else:
            self._tensor = tensor

        for idx1, component1 in enumerate(COMPONENTS):
            for idx2, component2 in enumerate(COMPONENTS):
                def fget(self, idx1=idx1, idx2=idx2):
                    return self._tensor[idx1][idx2]

                def fset(self, value, idx1=idx1, idx2=idx2):
                    self._tensor[idx1][idx2] = value

                setattr(Components2D, component1 + component2, property(fget, fset))

    def __repr__(self):
        tensor_str = np.array2string(np.array(self.tensor), separator=', ')
        return f'{type(self).__name__}(tensor={tensor_str})'

    @property
    def tensor(self):
        """Tensor representation of the components."""
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor