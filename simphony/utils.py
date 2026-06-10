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

from .config import Config

import numpy as np
import weakref
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm
from scipy.linalg import expm as scipy_expm
from typing import Optional, Union, List, Sequence, Tuple

from .exceptions import SimphonyError

COMPONENTS = ['x', 'y', 'z']


def _values_equal(left, right) -> bool:
    """Return True when two scalar or array-like values are equal."""
    try:
        result = left == right
    except Exception:
        result = None

    if isinstance(result, (bool, np.bool_)):
        return bool(result)

    if result is not None:
        try:
            return bool(np.all(np.asarray(result)))
        except Exception:
            pass

    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception:
        return False


uarray = Union[np.ndarray, jnp.ndarray]


def _unp_from_array(obj):
    if isinstance(obj, (np.ndarray, list)):
        return np
    else:
        return jnp


def _axis_angle_from_local_z_to_direction(direction: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Return the axis-angle rotation that maps local ``z`` onto ``direction``.

    Args:
        direction: Target direction expressed in the lab frame.

    Returns:
        A pair ``(rotation_axis, angle)`` where ``rotation_axis`` is a normalized
        3-component vector and ``angle`` is the rotation angle in radians. The
        pair is suitable for constructing a Rodrigues rotation matrix or a
        Hilbert-space rotation operator.
    """

    target_direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(target_direction)
    if norm == 0:
        raise SimphonyError('direction must be a non-zero vector.')
    target_direction = target_direction / norm

    local_z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(local_z_axis, target_direction), -1.0, 1.0))
    angle = float(np.arccos(dot))

    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    rotation_axis = np.cross(local_z_axis, target_direction)
    axis_norm = np.linalg.norm(rotation_axis)
    if axis_norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float), angle

    return rotation_axis / axis_norm, angle


def _rotation_matrix_from_axis_angle(rotation_axis: Sequence[float],
                                     angle: float) -> np.ndarray:
    """Return the 3x3 Rodrigues rotation matrix for an axis-angle rotation.

    Args:
        rotation_axis: Rotation axis as a 3-component vector. The input is
            normalized internally.
        angle: Rotation angle in radians.

    Returns:
        A 3x3 orthogonal matrix that rotates lab-frame vectors and tensors
        according to the supplied axis-angle pair.
    """

    rotation_axis = np.asarray(rotation_axis, dtype=float)
    if rotation_axis.shape != (3,):
        raise SimphonyError('rotation_axis must be a three-component vector.')

    axis_norm = np.linalg.norm(rotation_axis)
    if axis_norm == 0:
        raise SimphonyError('rotation_axis must be a non-zero vector.')
    rotation_axis = rotation_axis / axis_norm

    if np.isclose(angle, 0.0):
        return np.eye(3, dtype=float)

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    K = np.array([
        [0.0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0.0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0.0],
    ])
    return np.eye(3, dtype=float) + sin_a * K + (1 - cos_a) * (K @ K)


def expm(matrix: uarray,
         *,
         unp_inner=None) -> uarray:
    """Matrix exponential that uses the configured namespace."""
    if unp_inner is None:
        unp_inner = _unp_from_array(matrix)

    if unp_inner is np:
        return scipy_expm(matrix)

    return jax_expm(matrix, max_squarings=Config.get_jax_expm_max_squarings())


def tensorprod(*args,
               unp_inner=None) -> uarray:
    """Return the Kronecker tensor product of arrays using the provided namespace.

    Args:
        *args: Input arrays to compute the Kronecker tensor product.
        unp_inner: Array namespace (e.g., ``numpy`` or ``jax.numpy``).

    Returns:
        Kronecker tensor product of the input arrays.

    """

    if not args:
        raise ValueError('tensorprod requires at least one operand.')

    if unp_inner is None:
        unp_inner = _unp_from_array(args[0])

    result = args[0]
    for operand in args[1:]:
        result = unp_inner.kron(result, operand)
    return result

def _product_with_diagonal_from_left(diagonal: uarray,
                                     matrix: uarray,
                                     *,
                                     unp_inner=None) -> uarray:
    if unp_inner is None:
        unp_inner = _unp_from_array(diagonal)
    return diagonal[:, unp_inner.newaxis] * matrix

def _product_with_diagonal_from_right(matrix: uarray,
                                      diagonal: uarray,
                                      *,
                                      unp_inner=None) -> uarray:
    if unp_inner is None:
        unp_inner = _unp_from_array(diagonal)
    return matrix * diagonal[unp_inner.newaxis, :]


def average_gate_fidelity_from_unitaries(unitary_1: uarray,
                                         unitary_2: uarray,
                                         *,
                                         unp_inner=None) -> Union[float, uarray]:
    """Return the average gate fidelity between two unitary matrices.

    Args:
        unitary_1: The first unitary matrix, representing the ideal or target quantum operation.
        unitary_2: The second unitary matrix, representing the implemented or actual quantum operation. It could be a
            multidimensional array where the last two dimensions correspond to the unitary matrices themselves.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

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

    if unitary_1.ndim != 2 or unitary_1.shape[0] != unitary_1.shape[1]:
        raise SimphonyError('unitary_1 must be a square matrix.')

    if unitary_2.ndim >= 2 and unitary_2.shape[-1] != unitary_2.shape[-2]:
        raise SimphonyError('unitary_2 must be a square matrix in the last two dimensions.')

    if unitary_1.shape[-1] != unitary_2.shape[-1]:
        raise SimphonyError('Matrices must have the same shape in the last two dimensions.')

    if unp_inner is None:
        unp_inner = _unp_from_array(unitary_1)

    d = unitary_1.shape[0]
    overlap = unp_inner.einsum(
        'ij,...ji->...', unitary_1.conjugate().transpose(), unitary_2
    )
    process_fidelity = (overlap.conjugate() * overlap).real / (d ** 2)

    gate_fidelity = (d * process_fidelity + 1) / (d + 1)
    return gate_fidelity


def average_gate_fidelity_from_process_matrices(process_matrix_1: uarray,
                                                process_matrix_2: uarray,
                                                *,
                                                unp_inner=None) -> Union[float, uarray]:
    """Return the average gate fidelity between two process matrices.

    Args:
        process_matrix_1: The first process matrix, representing the ideal or target quantum operation.
        process_matrix_2: The second process matrix, representing the implemented or actual quantum operation. It
            could be a multidimensional array where the last two dimensions correspond to the process matrices
            themselves.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

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

    if process_matrix_1.ndim != 2 or process_matrix_1.shape[0] != process_matrix_1.shape[1]:
        raise SimphonyError('process_matrix_1 must be a square matrix.')

    if process_matrix_2.ndim >= 2 and process_matrix_2.shape[-1] != process_matrix_2.shape[-2]:
        raise SimphonyError('process_matrix_2 must be a square matrix in the last two dimensions.')

    if process_matrix_1.shape[-1] != process_matrix_2.shape[-1]:
        raise SimphonyError('Matrices must have the same shape in the last two dimensions.')

    if unp_inner is None:
        unp_inner = _unp_from_array(process_matrix_1)

    d_process_matrix = process_matrix_1.shape[0]
    if not (d_process_matrix > 0 and (d_process_matrix & (d_process_matrix - 1) == 0) and (d_process_matrix % 3 == 1)):
        raise SimphonyError('Dimension of the process matrices must be a power of 4')

    d = unp_inner.sqrt(d_process_matrix)

    process_fidelity = unp_inner.einsum('ij,...ji->...', process_matrix_1, process_matrix_2).real

    return (d * process_fidelity + 1) / (d + 1)


def is_unitary(matrix: uarray,
               *,
               unp_inner=None) -> Union[bool, uarray]:
    """Check if a given matrix is unitary.

    Args:
        matrix: A square matrix to be checked for unitarity.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

    Raises:
        SimphonyError: If the provided matrix is not 2-dimensional or not square.

    """

    if matrix.ndim != 2:
        raise SimphonyError('Not a 2-dimensional matrix provided.')

    if matrix.shape[0] != matrix.shape[1]:
        raise SimphonyError('Not a square matrix provided.')

    if unp_inner is None:
        unp_inner = _unp_from_array(matrix)

    return unp_inner.allclose(unp_inner.eye(matrix.shape[0]), matrix.conjugate().transpose() @ matrix)


def leakage_from_process_matrix(matrix: uarray,
                                *,
                                unp_inner=None) -> Union[float, uarray]:
    """Return the leakage from the qubit subspace.

    Leakage is defined as the probability that a quantum operation results in the system leaving the qubit subspace.
    For an ideal process matrix, the trace of the process matrix should equal 1, indicating no leakage.

    Args:
        matrix: The process matrix representing the quantum operation.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

    Returns:
        The leakage from the qubit subspace.

    Raises:
        SimphonyError: If the process matrix is not square.

    Note:
        The leakage :math:`L` is calculated as:

        .. math::
            L = 1 - \\text{Tr}(\\chi),

        where :math:`\\chi` is the process matrix.

    """

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SimphonyError("The process matrix must be square.")

    if unp_inner is None:
        unp_inner = _unp_from_array(matrix)

    return 1 - unp_inner.real(unp_inner.trace(matrix))

def leakage_from_time_evolution_operator(matrix: uarray,
                                         *,
                                         unp_inner=None) -> Union[float, uarray]:
    """Compute the leakage from the qubit subspace of a quantum operation.

    Leakage is defined as the probability that a quantum operation causes the system to evolve outside the qubit
    subspace. For a unitary operator, the leakage is zero. For a non-unitary (e.g., projected) operator describing
    dynamics within the qubit subspace, the leakage is expected to be less than 1.

    Args:
        matrix: A square complex matrix representing the quantum operation. It could be a multidimensional array where
            the last two dimensions correspond to the time evolution operator themselves.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

    Returns:
        Leakage from the qubit subspace.

    Raises:
        SimphonyError: If the input matrix is not square in the last two dimensions.

    Note:
        The leakage :math:`L` is calculated as:

        .. math::

            L = 1 - \\frac{1}{d} \\text{Tr}(U^\\dagger U),

        where :math:`d` is the dimension of the matrix :math:`U`, and :math:`U^\\dagger` is the Hermitian conjugate of
        :math:`U`.

    """

    if len(matrix.shape) < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise SimphonyError("The matrix must be square.")

    if unp_inner is None:
        unp_inner = _unp_from_array(matrix)

    d = matrix.shape[-1]
    UdagU = unp_inner.einsum('...ij,...ij->...', matrix.conjugate(), matrix)

    return 1 - (unp_inner.real(UdagU) / d)


def partial_trace(matrix: uarray,
                  sub_dims: List[int],
                  keep: list,
                  *,
                  unp_inner=None) -> uarray:
    """Compute the partial trace of a matrix. The matrix may be a multidimensional array, with the partial trace taken
    over the last two dimensions.

    Parameters:
        matrix: The input matrix to compute the partial trace of.
        sub_dims: The dimensions of the subsystems.
        keep: Indices of the subsystems to keep.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

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
    if unp_inner is None:
        unp_inner = _unp_from_array(matrix)
    if unp_inner.prod(unp_inner.asarray(sub_dims)) != matrix.shape[-1]:
        raise SimphonyError("Subdimensions do not match matrix size.")

    rest_dims = list(matrix.shape[:-2])
    dim = len(sub_dims)
    trace = list(set(range(dim)) - set(keep))

    dim_keep = unp_inner.prod(unp_inner.asarray([sub_dims[i] for i in keep], dtype=int))
    dim_trace = unp_inner.prod(unp_inner.asarray([sub_dims[i] for i in trace], dtype=int))

    traced_matrix = matrix.copy()
    traced_matrix = unp_inner.reshape(traced_matrix, rest_dims + sub_dims + sub_dims)

    permute_axes = (
            list(range(len(rest_dims))) +
            [len(rest_dims) + i for i in keep] +
            [len(rest_dims) + i for i in trace] +
            [len(rest_dims) + dim + i for i in keep] +
            [len(rest_dims) + dim + i for i in trace]
    )
    traced_matrix = unp_inner.transpose(traced_matrix, permute_axes)

    reshaped_dims = rest_dims + [dim_keep, dim_trace, dim_keep, dim_trace]
    traced_matrix = unp_inner.reshape(traced_matrix, reshaped_dims)

    traced_matrix = unp_inner.einsum('...ikjk->...ij', traced_matrix)

    return traced_matrix


def fill_array_by_idx(array: uarray,
                      dimension: int,
                      idxs: List[int],
                      *,
                      unp_inner=None) -> uarray:
    """Extend a square array using specified indices.

    Args:
        array: Input array to be extended.
        dimension: Dimension of the new square array.
        idxs: A list of indices indicating positions in the input array to be filled.
        unp_inner: Array namespace (``numpy`` or ``jax.numpy``) that performs the computation.

    Returns:
        Extended square array.

    """

    if unp_inner is None:
        unp_inner = _unp_from_array(array)

    array_to_insert = unp_inner.asarray(array)

    # Create an extended array with zeros
    extended_array = unp_inner.zeros(shape=(dimension, dimension), dtype=array_to_insert.dtype)
    indexer = unp_inner.ix_(idxs, idxs)

    if hasattr(extended_array, 'at'):
        extended_array = extended_array.at[indexer].set(array_to_insert)
    else:
        extended_array[indexer] = array_to_insert

    return extended_array


def _callback_identity(callback):
    """Return a stable identity for a callback."""
    owner = getattr(callback, '__self__', None)
    func = getattr(callback, '__func__', None)
    if owner is not None and func is not None:
        return (id(owner), id(func))
    return ('callable', id(callback))


class _AttachedCallbackMixin:
    """Provide attachable weak callback lifecycle hooks."""
    def _init_attached_callbacks(self) -> None:
        existing = getattr(self, '_change_callback', None)
        self._attached_callbacks = {}
        if existing is not None:
            self._attach_callback(existing)

    def __getstate__(self):
        """Drop attached callback refs before deepcopy/pickling."""
        state = self.__dict__.copy()
        state['_attached_callbacks'] = {}
        return state

    def __setstate__(self, state):
        """Restore instance state with an empty callback registry."""
        self.__dict__.update(state)
        self._attached_callbacks = {}

    def _make_callback_ref(self, callback):
        owner = getattr(callback, '__self__', None)
        func = getattr(callback, '__func__', None)
        if owner is not None and func is not None:
            return weakref.WeakMethod(callback)
        try:
            return weakref.ref(callback)
        except TypeError:
            return lambda: callback

    def _attach_callback(self, callback) -> None:
        if callback is None:
            return
        if not hasattr(self, '_attached_callbacks'):
            self._attached_callbacks = {}
        self._attached_callbacks[_callback_identity(callback)] = self._make_callback_ref(callback)

    def _detach_callback(self, callback) -> None:
        if not hasattr(self, '_attached_callbacks'):
            return
        self._attached_callbacks.pop(_callback_identity(callback), None)

    def _notify_callbacks(self):
        callbacks = getattr(self, '_attached_callbacks', None)
        if callbacks:
            dead_keys = []
            for key, callback_ref in callbacks.items():
                callback = callback_ref()
                if callback is None:
                    dead_keys.append(key)
                    continue
                callback()
            for key in dead_keys:
                callbacks.pop(key, None)

        legacy_callback = getattr(self, '_change_callback', None)
        if legacy_callback is not None:
            legacy_key = _callback_identity(legacy_callback)
            if not callbacks or legacy_key not in callbacks:
                legacy_callback()


class Components1D(_AttachedCallbackMixin):
    """1D components (vector) class.

    Initialize the components by a vector. Attributes :attr:`x`, :attr:`y` and :attr:`z` can be used to set the components.

    Args:
        vec: The initial vector. Defaults to ``None``, resulted in a zero vector.
        normalize: Whether to normalize the vector on creation.

    """

    def __init__(self,
                 vec: Optional[List[float]] = None,
                 *,
                 normalize: bool = False):
        self._init_attached_callbacks()
        if vec is None:
            self._vec = [0 for _ in COMPONENTS]
        else:
            self.vec = vec
            if normalize:
                self.normalize()

        for idx, component in enumerate(COMPONENTS):
            def fget(self, idx=idx):
                return self._vec[idx]

            def fset(self, value, idx=idx):
                if _values_equal(self._vec[idx], value):
                    return
                self._vec[idx] = value
                self._notify_callbacks()

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
        return tuple(self._vec)

    @vec.setter
    def vec(self, vec):
        values = list(vec)
        if len(values) != len(COMPONENTS):
            raise SimphonyError(f'Components1D requires exactly {len(COMPONENTS)} components.')
        if hasattr(self, '_vec') and all(_values_equal(current, new) for current, new in zip(self._vec, values)):
            return
        self._vec = values
        self._notify_callbacks()

    def normalize(self):
        """Normalize the vector to unit length."""
        vec = np.asarray(self._vec, dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector in Components1D.")
        vec /= norm
        self.vec = vec.tolist()


class Components2D(_AttachedCallbackMixin):
    """2D components (tensor) class.

    Initialize the components by a tensor. Attributes :attr:`xx`, :attr:`xy`, ..., :attr:`zz` can be used to set the
    components.

    Args:
        tensor: The initial tensor. Defaults to ``None``, results in a zero tensor.

    """

    def __init__(self,
                 tensor: Optional[List[List[float]]] = None):
        self._init_attached_callbacks()

        if tensor is None:
            self._tensor = [[0 for _ in COMPONENTS] for _ in COMPONENTS]
        else:
            self.tensor = tensor

        for idx1, component1 in enumerate(COMPONENTS):
            for idx2, component2 in enumerate(COMPONENTS):
                def fget(self, idx1=idx1, idx2=idx2):
                    return self._tensor[idx1][idx2]

                def fset(self, value, idx1=idx1, idx2=idx2):
                    if _values_equal(self._tensor[idx1][idx2], value):
                        return
                    self._tensor[idx1][idx2] = value
                    self._notify_callbacks()

                setattr(Components2D, component1 + component2, property(fget, fset))

    def __repr__(self):
        tensor_str = np.array2string(np.array(self.tensor), separator=', ')
        return f'{type(self).__name__}(tensor={tensor_str})'

    @property
    def tensor(self):
        """Tensor representation of the components."""
        return tuple(tuple(row) for row in self._tensor)

    @tensor.setter
    def tensor(self, tensor):
        values = [list(row) for row in tensor]
        expected_len = len(COMPONENTS)
        if len(values) != expected_len or any(len(row) != expected_len for row in values):
            raise SimphonyError(f'Components2D requires a {expected_len}x{expected_len} tensor.')
        if hasattr(self, '_tensor') and all(_values_equal(current, new) for row_current, row_new in zip(self._tensor, values) for current, new in zip(row_current, row_new)):
            return
        self._tensor = values
        self._notify_callbacks()
