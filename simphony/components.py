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

"""Components like spins, interaction, static and driving field, furthermore low API classes."""

from __future__ import annotations

from .config import unp_static, unp_dynamic
from .utils import fill_array_by_idx, Components1D, Components2D, _axis_angle_from_local_z_to_direction, _rotation_matrix_from_axis_angle
from .exceptions import SimphonyError

from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass

import numpy as np
import weakref
import jax.numpy as jnp

import matplotlib.pyplot as plt

from jax.lax import stop_gradient

uarray = Union[np.ndarray, jnp.ndarray]


class ModelComponent:
    """Base class for model-attached mutable components."""

    _component_change_kind: Optional[str] = None

    def __init__(self):
        self._attached_models = weakref.WeakSet()

    def _attach_model(self, model) -> None:
        self._attached_models.add(model)

    def _detach_model(self, model) -> None:
        self._attached_models.discard(model)

    def _notify_models(self, change_kind: Optional[str] = None) -> None:
        if change_kind is None:
            change_kind = self._component_change_kind
        if change_kind is None:
            return
        for model in tuple(self._attached_models):
            model._mark_component_dirty(self, change_kind)


def _rewire_component_callbacks(component) -> None:
    """Rebuild helper callback wiring after deepcopy."""
    if isinstance(component, BaseSpin):
        component.anisotropy_axis._attach_callback(component._notify_models)
        component.local_quasistatic_noise._attach_callback(component._notify_models)
        if isinstance(component.quantization_axis, Components1D):
            component.quantization_axis._attach_callback(component._notify_models)

    if isinstance(component, Interaction):
        component._attach_callback(component._notify_models)
    elif isinstance(component, StaticField):
        component._attach_callback(component._notify_models)
    elif isinstance(component, LinearDrivingField):
        component._attach_callback(component._notify_models)
    elif isinstance(component, CircularDrivingField):
        component.plane_normal._attach_callback(component._on_geometry_changed)
        component.phase_reference._attach_callback(component._on_geometry_changed)
        component.phase_quadrature._attach_callback(component._on_geometry_changed)


def _quantum_nums_from_dimension(dimension):
    """Return quantum numbers from dimension."""
    return [(i - dimension / 2 + 0.5) if dimension % 2 == 0 else (i - (dimension - 1) / 2) for i in range(dimension)]


def _check_quantum_num(dimension,
                       quantum_num):
    """Check quantum numbers based on dimension."""
    quantum_nums = _quantum_nums_from_dimension(dimension)
    if not quantum_num in quantum_nums:
        raise SimphonyError(f'The quantum number must be in {quantum_nums}')

    return quantum_num


def _check_qubit_quantum_nums(dimension,
                              qubit_quantum_nums):
    """Check qubit quantum numbers based on dimension."""
    if len(set(qubit_quantum_nums)) != 2:
        raise SimphonyError('Two different quantum numbers are needed')

    return tuple([_check_quantum_num(dimension, qubit_quantum_num) for qubit_quantum_num in qubit_quantum_nums])


class BaseSpin(ModelComponent):
    """Base functionality shared by electron and nuclear spins.

    Note:
        In most user code, prefer :class:`ElectronSpin` or :class:`NuclearSpin` instead of
        instantiating :class:`BaseSpin` directly.

    Args:
        dimension: Dimension, must be 2 for spin-1/2, 3 for spin-1, etc.
        name: Name of the spin.
        qubit_subspace: The quantum numbers that define the qubit subspace.
        gyromagnetic_ratio: Gyromagnetic ratio (in :math:`\\text{MHz}/\\text{T}`).
        axial_splitting: Axial anisotropy parameter (in :math:`\\text{MHz}`).
        anisotropy_axis: Local anisotropy axis of the spin, given in the global coordinate system.
        local_quasistatic_noise: Local-quasistatic noise components of the spin (in :math:`\\text{MHz}`).
        quantization_axis: Quantization axis of the spin, given by the keyword ``'global_z'``,
            ``'local_static_field'``, or ``'anisotropy_axis'``, or by an explicit vector in the
            global coordinate system. It determines which direction is treated as the spin's
            local z-axis when building operators and states for that spin.

    Raises:
        SimphonyError: If an invalid dimension or qubit subspace is provided.

    """

    ZEEMAN_SIGN: ClassVar[float] = 1.0
    """Sign of the Zeeman term."""

    _component_change_kind = 'spin_changed'

    def __init__(self,
                 dimension: int,
                 name: str,
                 qubit_subspace: Tuple[float, float],
                 gyromagnetic_ratio: float,
                 *,
                 axial_splitting: float = 0.,
                 transverse_splitting: float = 0.,
                 anisotropy_axis: Optional[List[float]] = None,
                 local_quasistatic_noise: Optional[List[float]] = None,
                 quantization_axis: Union[str, Sequence[float]] = 'global_z'):

        if not isinstance(dimension, int) or dimension < 2:
            raise SimphonyError("dimension must be an integer greater than 1.")

        ModelComponent.__init__(self)

        self.dimension: int = dimension
        """Dimension, it is 2 for spin-1/2, 3 for spin-1, etc."""

        self.name: str = name
        """Name of the spin."""

        self.gyromagnetic_ratio: float = float(gyromagnetic_ratio)
        """Gyromagnetic ratio (in :math:`\\text{MHz}/\\text{T}`)."""

        self._axial_splitting: float = axial_splitting

        self._transverse_splitting = transverse_splitting

        if anisotropy_axis is None:
            anisotropy_axis = Components1D([0., 0., 1.], normalize=True)

        self.anisotropy_axis = anisotropy_axis

        self.quantum_nums: Tuple[float]
        """Quantum numbers corresponding to spin operator."""

        self.rotating_frame_frequency = 0.
        """Rotating frame frequency (in :math:`\\text{MHz}`)."""

        self.virtual_phase = 0.
        """Virtual z-phase of the spin.

        This phase is applied in software rather than by a physical pulse and
        changes the effective phase reference of subsequent pulses acting on the
        spin.
        """

        self.operator: Components1D = Components1D()
        """Spin operator.
        
        Tip:
            The components (``x``, ``y`` or ``z``) of the spin operator are available, for example, by:
    
            .. code-block:: python
    
                >>> spin.operator.x
                array([[0.     +0.j, 0.70711+0.j, 0.     +0.j],
                       [0.70711+0.j, 0.     +0.j, 0.70711+0.j],
                       [0.     +0.j, 0.70711+0.j, 0.     +0.j]])
                
            The identity operator is also stored:
            
            .. code-block:: python
    
                >>> spin.operator.i
                array([[1.+0.j, 0.+0.j, 0.+0.j],
                       [0.+0.j, 1.+0.j, 0.+0.j],
                       [0.+0.j, 0.+0.j, 1.+0.j]])
        
        """

        self.operator_qubit_subspace: Components1D = Components1D()
        """Pauli operators defined on the qubit subspace. 
        
        Tip:
            The components (``x``, ``y`` or ``z``) are available, for example, by:
    
            .. code-block:: python
    
                >>> spin.operator_qubit_subspace.z
                array([[ 0.+0.j,  0.+0.j,  0.+0.j],
                       [ 0.+0.j,  1.+0.j,  0.+0.j],
                       [ 0.+0.j,  0.+0.j, -1.+0.j]])
                
            The identity operator is also stored:
            
            .. code-block:: python
    
                >>> spin.operator_qubit_subspace.i
                array([[0.+0.j, 0.+0.j, 0.+0.j],
                       [0.+0.j, 1.+0.j, 0.+0.j],
                       [0.+0.j, 0.+0.j, 1.+0.j]])
        
        """

        s = (dimension - 1) / 2
        self.quantum_nums = tuple(s - i for i in range(dimension))

        sx = unp_static.zeros((dimension, dimension), dtype=complex)
        sy = unp_static.zeros((dimension, dimension), dtype=complex)
        sz = unp_static.diag(unp_static.asarray(self.quantum_nums)).astype(complex)
        si = unp_static.eye(dimension, dtype=complex)

        for i in range(dimension - 1):
            factor = unp_static.sqrt((s + 1) * (2 * i + 2) - (i + 1) * (i + 2))
            sx[i, i + 1] = sx[i + 1, i] = 0.5 * factor
            sy[i, i + 1] = -0.5j * factor
            sy[i + 1, i] = 0.5j * factor

        self.operator.i = si
        self.operator.x = sx
        self.operator.y = sy
        self.operator.z = sz

        self.qubit_subspace = qubit_subspace

        self.local_quasistatic_noise = local_quasistatic_noise

        self.quantization_axis = quantization_axis

    def __repr__(self):
        return self._repr_with_extras()

    def _repr_base_parts(self) -> list[str]:
        parts = [
            f'dimension={self.dimension}',
            f"name='{self.name}'",
            f'qubit_subspace={self.qubit_subspace}',
        ]

        anisotropy_axis = tuple(self.anisotropy_axis.vec)
        if anisotropy_axis != (0, 0, 1):
            parts.append(f'anisotropy_axis={anisotropy_axis}')

        local_quasistatic_noise = tuple(self.local_quasistatic_noise.vec)
        if local_quasistatic_noise != (0, 0, 0):
            parts.append(f'local_quasistatic_noise={local_quasistatic_noise}')

        if isinstance(self.quantization_axis, str):
            if self.quantization_axis != 'global_z':
                parts.append(f"quantization_axis='{self.quantization_axis}'")
        else:
            parts.append(f'quantization_axis={tuple(self.quantization_axis.vec)}')

        return parts

    def _repr_with_extras(self, *extras: str) -> str:
        parts = self._repr_base_parts()
        parts.extend(extra for extra in extras if extra is not None)
        return f"{type(self).__name__}({', '.join(parts)})"


    def __setstate__(self, state):
        self.__dict__.update(state)
        _rewire_component_callbacks(self)

    @property
    def qubit_subspace(self) -> tuple:
        """Qubit subspace that is defined by a two-length subset of the :attr:`quantum_nums`. The first (second) element
        corresponds to the qubit state :math:`\\ket{0}` (:math:`\\ket{1}`).
        """
        return self._qubit_subspace

    @qubit_subspace.setter
    def qubit_subspace(self, qubit_subspace):

        self._qubit_subspace = _check_qubit_quantum_nums(self.dimension, qubit_subspace)

        idxs = [self.quantum_nums.index(s) for s in qubit_subspace]

        self.operator_qubit_subspace.i = fill_array_by_idx(
            array=unp_static.array([[1, 0], [0, 1]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.x = fill_array_by_idx(
            array=unp_static.array([[0, 1], [1, 0]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.y = fill_array_by_idx(
            array=unp_static.array([[0, -1j], [1j, 0]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.z = fill_array_by_idx(
            array=unp_static.array([[1, 0], [0, -1]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self._notify_models()

    @property
    def anisotropy_axis(self) -> Components1D:
        """Local anisotropy axis of the spin, given in the global coordinate system."""
        return self._anisotropy_axis

    @anisotropy_axis.setter
    def anisotropy_axis(self, value: Union[Components1D, Sequence[float]]) -> None:
        old_axis = getattr(self, '_anisotropy_axis', None)
        if isinstance(old_axis, Components1D):
            old_axis._detach_callback(self._notify_models)

        axis = value if isinstance(value, Components1D) else Components1D(value, normalize=True)
        axis.normalize()
        axis._attach_callback(self._notify_models)
        self._anisotropy_axis = axis
        self._notify_models()

    @property
    def local_quasistatic_noise(self) -> Components1D:
        """Local quasistatic noise components."""
        return self._local_quasistatic_noise

    @local_quasistatic_noise.setter
    def local_quasistatic_noise(self, value: Optional[Union[Components1D, Sequence[float]]]) -> None:
        old_noise = getattr(self, '_local_quasistatic_noise', None)
        if isinstance(old_noise, Components1D):
            old_noise._detach_callback(self._notify_models)

        noise = value if isinstance(value, Components1D) else Components1D(value)
        noise._attach_callback(self._notify_models)
        self._local_quasistatic_noise = noise
        self._notify_models()

    @property
    def anisotropy_tensor(self) -> uarray:
        """Anisotropy tensor in the global coordinate system (in :math:`\\text{MHz}`)."""
        D, E = self._axial_splitting, self._transverse_splitting
        base = unp_static.diag([-D / 3 + E, -D / 3 - E, 2 * D / 3])

        rotation_axis, angle = _axis_angle_from_local_z_to_direction(self.anisotropy_axis.vec)
        if np.isclose(angle, 0.0):
            return base

        R = unp_static.asarray(_rotation_matrix_from_axis_angle(rotation_axis, angle))
        return R @ base @ R.T

    @property
    def quantization_axis(self) -> Union[str, Components1D]:
        """Quantization axis of the spin, given by the keyword ``'global_z'``, ``'local_static_field'``,
        or ``'anisotropy_axis'``, or by an explicit vector in the global coordinate system. It
        determines which direction is treated as the spin's local z-axis when building operators
        and states for that spin."""
        return self._quantization_axis

    @quantization_axis.setter
    def quantization_axis(self, value: Union[str, Sequence[float]]) -> None:
        if isinstance(value, str):
            allowed = ('global_z', 'local_static_field', 'anisotropy_axis')
            if value not in allowed:
                raise SimphonyError(f"quantization_axis string must be one of {allowed}.")
            old_axis = getattr(self, '_quantization_axis', None)
            if isinstance(old_axis, Components1D):
                old_axis._detach_callback(self._notify_models)
            self._quantization_axis = value
            self._notify_models()
            return

        old_axis = getattr(self, '_quantization_axis', None)
        if isinstance(old_axis, Components1D):
            old_axis._detach_callback(self._notify_models)

        axis = Components1D(value, normalize=True)
        axis._attach_callback(self._notify_models)
        self._quantization_axis = axis
        self._notify_models()



class ElectronSpin(BaseSpin):
    """Spin representation tailored for electron spins.

    Args:
        dimension: Dimension, must be 2 for spin-1/2, 3 for spin-1, etc.
        name: Name of the spin.
        qubit_subspace: The quantum numbers that define the qubit subspace.
        gyromagnetic_ratio: Gyromagnetic ratio (in :math:`\\text{MHz}/\\text{T}`).
        zero_field_splitting: Zero-field splitting parameter (in :math:`\\text{MHz}`).
        transverse_splitting: Transverse splitting parameter (in :math:`\\text{MHz}`).
        anisotropy_axis: Local anisotropy axis of the spin, given in the global coordinate system.
        local_quasistatic_noise: Local-quasistatic noise components of the spin (in :math:`\\text{MHz}`).
        quantization_axis: Quantization axis of the spin, given by the keyword ``'global_z'``,
            ``'local_static_field'``, or ``'anisotropy_axis'``, or by an explicit vector in the
            global coordinate system. It determines which direction is treated as the spin's
            local z-axis when building operators and states for that spin.

    Raises:
        SimphonyError: If an invalid dimension or qubit subspace is provided.

    Note:
        Hamiltonian of the electron reads

        .. math::
            H_e = \gamma_e \\mathbf{B} \cdot \\mathbf{S} + D \\left( S_{z'}^2-\\frac{S(S+1)}{3} \\right)
            + E (S_{x'}^2 - S_{y'}^2),

        where :math:`\gamma_e` is the ``gyromagnetic_ratio``, :math:`D` is ``zero_field_splitting``, :math:`E` is the
        ``transverse_splitting``. The zero-field terms are written in the principal axis frame :math:`(x',y',z'),` that
        is defined by the ``anisotropy_axis``.

    """

    ZEEMAN_SIGN: ClassVar[float] = 1.0

    def __init__(self,
                 dimension: int,
                 name: str,
                 qubit_subspace: Tuple[float, float],
                 gyromagnetic_ratio: float,
                 zero_field_splitting: float = 0.,
                 transverse_splitting: float = 0.,
                 anisotropy_axis: Optional[List[float]] = None,
                 local_quasistatic_noise: Optional[List[float]] = None,
                 quantization_axis: Union[str, Sequence[float]] = 'global_z'):
        super().__init__(
            dimension=dimension,
            name=name,
            qubit_subspace=qubit_subspace,
            gyromagnetic_ratio=gyromagnetic_ratio,
            axial_splitting=zero_field_splitting,
            transverse_splitting=transverse_splitting,
            anisotropy_axis=anisotropy_axis,
            local_quasistatic_noise=local_quasistatic_noise,
            quantization_axis=quantization_axis
        )

    @property
    def zero_field_splitting(self) -> float:
        """Zero-field splitting parameter (in :math:`\\text{MHz}`)."""
        return self._axial_splitting

    @zero_field_splitting.setter
    def zero_field_splitting(self, value: float) -> None:
        self._axial_splitting = value
        self._notify_models()

    @property
    def transverse_splitting(self) -> float:
        """Transverse splitting parameter (in :math:`\\text{MHz}`)."""
        return self._transverse_splitting

    @transverse_splitting.setter
    def transverse_splitting(self, value: float) -> None:
        self._transverse_splitting = value
        self._notify_models()

    def __repr__(self):
        return self._repr_with_extras(
            f'zero_field_splitting={self.zero_field_splitting}' if self.zero_field_splitting != 0 else None,
            f'transverse_splitting={self.transverse_splitting}' if self.transverse_splitting != 0 else None,
        )


class NuclearSpin(BaseSpin):
    """Spin representation tailored for nuclear spins.

    Args:
        dimension: Dimension, must be 2 for spin-1/2, 3 for spin-1, etc.
        name: Name of the spin.
        qubit_subspace: The quantum numbers that define the qubit subspace.
        gyromagnetic_ratio: Gyromagnetic ratio (in :math:`\\text{MHz}/\\text{T}`).
        quadrupole_splitting: Quadrupole splitting parameter (in :math:`\\text{MHz}`).
        anisotropy_axis: Local anisotropy axis of the spin, given in the global coordinate system.
        local_quasistatic_noise: Local-quasistatic noise components of the spin (in :math:`\\text{MHz}`).
        quantization_axis: Quantization axis of the spin, given by the keyword ``'global_z'``,
            ``'local_static_field'``, or ``'anisotropy_axis'``, or by an explicit vector in the
            global coordinate system. It determines which direction is treated as the spin's
            local z-axis when building operators and states for that spin.

    Raises:
        SimphonyError: If an invalid dimension or qubit subspace is provided.

    Note:
        Hamiltonian of the nuclear spin reads

        .. math::
            H_n = -\gamma_n \\mathbf{B} \cdot \\mathbf{I} + Q \\left( I_{z'}^2-\\frac{I(I+1)}{3} \\right),

        where :math:`\gamma_n` is the ``gyromagnetic_ratio``, :math:`Q` is ``quadrupole_splitting``. The quadrupole term
        is written in the principal axis frame :math:`(x',y',z'),` that is defined by the ``anisotropy_axis``.

    """

    ZEEMAN_SIGN: ClassVar[float] = -1.0

    def __init__(self,
                 dimension: int,
                 name: str,
                 qubit_subspace: Tuple[float, float],
                 gyromagnetic_ratio: float,
                 quadrupole_splitting: float = 0.,
                 anisotropy_axis: Optional[List[float]] = None,
                 local_quasistatic_noise: Optional[List[float]] = None,
                 quantization_axis: Union[str, Sequence[float]] = 'global_z'):
        super().__init__(
            dimension=dimension,
            name=name,
            qubit_subspace=qubit_subspace,
            gyromagnetic_ratio=gyromagnetic_ratio,
            axial_splitting=quadrupole_splitting,
            anisotropy_axis=anisotropy_axis,
            local_quasistatic_noise=local_quasistatic_noise,
            quantization_axis=quantization_axis
        )

    @property
    def quadrupole_splitting(self) -> float:
        """Quadrupole splitting parameter (in :math:`\\text{MHz}`)."""
        return self._axial_splitting

    @quadrupole_splitting.setter
    def quadrupole_splitting(self, value: float) -> None:
        self._axial_splitting = value
        self._notify_models()

    def __repr__(self):
        return self._repr_with_extras(
            f'quadrupole_splitting={self.quadrupole_splitting}' if self.quadrupole_splitting != 0 else None,
        )


class Interaction(ModelComponent, Components2D):
    """Represent an interaction between two spins. By appropriately setting the tensor components, this class can model
    various types of spin interactions, such as:

        - The hyperfine interaction between an electron and a nuclear spin.
        - The dipole-dipole interaction between two nuclear spins.

    Initialize an interaction.

    Args:
        spin1: First spin.
        spin2: Second spin.
        tensor: Interaction coefficients (in :math:`\\text{MHz}`, defaults to zero tensor).

    Hint:
        Interaction tensor can be set at initialization:

        .. code-block:: python

            interaction = Interaction(spin1, spin2, tensor=[[0,0,0.003],[0,0,0],[0.003,0,0.213154]])

        Or coefficients can be set one-by-one:

        .. code-block:: python

            interaction = Interaction(spin1, spin2)
            interaction.zz = 0.213154
            interaction.xz = 0.003
            interaction.zx = 0.003

    """

    _component_change_kind = 'interaction_changed'

    def __init__(self,
                 spin1: BaseSpin,
                 spin2: BaseSpin,
                 tensor: np.array = None):
        ModelComponent.__init__(self)
        Components2D.__init__(self, tensor)
        self._attach_callback(self._notify_models)
        self.spin1 = spin1
        """First spin."""

        self.spin2 = spin2
        """Second spin."""

        self.tensor: np.array
        """Full tensor representing the interaction."""

        self.xx: float
        """The :math:`xx` components of the interaction."""

        self.xy: float
        """The :math:`xy` components of the interaction."""

        self.xz: float
        """The :math:`xz` components of the interaction."""

        self.yx: float
        """The :math:`yx` components of the interaction."""

        self.yy: float
        """The :math:`yy` components of the interaction."""

        self.yz: float
        """The :math:`yz` components of the interaction."""

        self.zx: float
        """The :math:`zx` components of the interaction."""

        self.zy: float
        """The :math:`zy` components of the interaction."""

        self.zz: float
        """The :math:`zz` components of the interaction."""

    def __repr__(self):
        diag = [round(float(value), 6) for value in (self.xx, self.yy, self.zz)]
        offdiag_values = [float(value) for value in (self.xy, self.xz, self.yx, self.yz, self.zx, self.zy)]

        if all(value == 0 for value in offdiag_values):
            offdiag = 'zero'
        elif max(abs(value) for value in offdiag_values) <= 1e-6:
            offdiag = 'approx_zero'
        else:
            offdiag = 'nonzero'

        return (
            f"{type(self).__name__}("
            f"spin_name_1='{self.spin1.name}', "
            f"spin_name_2='{self.spin2.name}', "
            f'diag={diag}, '
            f"offdiag='{offdiag}'"
            f')'
        )


    def __setstate__(self, state):
        self.__dict__.update(state)
        _rewire_component_callbacks(self)


def _parse_target_spins(target_spins: Optional[Union[str, Sequence[str]]]) -> Optional[Tuple[str, ...]]:
    """Return an immutable, duplicate-free tuple of spin names or ``None``."""

    if target_spins is None:
        return None

    if isinstance(target_spins, str):
        parsed = (target_spins,)
    else:
        try:
            parsed = tuple(target_spins)
        except TypeError as exc:
            raise SimphonyError('target_spins must be an iterable of spin names') from exc

    if not parsed:
        raise SimphonyError('target_spins must contain at least one spin name when provided')

    if any(not isinstance(name, str) for name in parsed):
        raise SimphonyError('target_spins must contain only string spin names')

    if len(set(parsed)) != len(parsed):
        raise SimphonyError('target_spins must not contain duplicate spin names')

    return parsed


class StaticField(ModelComponent, Components1D):
    """Represent a static magnetic field.

    Initialize the static magnetic field.

    Args:
        strengths: Strengths of the components (in :math:`\\text{T}`, defaults to zero vector).
        target_spins: Iterable (or single string) of spin names the field should affect; ``None`` applies it to every spin.

    Hint:
        Strengths of the components can be set at initialization:
        
        .. code-block:: python
        
            static_field = StaticField([0, 0.001, 0.01])

        Or it can be set after initialization:

        .. code-block:: python

            static_field = StaticField()
            static_field.vec = [0, 0.001, 0.01]

        Or components can be set one-by-one:

        .. code-block:: python

            static_field = StaticField()
            static_field.y = 0.001
            static_field.z = 0.01

    """

    _component_change_kind = 'static_field_changed'

    def __init__(self,
                 strengths: List[float] = None,
                 target_spins: Optional[Sequence[str]] = None):
        ModelComponent.__init__(self)
        Components1D.__init__(self, strengths)
        self._attach_callback(self._notify_models)

        self.vec: Union[List[float], np.array]
        """Vector corresponding the static magnetic field."""

        self.x: float
        """The :math:`x` components of the static magnetic field."""

        self.y: float
        """The :math:`y` components of the static magnetic field."""

        self.z: float
        """The :math:`z` components of the static magnetic field."""

        self.target_spins = target_spins
        """Names of the spins affected by the field, or ``None`` for all spins."""

    def __repr__(self):
        base = f"{type(self).__name__}(strengths={self.vec}"
        if self.target_spins is not None:
            base += f", target_spins={self.target_spins}"
        return base + ")"


    def __setstate__(self, state):
        self.__dict__.update(state)
        _rewire_component_callbacks(self)

    @classmethod
    def _parse_target_spins(cls,
                           target_spins: Optional[Union[str, Sequence[str]]]) -> Optional[Tuple[str, ...]]:
        return _parse_target_spins(target_spins)

    @property
    def target_spins(self) -> Optional[Tuple[str, ...]]:
        return self._target_spins

    @target_spins.setter
    def target_spins(self, target_spins: Optional[Union[str, Sequence[str]]]) -> None:
        parsed = self._parse_target_spins(target_spins)
        if hasattr(self, '_target_spins') and self._target_spins == parsed:
            return
        self._target_spins = parsed
        self._notify_models()


class BaseDrivingField(ModelComponent):
    """Shared pulse-management functionality for driving fields.

    Note:
        In most user code, prefer :class:`LinearDrivingField` or :class:`CircularDrivingField`
        instead of instantiating :class:`BaseDrivingField` directly.

    Args:
        name: Name of the driving field.
        target_spins: Spin names affected by the field, or ``None`` for all spins.

    """

    _component_change_kind = 'driving_field_changed'

    def __init__(self,
                 name: str,
                 target_spins: Optional[Sequence[str]] = None):
        ModelComponent.__init__(self)
        self.name: str = name
        self.last_pulse_end: float = 0.
        self.pulses: PulseList = PulseList()
        self.target_spins = target_spins

    @classmethod
    def _parse_target_spins(cls,
                           target_spins: Optional[Union[str, Sequence[str]]]) -> Optional[Tuple[str, ...]]:
        return _parse_target_spins(target_spins)

    @property
    def target_spins(self) -> Optional[Tuple[str, ...]]:
        """Spin names affected by this driving field, or ``None`` for all spins."""
        return self._target_spins

    @target_spins.setter
    def target_spins(self, target_spins: Optional[Union[str, Sequence[str]]]) -> None:
        parsed = self._parse_target_spins(target_spins)
        if hasattr(self, '_target_spins') and self._target_spins == parsed:
            return
        self._target_spins = parsed
        self._notify_models()

    def remove_all_pulses(self):
        """Remove all previously added pulses from the driving field."""
        self.last_pulse_end = 0.
        self.pulses = PulseList()

    def add_pulse(self,
                  pulse):
        """Add a pulse object to the driving field."""
        if pulse.driving_field is None:
            pulse.driving_field = self
            self.pulses.append(pulse)
        else:
            raise SimphonyError(f'Pulse is already added to {pulse.driving_field}')
        self.last_pulse_end = unp_dynamic.maximum(self.last_pulse_end, pulse.end)

    def add_rectangle_pulse(self,
                            amplitude: float,
                            frequency: float,
                            phase: float,
                            duration: float,
                            start: Optional[float] = None,
                            rise_time: float = 0,
                            fall_time: float = 0,
                            is_effective_amplitude: bool = False):
        """Add a rectangular pulse."""
        if duration <= 0:
            raise SimphonyError('duration must be positive')

        if rise_time < 0 or fall_time < 0:
            raise SimphonyError('rise_time and fall_time must be non-negative')

        tol = 1e-9
        if rise_time + fall_time > duration + tol:
            raise SimphonyError('Sum of the rise_time and fall_time must be less than or equal to the duration')

        if start is None:
            start = self.last_pulse_end

        if is_effective_amplitude:
            amplitude = amplitude * duration / (duration - 0.5 * (rise_time + fall_time))

        if rise_time > 0:
            def rise(t):
                return (
                    unp_dynamic.exp(1j * phase)
                    * 0.5
                    * amplitude
                    * (1 + unp_dynamic.cos(unp_dynamic.pi * (1 - (start - t) / rise_time)))
                )

            pulse = Pulse(start=start,
                          end=start + rise_time,
                          frequency=frequency,
                          complex_envelope=rise,
                          driving_field=self)
            self.pulses.append(pulse)

        pulse = Pulse(start=start + rise_time,
                      end=start + duration - fall_time,
                      frequency=frequency,
                      complex_envelope=amplitude * unp_dynamic.exp(1j * phase),
                      driving_field=self)
        self.pulses.append(pulse)

        if fall_time > 0:
            def fall(t):
                return (
                    unp_dynamic.exp(1j * phase)
                    * 0.5
                    * amplitude
                    * (1 + unp_dynamic.cos(unp_dynamic.pi * (1 - (start + duration - t) / fall_time)))
                )

            pulse = Pulse(start=start + duration - fall_time,
                          end=start + duration,
                          frequency=frequency,
                          complex_envelope=fall,
                          driving_field=self)
            self.pulses.append(pulse)

        self.last_pulse_end = unp_dynamic.maximum(self.last_pulse_end, start + duration)

    def add_discrete_pulse(self,
                           samples: List[Union[float, complex]],
                           frequency: float,
                           dt: Union[float, List[float]],
                           start: Optional[float] = None):
        """Add a piecewise-constant pulse."""
        samples = unp_dynamic.asarray(samples, dtype=complex)
        len_samples = samples.shape[0]

        if start is None:
            start = self.last_pulse_end

        dt = unp_dynamic.atleast_1d(unp_dynamic.asarray(dt))
        len_dt = dt.shape[0]

        if len_dt == 1:
            if dt[0] <= 0:
                raise SimphonyError('dt must be positive')
            ts = unp_dynamic.linspace(start, start + dt[0] * len_samples, len_samples + 1)
        elif len_dt == samples.shape[0]:
            if unp_dynamic.any(dt <= 0):
                raise SimphonyError('All dt must be positive')
            ts = unp_dynamic.concatenate([unp_dynamic.asarray([start]), start + unp_dynamic.cumsum(dt)])
        else:
            raise SimphonyError('ts must be a single value, or dt and samples must have the same length')

        for sample, start, end in zip(samples, ts[:-1], ts[1:]):
            pulse = Pulse(start=start,
                          end=end,
                          frequency=frequency,
                          complex_envelope=sample,
                          driving_field=self)
            self.pulses.append(pulse)

        self.last_pulse_end = unp_dynamic.maximum(self.last_pulse_end, ts[-1])

    def add_wait(self,
                 duration: float):
        """Add an idle section to the driving field."""
        if duration <= 0:
            raise SimphonyError('duration must be positive')
        self.last_pulse_end += duration

    def plot_pulses(self,
                    function: str = 'full_waveform',
                    start: float = 0.,
                    end: Optional[float] = None,
                    ax=None):
        """Plot the added pulse(s)."""
        if end is None:
            end = self.last_pulse_end

        if end <= start:
            raise SimphonyError('end must be greater than start')

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2.5))

        time_segment_sequence = self.pulses.convert_to_time_segment_sequence(start=start,
                                                                             end=end)
        self._time_segment_sequence = time_segment_sequence
        time_segment_sequence.plot(function=function, ax=ax)

        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel('$B_{ac}$ [T]')
        ax.grid()
        ax.set_title('{} ({})'.format(self.name, function))
        plt.tight_layout()

    def _component_directions(self) -> List[List[float]]:
        raise NotImplementedError


    def _component_coefficients(self) -> List[complex]:
        raise NotImplementedError

    def _component_signals(self,
                          pulse_list: Optional['PulseList'] = None,
                          clip: bool = True) -> List[Union['Signal', 'SignalSum']]:
        coefficients = self._component_coefficients()
        signals = [Signal(0.0) for _ in coefficients]
        pulses = self.pulses if pulse_list is None else pulse_list

        for pulse in pulses:
            if pulse.driving_field is None or pulse.driving_field.name != self.name:
                continue

            for idx, coefficient in enumerate(coefficients):
                if clip:
                    if pulse.is_constant_envelope:
                        def envelope(ts, pulse=pulse, coefficient=coefficient):
                            ts_arr = unp_dynamic.asarray(ts)
                            return unp_dynamic.where(
                                ts_arr < pulse.start,
                                0.,
                                unp_dynamic.where(ts_arr < pulse.end, coefficient * pulse.complex_envelope, 0.),
                            )
                    else:
                        def envelope(ts, pulse=pulse, coefficient=coefficient):
                            ts_arr = unp_dynamic.asarray(ts)
                            values = coefficient * pulse.complex_envelope(ts_arr)
                            return unp_dynamic.where(
                                ts_arr < pulse.start,
                                0.,
                                unp_dynamic.where(ts_arr < pulse.end, values, 0.),
                            )
                else:
                    if pulse.is_constant_envelope:
                        envelope = coefficient * pulse.complex_envelope
                    else:
                        def envelope(ts, pulse=pulse, coefficient=coefficient):
                            return coefficient * pulse.complex_envelope(ts)

                signals[idx] += Signal(envelope=envelope, frequency=pulse.frequency)

        return signals


class LinearDrivingField(BaseDrivingField, Components1D):
    """Represent a linearly polarized AC magnetic field.

    Args:
        name: Name of the driving field.
        direction: Polarization direction of the field.
        target_spins: Spin names affected by the field, or ``None`` for all spins.

    """

    def __init__(self,
                 name: str,
                 direction: List[float],
                 target_spins: Optional[Sequence[str]] = None):
        Components1D.__init__(self, direction)
        self.normalize()
        BaseDrivingField.__init__(self, name=name, target_spins=target_spins)
        self._attach_callback(self._notify_models)

    def __repr__(self):
        base = f"{type(self).__name__}(name='{self.name}', direction={self.vec}"
        if self.target_spins is not None:
            base += f", target_spins={self.target_spins}"
        return base + ")"


    def __setstate__(self, state):
        self.__dict__.update(state)
        _rewire_component_callbacks(self)

    def _component_directions(self) -> List[List[float]]:
        return [self.vec]

    def _component_coefficients(self) -> List[complex]:
        return [1.0]


class CircularDrivingField(BaseDrivingField):
    """Represent a circularly polarized AC magnetic field in a plane.

    Args:
        name: Name of the driving field.
        plane_normal: Normal vector of the polarization plane.
        phase_reference: Reference direction defining the zero-phase axis in the plane.
        target_spins: Spin names affected by the field, or ``None`` for all spins.

    Note:
        ``phase_reference`` is projected onto the plane orthogonal to ``plane_normal`` and normalized.
        The projected direction defines the in-phase polarization axis, while the quadrature axis is
        constructed as ``cross(plane_normal, phase_reference)``. Therefore ``phase_reference`` must
        have a non-zero component within the polarization plane.

    """

    def __init__(self,
                 name: str,
                 plane_normal: List[float],
                 phase_reference: List[float],
                 target_spins: Optional[Sequence[str]] = None):
        BaseDrivingField.__init__(self, name=name, target_spins=target_spins)

        self.plane_normal = Components1D(plane_normal)
        self.phase_reference = Components1D(phase_reference)
        self.phase_quadrature = Components1D()
        self.plane_normal._attach_callback(self._on_geometry_changed)
        self.phase_reference._attach_callback(self._on_geometry_changed)
        self.phase_quadrature._attach_callback(self._on_geometry_changed)
        self._refresh_geometry()

    def _refresh_geometry(self) -> None:
        normal_vec = np.asarray(self.plane_normal.vec, dtype=float)
        normal_norm = np.linalg.norm(normal_vec)
        if normal_norm <= 1e-12:
            raise SimphonyError('plane_normal must be non-zero')
        normal_vec = normal_vec / normal_norm

        reference_vec = np.asarray(self.phase_reference.vec, dtype=float)
        projected_reference = reference_vec - np.dot(reference_vec, normal_vec) * normal_vec
        projected_norm = np.linalg.norm(projected_reference)
        if projected_norm <= 1e-12:
            raise SimphonyError('phase_reference must have a non-zero component in the polarization plane')
        projected_reference = projected_reference / projected_norm

        phase_quadrature_vec = np.cross(normal_vec, projected_reference)
        quadrature_norm = np.linalg.norm(phase_quadrature_vec)
        if quadrature_norm <= 1e-12:
            raise SimphonyError('Unable to construct a quadrature direction for the circular driving field')
        phase_quadrature_vec = phase_quadrature_vec / quadrature_norm

        self.plane_normal._vec = normal_vec.tolist()
        self.phase_reference._vec = projected_reference.tolist()
        self.phase_quadrature._vec = phase_quadrature_vec.tolist()

    def _on_geometry_changed(self) -> None:
        self._refresh_geometry()
        self._notify_models()

    def __repr__(self):
        base = (
            f"{type(self).__name__}(name='{self.name}', plane_normal={self.plane_normal.vec}, "
            f"phase_reference={self.phase_reference.vec}"
        )
        if self.target_spins is not None:
            base += f", target_spins={self.target_spins}"
        return base + ")"


    def __setstate__(self, state):
        self.__dict__.update(state)
        _rewire_component_callbacks(self)

    def _component_directions(self) -> List[List[float]]:
        return [self.phase_reference.vec, self.phase_quadrature.vec]

    def _component_coefficients(self) -> List[complex]:
        scale = 1.0 / np.sqrt(2.0)
        return [scale, -1j * scale]


class Signal:
    """A real-valued harmonic signal.

    Args:
        envelope: Constant value or callable function.
        frequency: Frequency of the signal.

    Note:
        The signal is the real part of a complex exponential:

        .. math::
            s(t) = \\Re\\{A(t)e^{i 2 \\pi f t}\\},

        where :math:`A(t)` is represented by the ``envelope``, :math:`f` represented by the ``frequency``, and
        :math:`t` is the time.
    """

    def __init__(self, envelope: Union[complex, Callable[[float], complex], Callable[[uarray], uarray]] = 0.0,
                 frequency: float = 0.0):
        self.envelope = envelope
        self.frequency = frequency

    def __call__(self, ts: uarray) -> uarray:
        frequency = unp_dynamic.asarray(self.frequency)
        ts_arr = unp_dynamic.asarray(ts)
        env = self.envelope(ts_arr) if callable(self.envelope) else unp_dynamic.asarray(self.envelope)
        return unp_dynamic.real(
            env * unp_dynamic.exp(1j * 2.0 * unp_dynamic.pi * frequency * ts_arr)
        )

    def __add__(self, other: Signal) -> SignalSum:
        return SignalSum([self, other])

    def __radd__(self, other: Signal) -> SignalSum:
        return SignalSum([other, self])

    def __iadd__(self, other: Signal) -> SignalSum:
        return SignalSum([self, other])


class SignalSum:
    """Sum of signals."""

    def __init__(self, signals: Optional[Sequence[Signal]] = None):
        self.signals: List[Signal] = list(signals) if signals is not None else []

    def __call__(self, ts: uarray) -> uarray:
        if not self.signals:
            return unp_dynamic.zeros_like(unp_dynamic.asarray(ts), dtype=unp_dynamic.complex64)
        out = self.signals[0](ts)
        for s in self.signals[1:]:
            out = out + s(ts)
        return out

    def __add__(self, other: Union[Signal, SignalSum]) -> SignalSum:
        if isinstance(other, SignalSum):
            return SignalSum(self.signals + other.signals)
        return SignalSum(self.signals + [other])

    def __radd__(self, other: Union[Signal, SignalSum]) -> SignalSum:
        if isinstance(other, SignalSum):
            return SignalSum(other.signals + self.signals)
        return SignalSum([other] + self.signals)

    def __iadd__(self, other: Union[Signal, SignalSum]) -> SignalSum:
        if isinstance(other, SignalSum):
            self.signals.extend(other.signals)
        else:
            self.signals.append(other)
        return self

class Pulse:
    """Represents a single pulse.

    The class defines a finite-duration pulse characterized by a constant frequency and a specified complex envelope. It
    can be attached to a driving field to specify its time-dependent behavior.

    Note:
        :class:`Pulse` is a low-level object representing a pulse or a pulse fragment (e.g. rising edge of a smooth
        pulse). To add pulses to driving fields, use the appropriate :meth:`add_..._pulse` methods of the corresponding
        :class:`LinearDrivingField` or :class:`CircularDrivingField` instances.

    Initialize a pulse.

    Args:
        start: Trigger (start) time of the pulse (in :math:`\\mu\\text{s}`).
        end: End time of the pulse (in :math:`\\mu\\text{s}`).
        frequency: Frequency of the pulse (in :math:`\\text{MHz}`).
        complex_envelope: Complex envelope of the pulse. It can be a complex number or a callable returning a
            complex-valued function (in :math:`\\text{T}`).
        driving_field: Driving field which the pulse is attached to.

    """

    _component_change_kind = 'static_field_changed'

    def __init__(self,
                 start: float,
                 end: float,
                 frequency: float,
                 complex_envelope: Union[complex, Callable],
                 driving_field: Optional[BaseDrivingField] = None):

        self.start: float = start
        """Trigger (start) time of the pulse (in :math:`\\mu\\text{s}`)."""

        if end <= start:
            raise SimphonyError('end should be greater than start')

        self.end: float = end
        """End time of the pulse (in :math:`\\mu\\text{s}`)."""

        self.complex_envelope: Union[complex, Callable] = complex_envelope
        """Complex envelope of the pulse (in :math:`\\text{T}`)."""

        self.frequency = frequency
        """Frequency of the pulse (in :math:`\\text{MHz}`)."""

        self.driving_field = driving_field
        """Driving field which the pulse is attached to."""

    @property
    def abs_frequency(self) -> float:
        """Absolute value of the pulse frequency (in :math:`\\text{MHz}`)."""
        return abs(self.frequency)

    def __repr__(self):
        class_name = type(self).__name__
        envelope = (
            self.complex_envelope if self.is_constant_envelope else f"fn(id={id(self.complex_envelope)})"
        )
        return (
            f"{class_name}("
            f"start={self.start}, "
            f"end={self.end}, "
            f"frequency={self.frequency}, "
            f"complex_envelope={envelope}, "
            f"driving_field={self.driving_field.name if self.driving_field else None})"
        )

    @property
    def is_constant_envelope(self) -> bool:
        """Returns whether the pulse has a constant envelope or not."""
        return not callable(self.complex_envelope)

    @property
    def signal(self) -> Signal:
        """Return this pulse as a Simphony ``Signal`` with its carrier and envelope."""
        if self.is_constant_envelope:
            def complex_envelope_fn(ts):
                ts_arr = unp_dynamic.asarray(ts)
                return unp_dynamic.where(ts_arr < self.start, 0.,
                                         unp_dynamic.where(ts_arr < self.end, self.complex_envelope, 0.))
        else:
            def complex_envelope_fn(ts):
                ts_arr = unp_dynamic.asarray(ts)
                envelope = self.complex_envelope(ts_arr)
                return unp_dynamic.where(ts_arr < self.start, 0.,
                                         unp_dynamic.where(ts_arr < self.end, envelope, 0.))

        return Signal(envelope=complex_envelope_fn, frequency=self.frequency)


@dataclass
class SimulationSpecBase:
    """Metadata describing a discretized simulation time segment."""

    method: str
    """Name of the integration backend (for example ``"basic"`` or ``"single_sine_wave"``)."""

    t_span: uarray
    """Two-element array containing the start and end times (in microseconds)."""

    t_eval: uarray
    """Monotonic array of time samples where solver output is requested."""

    max_dt: float
    """Largest allowed time step when refining the integration grid."""

    signals: List[Union[Signal, SignalSum]]
    """Signals (``Signal`` or ``SignalSum``) scheduled to drive the segment."""

    time_grid_jitter: Optional[float]
    """Amplitude of random jitter applied per sub-interval (``None`` disables jitter)."""


@dataclass
class SimulationSpecBasic(SimulationSpecBase):
    """Simulation metadata for the basic method."""
    pass


@dataclass
class SimulationSpecSingleSineWave(SimulationSpecBase):
    """Simulation metadata optimized for single sine wave method."""

    periodic_time: float
    """Duration of one period of the driving tone (in microseconds)."""

    t_span_projected_sorted: uarray
    """Start/end times of the base cycle used for single-period integration."""

    t_eval_projected_sorted: uarray
    """Sorted projection of ``t_eval`` into the first period."""

    unsort_indices: uarray
    """Indices that restore the original ``t_eval`` ordering after projection."""

    cycle_indices: uarray
    """Integer index of the period each ``t_eval`` sample belongs to."""


@dataclass
class PlotSpec:
    """Samples used for plotting time-segments."""

    function: str
    """Identifier of the sampled quantity (for example ``"signal"`` or ``"envelope"``)."""

    ts: uarray
    """Time samples associated with the plotted data."""

    ps: Union[uarray, List[uarray]]
    """One or more arrays containing values evaluated at ``ts``."""


class PulseList:
    """
    Represent multiple pulses.

    The class behaves similarly to a standard :class:`list`, it supports indexing, setting items, iteration, and
    querying its length. It is specifically tailored for handling collections of :class:`Pulse` instances.

    Initialize a list of pulses.

    Args:
        iterable: Initial list of :class:`Pulse` instances.

    Hint:
        - Iterable over :class:`Pulse` elements:

        .. code-block:: python

            for pulse in pulse_list:
                print(pulse)

        - Supports indexing and item assignment:

        .. code-block:: python

            pulse = pulse_list[0]
            pulse_list[1] = Pulse(...)

        - Concatenate with another ``PulseList``:

        .. code-block:: python

            new_pulse_list = pulse_list1 + pulse_list2

    """

    def __init__(self,
                 iterable: Optional[List[Pulse]] = None):
        if iterable is not None:
            self._pulses = iterable
        else:
            self._pulses = []

    def __len__(self) -> int:
        return len(self._pulses)

    def __getitem__(self, index: int) -> Pulse:
        return self._pulses[index]

    def __setitem__(self, index: int, value: Pulse) -> None:
        self._pulses[index] = value

    def __repr__(self):
        return f"{type(self).__name__}(n_pulses={len(self)})"

    def __str__(self):
        lines = [f"{type(self).__name__}(["]
        for pulse in self._pulses:
            lines.append(f"   {repr(pulse)},")
        lines.append("])")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[Pulse]:
        return iter(self._pulses)

    def append(self,
               new_pulse: Pulse) -> None:
        """Append a pulse."""
        self._pulses.append(new_pulse)

    def __add__(self, other: PulseList) -> PulseList:
        new = self._pulses + other._pulses
        return self.__class__(new)

    def signal(self,
               driving_field_name: Optional[str] = None,
               clip: bool = True) -> Union[Signal, SignalSum]:
        """Aggregate pulses into a single Signal or SignalSum.

        Args:
            driving_field_name: If given, include only pulses whose driving_field
                name matches this value.
            clip: If True, sum each pulse's gated ``Pulse.signal``; if False,
                sum raw envelopes without per‑pulse gating. Default: True.

        Returns:
            A Signal (or SignalSum) representing the sum of all included pulses.
        """
        sig = Signal(0.0)
        for pulse in self:
            if (driving_field_name is None) or (pulse.driving_field and pulse.driving_field.name == driving_field_name):
                if clip:
                    sig += pulse.signal
                else:
                    sig += Signal(frequency=pulse.frequency, envelope=pulse.complex_envelope)
        return sig

    @property
    def n_pulses(self) -> int:
        """Number of pulses."""
        return len(self)

    @property
    def is_all_constant_envelope(self) -> bool:
        """Check if all pulses have a constant envelope."""
        return all([pulse.is_constant_envelope for pulse in self])

    def convert_to_time_segment_sequence(self,
                                         start: Optional[float] = None,
                                         end: Optional[float] = None) -> TimeSegmentSequence:
        """Convert the to :class:`TimeSegmentSequence`.

        Args:
            start: Start time of the inspected time interval (defaults to the start of the first pulse).
            end: End time of the inspected time interval (defaults to the end of the last pulse).

        Returns:
            A sequence of time segments, each containing pulses that overlap with the respective time interval.
        """
        division_points = _calculate_division_points(
            self,
            start=start,
            end=end
        )

        tol = 1e-10
        time_segment_sequence = []
        for segment_start, segment_end in zip(division_points, division_points[1:]):
            pulses = PulseList(
                [
                    pulse for pulse in self
                    if pulse.start <= segment_start + tol and pulse.end >= segment_end - tol
                ]
            )
            time_segment = TimeSegment(
                start=segment_start,
                end=segment_end,
                pulses=pulses
            )
            time_segment_sequence.append(time_segment)

        return TimeSegmentSequence(time_segment_sequence)


def _calculate_division_points(pulse_list: PulseList,
                               start: float = None,
                               end: float = None) -> uarray:
    """Return the division points of a list of pulses, where each point corresponds to either the start or end of a
    pulse. The returned array is sorted in ascending order.

    Args:
        pulse_list: List of pulses to analyze.
        start: Start time of the inspected time interval.
        end: End time of the inspected time interval.

    Returns:
        A sorted array of division points, corresponding to pulse boundaries.

    """

    division_points = [t for pulse in pulse_list for t in (pulse.start, pulse.end)
                       if (start is None or t > start) and (end is None or t < end)]
    if start is not None:
        division_points.insert(0, start)
    if end is not None:
        division_points.append(end)
    division_points = unp_dynamic.asarray(division_points, dtype=unp_dynamic.float64)

    division_points = unp_dynamic.sort(division_points)

    tol = 1e-10
    idxs = unp_dynamic.diff(division_points) > tol
    idxs = unp_dynamic.insert(idxs, 0, True)
    division_points = division_points[idxs]

    return division_points


class TimeSegment:
    """Represents a time segment associated with a list of pulses.

    A ``TimeSegment`` is a time interval that includes all pulses which are
    active throughout that interval. It is a read-only view that wraps a
    :class:`PulseList` and provides additional metadata for plotting and
    simulation.

    Initialize a time segment.

    Args:
        start: Start time of the segment.
        end: End time of the segment.
        pulses: Active pulses corresponding to the segment.

    """

    def __init__(self,
                 start,
                 end,
                 pulses):
        self._pulses: PulseList = pulses if isinstance(pulses, PulseList) else PulseList(pulses)

        self.start: float = start
        """Start time of the segment."""

        self.end: float = end
        """End time of the segment."""

        self.plot: Optional[PlotSpec] = None
        self.simulation: Optional[SimulationSpecBase] = None

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'start={self.start}, '
                f'end={self.end}, '
                f'n_pulses={self.n_pulses}, '
                f"simulation_type={'None' if self.simulation is None else repr(self.simulation.method)}"
                ')'
                )

    def append(self,
               new_pulse: Pulse):
        """Appending a pulse is disabled for TimeSegment instances."""
        raise SimphonyError('Appending a pulse is disabled for TimeSegment instances.')

    def __len__(self) -> int:
        return len(self._pulses)

    def __getitem__(self, index: int) -> Pulse:
        return self._pulses[index]

    def __iter__(self) -> Iterator[Pulse]:
        return iter(self._pulses)

    @property
    def duration(self) -> float:
        """Duration of the time segment."""
        return self.end - self.start

    @property
    def max_frequency(self) -> float:
        """Maximal frequency of the time segment's pulses."""
        if self.n_pulses == 0:
            return 0.
        elif self.n_pulses == 1:
            return self[0].abs_frequency
        else:
            return max(pulse.abs_frequency for pulse in self)

    @property
    def n_pulses(self) -> int:
        """Number of pulses in the segment."""
        return len(self._pulses)

    @property
    def is_all_constant_envelope(self) -> bool:
        """Check if all pulses have a constant envelope."""
        return all(pulse.is_constant_envelope for pulse in self._pulses)

    @property
    def n_nonzero_frequencies(self) -> int:
        """Number of unique non-zero frequencies in the segment."""
        tol = 1e-10
        frequencies = []

        for pulse in self._pulses:
            try:
                frequency = np.asarray(pulse.abs_frequency, dtype=float).reshape(()).item()
            except (TypeError, ValueError):
                return 0

            if frequency <= tol:
                continue

            if not any(abs(frequency - reference) <= tol for reference in frequencies):
                frequencies.append(frequency)

        return len(frequencies)

    def signal(self,
               driving_field_name: Optional[str] = None) -> Union[Signal, SignalSum]:
        """Return the combined signal for this time segment.

        Note:
            For performance, no extra segment‑level gate is applied. Evaluate
            only on this segment's discretized time grid.
        """
        return self._pulses.signal(driving_field_name, clip=False)

    def discretize_plot(self,
                        function: str):
        """Discretize the time segment for plotting. Results are returned in :attr:`plot_ts` (discrete time axis) and
        :attr:`plot_ps` (signal or envelope values).

        Args:
            function: Specify to show the full waveform (``'full_waveform'``) or only the complex envelope
                (``'complex_envelope'``)

        """

        if function == 'full_waveform':
            if self.n_pulses == 0:
                ts = unp_dynamic.asarray([self.start, self.end])
                ps = unp_dynamic.asarray([0., 0.])
            else:
                max_frequency = self.max_frequency
                duration = self.duration
                n_points = int(duration * max_frequency + 1) * 251
                max_n_points = 10000
                if n_points > max_n_points:
                    n_points = max_n_points
                ts = unp_dynamic.linspace(self.start, self.end, n_points)
                signal = self.signal()
                ps = signal(ts)

        elif function == 'complex_envelope':
            if self.n_pulses == 0:
                ts = unp_dynamic.asarray([self.start, self.end])
                ps = [unp_dynamic.asarray([0., 0.])]
            elif self.is_all_constant_envelope:
                ts = unp_dynamic.asarray([self.start, self.end])
                ps = [unp_dynamic.asarray([pulse.complex_envelope for _ in ts]) for pulse in self]
            else:
                n_points = 251
                ts = unp_dynamic.linspace(self.start, self.end, n_points)
                ps = [unp_dynamic.asarray([
                    pulse.complex_envelope if pulse.is_constant_envelope else pulse.complex_envelope(t) for t in ts
                ]) for pulse in self]

        else:
            raise SimphonyError("Invalid function, must be 'complex_envelope' or 'full_waveform'")
        self.plot = PlotSpec(function=function, ts=ts, ps=ps)

    def discretize_simulation(self,
                              included_driving_field_names: List[str],
                              method: str,
                              n_eval: int = 251,
                              n_split: int = 250,
                              time_grid_jitter: Optional[float] = None):
        """Discretize the time segment for simulation.

        For simulation, the time is discretized using one of methods:

        - ``'basic'``:
            Uniform discretization of the entire time segment. The maximum time step is chosen such that one period of
            the highest-frequency pulse in the segment is divided into ``n_split`` time steps. If the envelope is not
            constant, the time step is additionally limited by the total segment duration to ensure adequate resolution.
            This is the general-purpose option.

        - ``'single_sine_wave'``:
            Optimized for segments with a single constant-enveloped pulse. Instead of evaluating the full time segment,
            only one cycle of the sine wave is simulated. The full pulse is then reconstructed from this sample,
            resulting in lower memory usage and faster computation in applicable cases. If the segment does not meet the
            required conditions (multiple pulses or non-constant envelopes), this method falls back to ``'basic'``.

        Results are stored in the :attr:`simulation_...` attributes.

        Args:
            included_driving_field_names: Specify which driving-field names or field objects are stored in
                :attr:`simulation_signals`.
            method: Could be ``'basic'`` or ``'single_sine_wave'``.
            n_eval: Number of evaluation points at which the simulation results (time evolution operator or
                wave function) will be computed and returned.
            n_split: Number of time steps used discretize to one period of the highest-frequency component.
            time_grid_jitter: Amplitude of random jitter applied within each subinterval (set ``None`` to disable).

        """

        if method == 'basic':
            self._discretize_simulation_basic(n_eval=n_eval, n_split=n_split, time_grid_jitter=time_grid_jitter)
        elif method == 'single_sine_wave':
            self._discretize_simulation_single_sine_wave(n_eval=n_eval, n_split=n_split, time_grid_jitter=time_grid_jitter)
        else:
            raise SimphonyError("method must be 'basic' or 'single_sine_wave'")
        if self.simulation is None:
            raise SimphonyError('Simulation spec was not initialized')

        signals = []
        for driving_field in included_driving_field_names:
            if isinstance(driving_field, str):
                signals.append(self.signal(driving_field))
            else:
                signals.extend(driving_field._component_signals(self._pulses, clip=False))
        self.simulation.signals = signals

    def _discretize_simulation_basic(self,
                                     n_eval: int = 251,
                                     n_split: int = 250,
                                     time_grid_jitter: Optional[float] = None):
        max_frequency = self.max_frequency
        if max_frequency == 0:
            if self.is_all_constant_envelope:
                max_dt = self.duration
            else:
                max_dt = self.duration / n_split
        else:
            if self.is_all_constant_envelope:
                max_dt = 1. / max_frequency / n_split
            else:
                max_dt = unp_dynamic.min(unp_dynamic.asarray([1. / max_frequency, self.duration])) / n_split

        self.simulation = SimulationSpecBasic(
            method='basic',
            t_span=unp_dynamic.asarray([self.start, self.end]),
            t_eval=unp_dynamic.linspace(self.start, self.end, n_eval),
            max_dt=max_dt,
            signals=[],
            time_grid_jitter=time_grid_jitter
        )

    def _discretize_simulation_single_sine_wave(self,
                                                n_eval: int = 251,
                                                n_split: int = 250,
                                                time_grid_jitter: Optional[float] = None):
        if self.n_nonzero_frequencies == 1 and self.is_all_constant_envelope and 1. / self.max_frequency <= self.duration:
            periodic_time = 1. / self.max_frequency
            max_dt = periodic_time / n_split

            t_eval = unp_dynamic.linspace(self.start, self.end, n_eval)
            cycle_indices, t_eval_projected = unp_dynamic.divmod(t_eval - self.start, periodic_time)
            cycle_indices = cycle_indices.astype(int)
            t_eval_projected += self.start

            sort_indices = unp_dynamic.argsort(t_eval_projected)
            t_eval_projected_sorted = t_eval_projected[sort_indices]
            unsort_indices = unp_dynamic.argsort(sort_indices)

            t_end_first_sine_wave = self.start + periodic_time
            t_eval_projected_sorted = unp_dynamic.append(t_eval_projected_sorted, t_end_first_sine_wave)

            self.simulation = SimulationSpecSingleSineWave(
                method='single_sine_wave',
                t_span=unp_dynamic.asarray([self.start, self.end]),
                t_eval=t_eval,
                max_dt=max_dt,
                signals=[],
                time_grid_jitter=time_grid_jitter,
                periodic_time=periodic_time,
                t_span_projected_sorted=unp_dynamic.asarray([self.start, t_end_first_sine_wave]),
                t_eval_projected_sorted=t_eval_projected_sorted,
                unsort_indices=unsort_indices,
                cycle_indices=cycle_indices,
            )

        else:
            self._discretize_simulation_basic(n_eval=n_eval, n_split=n_split, time_grid_jitter=time_grid_jitter)


class TimeSegmentSequence:
    """Represent sequence of time segments.

    The class is a high-level structure that organizes a pulses into consecutive, non-overlapping :class:`TimeSegment`
    objects. Each segment contains the pulses that are active during its time interval and provides metadata for
    simulation and visualization.

    Initialize a time segment list.

    Args:
        iterable: Initial list of time segments.
    """

    def __init__(self,
                 iterable=None):
        if iterable is not None:
            self._time_segments = iterable
        else:
            self._time_segments = []

    def __len__(self) -> int:
        return len(self._time_segments)

    def __getitem__(self, index: int):
        return self._time_segments[index]

    def __iter__(self) -> Iterator[TimeSegment]:
        return iter(self._time_segments)

    def __repr__(self):
        return f"{type(self).__name__}(n_time_segments={len(self)})"

    def __str__(self):
        lines = [f"{type(self).__name__}(["]
        for segment in self._time_segments:
            lines.append(f"   {repr(segment)},")
        lines.append("])")
        return "\n".join(lines)

    def plot(self,
             function: str = 'full_waveform',
             ax=None):
        """Plot the pulses divided by the time segments.

        Args:
            function: Could be ``'full_waveform'`` or ``'complex_envelope'``.
            ax: Axis to customize the plot.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))

        if function == 'full_waveform':

            ts = []
            ps = []
            for time_segment in self:
                time_segment.discretize_plot(function=function)
                ts.append(time_segment.plot.ts)
                ps.append(time_segment.plot.ps)
            ts = unp_dynamic.concatenate(ts)
            ps = unp_dynamic.concatenate(ps)

            ts = stop_gradient(ts)
            ps = stop_gradient(ps)

            ax.plot(ts, ps)

        elif function == 'complex_envelope':

            c_real = '#1f77b4'
            c_imag = '#ff7f0e'

            ps_end_real = [0.]
            ps_end_imag = [0.]

            for idx, time_segment in enumerate(self):
                time_segment.discretize_plot(function=function)

                ps_start_real = ps_end_real
                ps_start_imag = ps_end_imag
                ps_end_real = []
                ps_end_imag = []

                plot_ts = stop_gradient(time_segment.plot.ts)
                plot_ps = stop_gradient(time_segment.plot.ps)

                for ps in plot_ps:
                    ps_start_real.append(ps[0].real)
                    ps_start_imag.append(ps[0].imag)
                    ps_end_real.append(ps[-1].real)
                    ps_end_imag.append(ps[-1].imag)
                t_start = plot_ts[0]

                ax.plot([t_start, t_start], [min(ps_start_real), max(ps_start_real)], c=c_real)
                ax.plot([t_start, t_start], [min(ps_start_imag), max(ps_start_imag)], '--', c=c_imag)

                for ps in plot_ps:
                    ax.plot(plot_ts, unp_dynamic.real(ps), c=c_real)
                    ax.plot(plot_ts, unp_dynamic.imag(ps), '--', c=c_imag)

                if idx == len(self) - 1:
                    t_end = plot_ts[-1]
                    ps_end_real.append(0.)
                    ps_end_imag.append(0.)
                    ax.plot([t_end, t_end], [min(ps_end_real), max(ps_end_real)], c=c_real)
                    ax.plot([t_end, t_end], [min(ps_end_imag), max(ps_end_imag)], '--', c=c_imag)

            ax.legend(['real', 'imag'])

        else:
            raise SimphonyError("Invalid function, must be 'complex_envelope' or 'full_waveform'")

    def discretize_simulation(self,
                              included_driving_field_names: List[str],
                              method: str,
                              n_eval: int = 251,
                              n_split: int = 250,
                              time_grid_jitter: Optional[float] = None):
        """Discretize the time segment sequence for simulation.

        Args:
            included_driving_field_names: Specify which driving-field names or field objects are included in
                :attr:`simulation_signals`.
            method: Specify the method. It could be ``'basic'`` or ``'single_sine_wave'``.
            n_eval: Number of evaluation points.
            n_split: Number of splitting points for non-constant envelope and for single sine wave.
            time_grid_jitter: Amplitude of random jitter applied when subdividing time steps (``None`` disables jitter).
        """

        for time_segment in self:
            time_segment.discretize_simulation(included_driving_field_names=included_driving_field_names,
                                               method=method,
                                               n_eval=n_eval,
                                               n_split=n_split,
                                               time_grid_jitter=time_grid_jitter)
