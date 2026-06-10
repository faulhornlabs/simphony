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

"""Model of a coupled spin system."""

from .config import Config, unp_static, unp_dynamic
from .exceptions import SimphonyError, SimphonyWarning
from .components import (
    ModelComponent, BaseSpin, ElectronSpin, NuclearSpin, Interaction, StaticField, BaseDrivingField, PulseList
)
from .utils import tensorprod, average_gate_fidelity_from_unitaries, expm, _axis_angle_from_local_z_to_direction
from .analysis import SimulationResult, TimeEvolOperator, TimeEvolState
from .solvers import Solver, solve_time_segment_sequence

from typing import Union, List, Tuple, Dict, Optional, Sequence, Set
import contextlib
import io
import warnings
from warnings import warn

from copy import deepcopy
from itertools import product

import math

import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text

from qiskit.quantum_info import Pauli

NUM_DECORATORS_VERBOSE = 75

uarray = Union[np.ndarray, jnp.ndarray]


@contextlib.contextmanager
def _quiet_adjust_text():
    """Suppress noisy adjustText stdout/debug warnings during label placement."""

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings("ignore", category=UserWarning, module=r"adjustText")
        yield

_BUILD_DEPENDENCIES: Dict[str, Set[str]] = {
    'quantization_transform': set(),
    'static_hamiltonian': {'quantization_transform'},
    'eigensystem': {'static_hamiltonian'},
    'rotating_frame': {'eigensystem'},
    'driving_operators': {'quantization_transform'},
    'noise_operators': {'quantization_transform'},
}


class RotatingFrameSetter(ModelComponent):
    """Set the rotating-frame frequency of each spin from the corresponding qubit splitting.

    When attached to a model, this helper recomputes those frequencies automatically
    as the Hamiltonian changes. For each target spin, it evaluates
    ``model.splitting_qubit(...)`` under the configured conditioning states and
    writes the result back to ``spin.rotating_frame_frequency``.

    Args:
        rest_electron_spin_state: How the other electron spins are fixed when
            computing the qubit splitting of a target spin. Use ``'0'`` or
            ``'1'`` to fix them to a qubit state, or ``'avg'`` to average over
            them. A single label is broadcast to all electron spins; a
            dictionary assigns values by spin name.
        rest_nuclear_spin_state: How the other nuclear spins are fixed when
            computing the qubit splitting of a target spin. Use ``'0'`` or
            ``'1'`` to fix them to a qubit state, or ``'avg'`` to average over
            them. A single label is broadcast to all nuclear spins; a
            dictionary assigns values by spin name.

    Hint:
        The conditioning states apply only to the other spins while the qubit
        splitting of a target spin is computed.

        Consider a model with one electron spin ``'e'`` and two nuclear spins
        ``'N'`` and ``'C'``.

        1. With the default setting,

        ``RotatingFrameSetter(rest_electron_spin_state='1', rest_nuclear_spin_state='avg')``

        the rotating-frame frequencies are computed as follows:

        - for ``'e'``: use the qubit splitting of ``'e'`` averaged over ``'N'``
          and ``'C'``;
        - for ``'N'``: use the qubit splitting of ``'N'`` conditioned on ``'e'``
          being in the :math:`|1\\rangle` qubit state and averaged over ``'C'``;
        - for ``'C'``: use the qubit splitting of ``'C'`` conditioned on ``'e'``
          being in the :math:`|1\\rangle` qubit state and averaged over ``'N'``.

        2. With

        ``RotatingFrameSetter(rest_electron_spin_state='0', rest_nuclear_spin_state='1')``

        the rotating-frame frequencies are computed as follows:

        - for ``'e'``: use the qubit splitting of ``'e'`` conditioned on both
          ``'N'`` and ``'C'`` being in the :math:`|1\\rangle` qubit state;
        - for ``'N'``: use the qubit splitting of ``'N'`` conditioned on ``'e'``
          being in the :math:`|0\\rangle` qubit state and ``'C'`` being in the
          :math:`|1\\rangle` qubit state;
        - for ``'C'``: use the qubit splitting of ``'C'`` conditioned on ``'e'``
          being in the :math:`|0\\rangle` qubit state and ``'N'`` being in the
          :math:`|1\\rangle` qubit state.

        3. With spin-specific nuclear settings,

        ``RotatingFrameSetter(rest_electron_spin_state='1', rest_nuclear_spin_state={'N': '0', 'C': 'avg'})``

        the rotating-frame frequencies are computed as follows:

        - for ``'e'``: use the qubit splitting of ``'e'`` conditioned on ``'N'`` being
          in the :math:`|0\\rangle` qubit state and averaged over ``'C'``;
        - for ``'N'``: use the qubit splitting of ``'N'`` conditioned on ``'e'`` being
          in the :math:`|1\\rangle` qubit state and averaged over ``'C'``;
        - for ``'C'``: use the qubit splitting of ``'C'`` conditioned on ``'e'`` being
          in the :math:`|1\\rangle` qubit state and ``'N'`` being in the
          :math:`|0\\rangle` qubit state.

    """

    _component_change_kind = 'rotating_frame_changed'
    _ALLOWED_STATES = {'0', '1', 'avg'}

    def __init__(self,
                 rest_electron_spin_state: Union[str, Dict[str, str]] = '1',
                 rest_nuclear_spin_state: Union[str, Dict[str, str]] = 'avg'):
        """Initialize the rotating-frame policy.

        Args:
            rest_electron_spin_state: Conditioning state specification for all electron
                spins other than the target spin. A single label is broadcast to
                every electron spin; a dictionary assigns labels explicitly by
                spin name.
            rest_nuclear_spin_state: Conditioning state specification for all nuclear
                spins other than the target spin. A single label is broadcast to
                every nuclear spin; a dictionary assigns labels explicitly by
                spin name.
        """
        super().__init__()
        self._rest_electron_spin_state: Union[str, Dict[str, str]] = '1'
        self._rest_nuclear_spin_state: Union[str, Dict[str, str]] = 'avg'
        self.rest_electron_spin_state = rest_electron_spin_state
        self.rest_nuclear_spin_state = rest_nuclear_spin_state

    @classmethod
    def _validate_state_spec(cls,
                             spec: Union[str, Dict[str, str]],
                             label: str) -> Union[str, Dict[str, str]]:
        if isinstance(spec, str):
            if spec not in cls._ALLOWED_STATES:
                raise SimphonyError(
                    f"{label} must be one of {sorted(cls._ALLOWED_STATES)} or a dictionary with that content."
                )
            return spec

        if isinstance(spec, dict):
            mapping = {}
            for spin_name, state in spec.items():
                if not isinstance(spin_name, str):
                    raise SimphonyError(f"{label} dictionary keys must be spin names (strings).")
                if state not in cls._ALLOWED_STATES:
                    raise SimphonyError(
                        f"{label} contains invalid value '{state}'. Allowed values are {sorted(cls._ALLOWED_STATES)}."
                    )
                mapping[spin_name] = state
            return mapping

        raise SimphonyError(
            f"{label} must be a string or dictionary of strings drawn from {sorted(cls._ALLOWED_STATES)}."
        )

    @staticmethod
    def _broadcast_states(spins,
                          spec: Union[str, Dict[str, str]],
                          label: str) -> dict:
        if isinstance(spec, str):
            return {spin: spec for spin in spins}

        expected_names = {spin.name for spin in spins}
        actual_names = set(spec.keys())
        if actual_names != expected_names:
            spin_kind = label.removeprefix('rest_').split('_')[0]
            raise SimphonyError(
                f"{label} dictionary keys must match the {spin_kind} spin names exactly: "
                f"{sorted(expected_names)}."
            )

        return {spin: spec[spin.name] for spin in spins}

    @staticmethod
    def _state_to_quantum_number(spin,
                                 state_label: str) -> Optional[float]:
        if state_label == 'avg':
            return None
        return spin.qubit_subspace[int(state_label)]

    @property
    def rest_electron_spin_state(self) -> Union[str, Dict[str, str]]:
        """Conditioning state specification used for electron spins other than the target spin."""
        return self._rest_electron_spin_state

    @rest_electron_spin_state.setter
    def rest_electron_spin_state(self, value: Union[str, Dict[str, str]]) -> None:
        self._rest_electron_spin_state = self._validate_state_spec(value, 'rest_electron_spin_state')
        self._notify_models()

    @property
    def rest_nuclear_spin_state(self) -> Union[str, Dict[str, str]]:
        """Conditioning state specification used for nuclear spins other than the target spin."""
        return self._rest_nuclear_spin_state

    @rest_nuclear_spin_state.setter
    def rest_nuclear_spin_state(self, value: Union[str, Dict[str, str]]) -> None:
        self._rest_nuclear_spin_state = self._validate_state_spec(value, 'rest_nuclear_spin_state')
        self._notify_models()

    def apply(self, model) -> None:
        """Recompute and assign per-spin rotating-frame frequencies.

        Args:
            model: Model whose spins should receive updated
                ``rotating_frame_frequency`` values.

        Raises:
            SimphonyError: If the model does not contain any electron spins, or if
                the configured state specifications do not match the model's spin
                content.
        """
        electron_spins = [spin for spin in model.spins if isinstance(spin, ElectronSpin)]
        if not electron_spins:
            raise SimphonyError('RotatingFrameSetter requires at least one ElectronSpin in the model.')

        nuclear_spins = [spin for spin in model.spins if isinstance(spin, NuclearSpin)]
        electron_states = self._broadcast_states(
            electron_spins,
            self.rest_electron_spin_state,
            'rest_electron_spin_state',
        )
        nuclear_states = self._broadcast_states(
            nuclear_spins,
            self.rest_nuclear_spin_state,
            'rest_nuclear_spin_state',
        )

        for target_spin in model.spins:
            rest_quantum_nums = {}
            for other_spin in model.spins:
                if other_spin is target_spin:
                    continue
                if isinstance(other_spin, ElectronSpin):
                    state_label = electron_states[other_spin]
                elif isinstance(other_spin, NuclearSpin):
                    state_label = nuclear_states.get(other_spin, 'avg')
                else:
                    state_label = 'avg'

                quantum_number = self._state_to_quantum_number(other_spin, state_label)
                if quantum_number is not None:
                    rest_quantum_nums[other_spin.name] = quantum_number

            target_spin.rotating_frame_frequency = model.splitting_qubit(
                spin_name=target_spin.name,
                rest_quantum_nums=rest_quantum_nums,
            )

    def __repr__(self) -> str:
        return (
            f'{type(self).__name__}('
            f'rest_electron_spin_state={self.rest_electron_spin_state!r}, '
            f'rest_nuclear_spin_state={self.rest_nuclear_spin_state!r})'
        )

class Model:
    r"""Represent a coupled spin system.

    This class provides a high-level abstraction for simulating quantum systems consisting of multiple coupled spins.
    It supports adding spins, interactions, static and driving magnetic fields. The model allows calculation of energy
    levels and eigenstates. In addition, it provides methods for computing parameters required to define control pulses
    (e.g. splittings, Rabi-frequencies, ...). It enables the simulation of the system's time evolution under the
    influence of driving fields and additional noise. It is especially suitable for modeling systems like
    nitrogen-vacancy centers.

    The model owns the lifecycle of attached components. Spins, interactions, static fields, and driving fields should
    be attached using the corresponding :meth:`add_spin`, :meth:`add_interaction`, :meth:`add_static_field`, and
    :meth:`add_driving_field` methods, and removed using the matching :meth:`remove_...` methods. After a component is
    attached, updating one of its physically relevant attributes automatically affects future accesses to derived model
    quantities such as the static Hamiltonian, eigensystem, or driving operators.

    Rotating-frame frequencies, virtual phases, and pulse lists remain directly editable runtime settings unless a
    model-level rotating-frame setter is attached.

    Initialize a model.

    Args:
        name: Name of the model.

    Hint:
        For example, to initialize a model for a nitrogen-vacancy center including the intrinsic nitrogen nuclear
        spin, one can use:

        .. code-block:: python

            model = simphony.default_nv_model(nitrogen_isotope=14)

    """

    def __init__(self,
                 name: str = None):

        self.name: Optional[str] = name
        """Name of the model."""

        self.warn_on_ambiguous_labels: bool = True
        """Whether to warn about ambiguous eigenstate labels."""

        self.spins: List[BaseSpin] = []
        """Spins attached to the model."""

        self.interactions: List[Interaction] = []
        """Interactions attached to the model."""

        self.static_fields: List[StaticField] = []
        """Static fields attached to the model."""

        self.driving_fields: List[BaseDrivingField] = []
        """Driving fields attached to the model."""

        self.initial_state = None

        self.initial_density_matrix = None

        self.rotating_frame_hamiltonian = 'local_qubit_z'

        self._rotating_frame: Optional[RotatingFrameSetter] = None
        """Optional model-managed rotating-frame policy."""

        self._static_hamiltonian: Optional[uarray] = None
        """Time-independent part of the Hamiltonian."""

        self._driving_operators: Dict[str, List[uarray]] = {}
        """Driving operator components keyed by driving-field name."""

        self._local_quasistatic_noise_operators: List[uarray] = []
        """Local quasistatic-noise operators."""

        self._basis: Optional[list] = None
        """Basis of the Hamiltonian."""

        self._basis_qubit_subspace: Optional[list] = None
        """Basis corresponding to the qubit subspace."""

        self._eigenbasis: Optional[uarray] = None
        """Eigenbasis of the static Hamiltonian."""

        self._eigenenergies: Optional[List[float]] = None
        """Eigenenergies of the static Hamiltonian."""

        self._map_quantum_nums_to_idx: Dict[Tuple[float, ...], int] = {}
        """Mapping from product-basis quantum numbers to basis indices."""

        self._quantization_rotation_operator: Optional[uarray] = None
        self._quantization_rotation_operator_dagger: Optional[uarray] = None

        self._build_flags: Dict[str, bool] = {target: False for target in _BUILD_DEPENDENCIES}
        """Internal cache state for derived full-Hilbert-space artifacts."""

        self._pending_component_changes: Set[str] = set()
        """Pending coarse-grained component lifecycle events."""


    def _validate_build_target(self, target: str) -> None:
        if target not in _BUILD_DEPENDENCIES:
            raise SimphonyError(f"Unknown build target '{target}'.")

    def _clear_build_artifact(self, target: str) -> None:
        if target == 'quantization_transform':
            self._quantization_rotation_operator = None
            self._quantization_rotation_operator_dagger = None
        elif target == 'static_hamiltonian':
            self._static_hamiltonian = None
            self._basis = None
            self._basis_qubit_subspace = None
            self._map_quantum_nums_to_idx = {}
        elif target == 'eigensystem':
            self._eigenenergies = None
            self._eigenbasis = None
        elif target == 'rotating_frame':
            pass
        elif target == 'driving_operators':
            self._driving_operators = {}
        elif target == 'noise_operators':
            self._local_quasistatic_noise_operators = []

    def _unbuild_target(self, target: str, visited: Set[str]) -> None:
        if target in visited:
            return
        visited.add(target)
        self._clear_build_artifact(target)
        self._build_flags[target] = False

        for dependent, dependencies in _BUILD_DEPENDENCIES.items():
            if target in dependencies:
                self._unbuild_target(dependent, visited)

    def _unbuild(self, *targets: str) -> None:
        visited: Set[str] = set()
        for target in targets:
            self._validate_build_target(target)
            self._unbuild_target(target, visited)

    def _mark_component_dirty(self, component, change_kind: str) -> None:
        self._pending_component_changes.add(change_kind)
        if self.rotating_frame is not None and change_kind in {
            'spin_changed',
            'static_field_changed',
            'interaction_changed',
            'rotating_frame_changed',
        }:
            self._autosync_rotating_frame(force=True)

    def _sync_component_changes(self) -> None:
        if not self._pending_component_changes:
            return

        pending = set(self._pending_component_changes)
        self._pending_component_changes.clear()

        if 'spin_changed' in pending or 'static_field_changed' in pending:
            self._unbuild('quantization_transform')

        if 'interaction_changed' in pending:
            self._unbuild('static_hamiltonian')

        if 'driving_field_changed' in pending:
            self._unbuild('driving_operators')

        if 'rotating_frame_changed' in pending:
            self._unbuild('rotating_frame')

    def _build_target(self, target: str, force: bool = False) -> None:
        self._validate_build_target(target)
        for dependency in _BUILD_DEPENDENCIES[target]:
            self._build_target(dependency, force=force)

        if force or not self._build_flags[target]:
            getattr(self, f'_build_{target}')()
            self._build_flags[target] = True

    def _build(self, *targets: str, force: bool = False) -> None:
        self._sync_component_changes()
        for target in targets:
            self._build_target(target, force=force)

    def _build_rotating_frame(self) -> None:
        if self.rotating_frame is None:
            return
        self.rotating_frame.apply(self)

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'(spin_names={self.spin_names}, '
            f'num_static_fields={len(self.static_fields)}, '
            f'driving_field_names={self.driving_field_names}, '
            f'dimension={self.dimension})'
        )

    @property
    def rotating_frame(self) -> Optional[RotatingFrameSetter]:
        """Optional rotating-frame setter that owns per-spin frame frequencies."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, value: Optional[RotatingFrameSetter]) -> None:
        if value is not None and not isinstance(value, RotatingFrameSetter):
            raise SimphonyError('rotating_frame must be a RotatingFrameSetter or None.')

        current = getattr(self, '_rotating_frame', None)
        if current is value:
            return

        if current is not None:
            current._detach_model(self)

        self._rotating_frame = value

        if value is not None:
            value._attach_model(self)

        self._unbuild('rotating_frame')

    @property
    def n_spins(self) -> int:
        """Number of the spins."""
        return len(self.spins)

    @property
    def subdimensions(self) -> List[int]:
        """Dimensions of the spins."""
        return [spin.dimension for spin in self.spins]

    @property
    def dimension(self) -> int:
        """Total dimension of the spin model."""
        if not self.spins:
            return 0
        return math.prod(self.subdimensions)

    @property
    def spin_names(self) -> List[str]:
        """Names of the spins."""
        return [spin.name for spin in self.spins]

    @property
    def driving_field_names(self) -> List[str]:
        """Names of the driving fields."""
        return [spin.name for spin in self.driving_fields]

    @property
    def driving_operators(self) -> Dict[str, List[uarray]]:
        r"""Driving operators are the time-independent operator parts of the driving
        Hamiltonian, whereas pulses are the time-dependent scalar parts:

        .. math::

            H_\text{driving} = \sum_{k\in\text{driving fields}}
            \underbrace{B_{k}(t)}_{\text{pulse}}
            \underbrace{\left[
            \sum_{i\in\text{spins}}
            \gamma_i\hat{\mathbf{u}}_{k}
            \cdot
            \mathbf{S}_{i}
            \right]}_{\text{driving operator}},

        where :math:`B_{k}(t)` and :math:`\hat{\mathbf{u}}_k` are the strength and
        direction vector of the :math:`k\text{th}` driving field.

        """
        self._build('driving_operators')
        return self._driving_operators

    @property
    def static_hamiltonian(self) -> Optional[uarray]:
        r"""Static part of the Hamiltonian in the model's full Hilbert space.

        It contains the anisotropy, spin-spin interaction, and static Zeeman terms:

        .. math::

            H_\text{static} = \sum_i \boldsymbol{S}_i^\mathsf{T} \boldsymbol{D}_i \boldsymbol{S}_i
            + \sum_{i < j} \boldsymbol{S}_i^\mathsf{T} \boldsymbol{A}_{ij} \boldsymbol{S}_j
            + \boldsymbol{B}_\text{static} \left[\sum_{i \in \text{electron spins}} \gamma_i \boldsymbol{S}_i
            - \sum_{i \in \text{nuclear spins}} \gamma_i \boldsymbol{S}_i\right]

        """
        self._build('static_hamiltonian')
        return self._static_hamiltonian

    @property
    def local_quasistatic_noise_operators(self) -> List[uarray]:
        r"""These operators represent the time-independent part of the noise Hamiltonian,
        while their strengths represent the shot-dependent scalar parts:

        .. math::

            H_\text{noise} = \sum_{i\in\text{spins}}
            \overbrace{\Delta\omega_{i}}^{\rlap{\text{local quasistatic noise strength}}}
            \underbrace{\hat{\mathbf{v}}_{i}
            \cdot
            \mathbf{S}_{i}}_{\rlap{\text{local quasistatic noise operator}}},

        where :math:`\Delta\omega_{i}` and :math:`\hat{\mathbf{v}}_{i}` are the strength
        and direction vector associated with the local quasistatic noise on the
        :math:`i\text{th}` spin.

        """
        self._build('noise_operators')
        return self._local_quasistatic_noise_operators

    @property
    def basis(self) -> Optional[list]:
        """Product-basis labels of the full Hilbert space.

        Each entry is a tuple of spin quantum numbers ordered according to
        :attr:`spin_names`. The basis is used to label eigenstates by a one-to-one
        maximal-overlap assignment with local-:math:`S_z` product-basis states.

        """
        self._build('static_hamiltonian')
        return self._basis

    @property
    def basis_qubit_subspace(self) -> Optional[list]:
        """Product-basis labels restricted to the qubit subspace of each spin.

        Each tuple is ordered according to :attr:`spin_names` and contains only the two
        qubit-subspace quantum numbers defined on the corresponding spins.

        """
        self._build('static_hamiltonian')
        return self._basis_qubit_subspace

    @property
    def eigenbasis(self) -> Optional[uarray]:
        """Eigenstates of :attr:`static_hamiltonian` expressed in the product basis.

        The returned array contains one eigenstate per row, written in the
        product basis defined by :attr:`basis`. The row ordering is not the raw
        order returned by the eigensolver. Instead, Simphony relabels the
        eigensystem with a one-to-one assignment that maximizes the total overlap
        between eigenstates and local-:math:`S_z` product-basis states. This makes the eigenbasis
        easier to interpret in common weak-mixing situations, where each eigenstate
        is still dominated by a single product-basis component.

        In strongly mixed, degenerate, or nearly degenerate cases, the assigned
        local-:math:`S_z` product-basis labels should be understood as a practical naming
        convention rather than as an exact statement about state identity. Use
        :meth:`test_labeling` to inspect the overlap structure explicitly when
        label interpretation matters.

        Returns:
            Array of shape ``(dimension, dimension)`` whose ``i``-th row is the
            eigenstate currently labelled by ``basis[i]``.

        """
        self._build('eigensystem')
        return self._eigenbasis

    @property
    def eigenenergies(self) -> Optional[List[float]]:
        """Eigenenergies of :attr:`static_hamiltonian` labeled by :attr:`basis`.

        The energies follow the same one-to-one labeling convention as
        :attr:`eigenbasis`.

        """
        self._build('eigensystem')
        return self._eigenenergies

    @property
    def rotating_frame_frequencies(self) -> uarray:
        """Per-spin rotating-frame frequencies used by ``'local_qubit_z'``.

        The returned array follows :attr:`spin_names` and is used when constructing
        :meth:`rotating_frame_operator` from the per-spin qubit-frame frequencies.

        """
        if self.rotating_frame is not None:
            self._build('rotating_frame')
        return unp_dynamic.array([spin.rotating_frame_frequency for spin in self.spins])

    @property
    def virtual_phases(self)  -> uarray:
        """Per-spin virtual z-phases.

        The returned array follows :attr:`spin_names` and is used by
        :meth:`virtual_phases_operator`.

        """
        return unp_dynamic.array([spin.virtual_phase for spin in self.spins])

    @property
    def initial_state(self) -> uarray:
        """Initial state used when ``state_only=True`` simulations are requested.

        The value should be a state vector in the model's full Hilbert space. When left
        as ``None``, :meth:`simulate_time_evolution` requires ``state_only=False``.

        Hint:
            Set this when you want Simphony to propagate a specific state vector
            instead of full propagators. It is mutually exclusive with
            :attr:`initial_density_matrix`.

        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        self._initial_state = None if initial_state is None else unp_dynamic.asarray(initial_state)
        if self._initial_state is not None:
            self._initial_density_matrix = None

    @property
    def initial_density_matrix(self) -> Optional[uarray]:
        """Initial density matrix used to derive time-evolved density matrices from propagators."""
        return self._initial_density_matrix

    @initial_density_matrix.setter
    def initial_density_matrix(self, initial_density_matrix):
        self._initial_density_matrix = None if initial_density_matrix is None else unp_dynamic.asarray(initial_density_matrix)
        if self._initial_density_matrix is not None:
            self._initial_state = None

    @property
    def rotating_frame_hamiltonian(self) -> Union[str, uarray]:
        """Hamiltonian used to construct the rotating-frame operator.

        Supported values are:

        - ``'local_qubit_z'``: use the effective Hamiltonian defined by the per-spin rotating-frame frequencies.
        - ``'static_hamiltonian'``: use :attr:`static_hamiltonian` directly.
        - any square matrix: interpret that matrix as a rotating-frame Hamiltonian :math:`H_\\mathrm{rf}`.
        """
        return self._rotating_frame_hamiltonian

    @rotating_frame_hamiltonian.setter
    def rotating_frame_hamiltonian(self, value: Union[str, uarray]) -> None:
        if isinstance(value, str):
            if value not in ['local_qubit_z', 'static_hamiltonian']:
                raise SimphonyError(
                    "rotating_frame_hamiltonian must be 'local_qubit_z', 'static_hamiltonian', or a square matrix."
                )
            self._rotating_frame_hamiltonian = value
            return

        value = unp_dynamic.asarray(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise SimphonyError('rotating_frame_hamiltonian matrix must be square.')
        self._rotating_frame_hamiltonian = value

    @property
    def last_pulse_end(self) -> float:
        """Latest pulse end time across all driving fields.

        Returns:
            End time of the latest registered pulse in microseconds.

        Raises:
            SimphonyError: If no driving field has been added yet.

        """
        if not self.driving_fields:
            raise SimphonyError('No driving field added.')

        return unp_dynamic.max(
            unp_dynamic.array([driving_field.last_pulse_end for driving_field in self.driving_fields])
        )

    @property
    def pulses(self) -> PulseList:
        """Concatenated pulse list from all driving fields.

        Returns:
            PulseList containing the pulses of every driving field in the model.

        """
        pulses = PulseList()
        for driving_field in self.driving_fields:
            pulses += driving_field.pulses
        return pulses

    def add_spin(self,
                 spin: BaseSpin):
        """Add a spin to the model.

        Args:
            spin: Spin object to attach to the model.

        Raises:
            SimphonyError: If another spin with the same name is already present.

        Note:
            Adding a spin changes the size and structure of the model, so derived quantities
            such as the Hamiltonian, eigensystem, and driving operators are recomputed when
            needed.

        """

        if spin.name in self.spin_names:
            raise SimphonyError(f"A spin with name '{spin.name}' is already added to the model")

        self.spins += [spin]
        spin._attach_model(self)
        self._unbuild('quantization_transform')
        self._autosync_rotating_frame(force=True)

    def add_interaction(self,
                        interaction: Interaction):
        """Add a spin-spin interaction to the model.

        Args:
            interaction: Interaction object to attach to the model.

        Note:
            Adding an interaction changes the model Hamiltonian, so derived quantities such
            as the eigensystem are recomputed when needed.

        """

        if interaction.spin1 not in self.spins:
            raise SimphonyError(f"Spin '{interaction.spin1.name}' of the interaction is not added to the model.")
        if interaction.spin2 not in self.spins:
            raise SimphonyError(f"Spin '{interaction.spin2.name}' of the interaction is not added to the model.")
        if interaction.spin1 == interaction.spin2:
            raise SimphonyError("A spin cannot interact with itself.")

        self.interactions += [interaction]
        interaction._attach_model(self)
        self._unbuild('static_hamiltonian')
        self._autosync_rotating_frame(force=True)

    def add_static_field(self,
                         static_field: StaticField,
                         target_spins: Optional[Union[str, Sequence[str]]] = None):
        """Add a static field to the model.

        Args:
            static_field: Static field instance to attach.
            target_spins: Optional override of the spins affected by the field. ``None`` keeps or
                sets the field to act on all spins. Providing a value replaces any existing
                target list and emits a warning when it changes.

        Raises:
            SimphonyError: If the provided targets reference spins that are not part of the model.

        """

        if target_spins is not None:
            parsed = static_field._parse_target_spins(target_spins)
            if static_field.target_spins is not None and static_field.target_spins != parsed:
                warn(
                    'Overriding target spins for static field: '
                    f'{static_field.target_spins} -> {parsed}',
                    SimphonyWarning,
                    stacklevel=1,
                )
            static_field.target_spins = parsed

        self._validate_field_targets(static_field.target_spins, 'static field')

        self.static_fields.append(static_field)
        static_field._attach_model(self)
        self._unbuild('quantization_transform')
        self._autosync_rotating_frame(force=True)

    def add_driving_field(self,
                          driving_field: BaseDrivingField,
                          target_spins: Optional[Union[str, Sequence[str]]] = None):
        """Add a driving field to the model.

        Args:
            driving_field: Driving field instance to attach.
            target_spins: Optional override of the spins driven by the field. ``None`` keeps or
                sets the field to act on all spins. Providing a value replaces any existing
                target list and emits a warning when it changes.

        Raises:
            SimphonyError: If a driving field with the same name is already registered.
            SimphonyError: If the (possibly overridden) targets reference spins that are not
                part of the model.

        """

        if driving_field.name in self.driving_field_names:
            raise SimphonyError(f"A driving field with name '{driving_field.name}' is already added to the model")

        if target_spins is not None:
            parsed = driving_field._parse_target_spins(target_spins)
            if driving_field.target_spins is not None and driving_field.target_spins != parsed:
                warn(
                    f"Overriding target spins for driving field '{driving_field.name}': "
                    f'{driving_field.target_spins} -> {parsed}',
                    SimphonyWarning,
                    stacklevel=1,
                )
            driving_field.target_spins = parsed

        self._validate_field_targets(driving_field.target_spins,
                                     f"driving field '{driving_field.name}'")

        self.driving_fields += [driving_field]
        driving_field._attach_model(self)
        self._unbuild('driving_operators')

    def remove_spin(self,
                    spin: Union[BaseSpin, str],
                    include_interactions: bool = False) -> None:
        """Remove a spin from the model.

        Args:
            spin: Spin instance or spin name.
            include_interactions: Whether interactions involving the spin should be removed automatically.

        Raises:
            SimphonyError: If the spin is not part of the model or other attached components
                still reference it.

        Note:
            Removing a spin is conservative by default. Interactions referencing the spin must be
            removed explicitly first, unless ``include_interactions=True`` is used. Targeted static
            fields and driving fields referencing the spin must always be removed explicitly before
            the spin can be removed.

        """
        spin_obj = self.spin(spin) if isinstance(spin, str) else spin
        if spin_obj not in self.spins:
            raise SimphonyError('The provided spin is not attached to the model.')

        spin_name = spin_obj.name
        dependent_interactions = [
            interaction for interaction in self.interactions
            if interaction.spin1 is spin_obj or interaction.spin2 is spin_obj
        ]
        if dependent_interactions and not include_interactions:
            labels = [(interaction.spin1.name, interaction.spin2.name) for interaction in dependent_interactions]
            raise SimphonyError(
                f"Cannot remove spin '{spin_name}' while interactions still reference it: {labels}"
            )

        dependent_static_fields = [
            static_field for static_field in self.static_fields
            if static_field.target_spins is not None and spin_name in static_field.target_spins
        ]
        if dependent_static_fields:
            raise SimphonyError(
                f"Cannot remove spin '{spin_name}' while targeted static fields still reference it."
            )

        dependent_driving_fields = [
            driving_field for driving_field in self.driving_fields
            if driving_field.target_spins is not None and spin_name in driving_field.target_spins
        ]
        if dependent_driving_fields:
            names = [driving_field.name for driving_field in dependent_driving_fields]
            raise SimphonyError(
                f"Cannot remove spin '{spin_name}' while targeted driving fields still reference it: {names}"
            )

        for interaction in dependent_interactions:
            self.interactions.remove(interaction)
            interaction._detach_model(self)

        self.spins.remove(spin_obj)
        spin_obj._detach_model(self)
        self._unbuild('quantization_transform')
        self._autosync_rotating_frame(force=True)

    def remove_interaction(self,
                           interaction: Interaction) -> None:
        """Remove an interaction from the model.

        Args:
            interaction: Interaction object to remove.

        Raises:
            SimphonyError: If the interaction is not attached to the model.
        """
        if interaction not in self.interactions:
            raise SimphonyError('The provided interaction is not attached to the model.')

        self.interactions.remove(interaction)
        interaction._detach_model(self)
        self._unbuild('static_hamiltonian')
        self._autosync_rotating_frame(force=True)

    def remove_static_field(self,
                            static_field: StaticField) -> None:
        """Remove a static field from the model.

        Args:
            static_field: Static field object to remove.

        Raises:
            SimphonyError: If the static field is not attached to the model.
        """
        if static_field not in self.static_fields:
            raise SimphonyError('The provided static field is not attached to the model.')

        self.static_fields.remove(static_field)
        static_field._detach_model(self)
        self._unbuild('quantization_transform')
        self._autosync_rotating_frame(force=True)

    def remove_driving_field(self,
                             driving_field: Union[BaseDrivingField, str]) -> None:
        """Remove a driving field from the model.

        Args:
            driving_field: Driving field object or driving-field name.

        Raises:
            SimphonyError: If the driving field is not attached to the model.
        """
        driving_field_obj = self.driving_field(driving_field) if isinstance(driving_field, str) else driving_field
        if driving_field_obj not in self.driving_fields:
            raise SimphonyError('The provided driving field is not attached to the model.')

        self.driving_fields.remove(driving_field_obj)
        driving_field_obj._detach_model(self)
        self._unbuild('driving_operators')

    def remove_all_pulses(self):
        """Remove all pulses from every driving field and reset virtual z-phases.

        This leaves the registered driving fields in place but clears their pulse lists.
        The per-spin virtual phases are reset to zero as well.

        """

        for driving_field in self.driving_fields:
            driving_field.remove_all_pulses()

        for spin in self.spins:
            spin.virtual_phase = 0.

    def sync_rotating_frame(self, force: bool = False) -> None:
        """Recompute model-managed rotating-frame frequencies.

        When :attr:`rotating_frame` is ``None``, the model stays in manual mode and
        this method does nothing. Otherwise the attached rotating-frame setter
        overwrites each spin's ``rotating_frame_frequency`` in place.

        Args:
            force: If ``True``, recompute the cached rotating-frame values even if the
                model currently considers them up to date.
        """

        if self.rotating_frame is None:
            return

        self._build('rotating_frame', force=force)

    def _autosync_rotating_frame(self, force: bool = False) -> None:
        """Best-effort rotating-frame sync for lifecycle hooks."""

        if self.rotating_frame is None:
            return

        if not any(isinstance(spin, ElectronSpin) for spin in self.spins):
            return

        self.sync_rotating_frame(force=force)

    def spin(self,
             name: str) -> BaseSpin:
        """Return the instance of the spin specified by ``name``.

        Args:
            name: Name of the spin.

        Returns:
            Instance of the specified spin.

        Raises:
            SimphonyError: If the given name is not found in :attr:`spin_names`.
        """

        if name not in self.spin_names:
            raise SimphonyError(f'Invalid name, must be in {self.spin_names}')

        idx = self.spin_names.index(name)

        return self.spins[idx]

    def project_to_qubit_subspace(self,
                                  unitary: uarray,
                                  only_diagonal: bool = False) -> uarray:
        """Project an operator or batch of operators into the qubit subspace.

        The projection uses the cached qubit-subspace basis indices derived from the
        model's static Hamiltonian. For batched inputs, the last one or two axes are
        interpreted as Hilbert-space axes.

        Args:
            unitary: Operator array in the full Hilbert space.
            only_diagonal: If ``True``, treat the last axis as a diagonal only and return
                the projected diagonal entries without adding a second matrix axis.

        Returns:
            Operator represented in the tensor-product qubit subspace.

        """

        self._build('static_hamiltonian')
        qubit_subspace_idxs = unp_dynamic.array([self._basis.index(i) for i in self._basis_qubit_subspace])
        projected_unitary = unitary[..., qubit_subspace_idxs]
        if not only_diagonal:
            projected_unitary = projected_unitary[..., qubit_subspace_idxs, :]

        return projected_unitary

    def _spin_idx_from_name(self,
                            spin_names: Union[str, List[str]]) -> List[int]:
        """Return the spin indices from name or names."""
        if isinstance(spin_names, str):
            spin_names = [spin_names]
        if len(set(spin_names)) != len(spin_names):
            raise SimphonyError('spin_names contains duplications.')
        if not set(spin_names).issubset(set(self.spin_names)):
            raise SimphonyError(f'Invalid spin_names, must be a subset of {self.spin_names}.')

        return [self.spin_names.index(spin_name) for spin_name in spin_names]

    def _field_target_indices(self,
                              target_spins: Optional[Tuple[str, ...]]) -> List[int]:
        """Return indices of spins affected by a field based on target names."""

        if target_spins is None:
            return list(range(self.n_spins))

        missing = set(target_spins) - set(self.spin_names)
        if missing:
            raise SimphonyError(f'Invalid spin name in field targets: {sorted(missing)}')

        return [self.spin_names.index(name) for name in target_spins]

    def _validate_field_targets(self,
                                target_spins: Optional[Tuple[str, ...]],
                                field_label: str) -> None:
        """Validate that the provided field targets refer to known spins when possible."""

        if target_spins is None or self.n_spins == 0:
            return

        missing = set(target_spins) - set(self.spin_names)
        if missing:
            raise SimphonyError(
                f"{field_label} references unknown spins: {sorted(missing)}"
            )

    def rotating_frame_operator(self,
                                t: Union[float, List[float], uarray],
                                only_diag: bool = False,
                                in_qubit_subspace: bool = False,
                                inverse: bool = False) -> uarray:
        """Return the rotating-frame operator defined by :attr:`rotating_frame_hamiltonian`.

        By default (``rotating_frame_hamiltonian='local_qubit_z'``),

        .. math::
            U_\\text{rot}(t) = \\bigotimes_{i\\in\\text{spins}} e^{i 2 \\pi f_i t \\sigma_{z,i}/2},

        where :math:`f_i` are the rotating frame frequencies of the spins and :math:`\\sigma_{z,i}` are the Pauli
        z-operators. Frequencies are specified in the :attr:`BaseSpin.rotating_frame_frequency` attribute.

        If :attr:`rotating_frame_hamiltonian` is ``'static_hamiltonian'`` or a square matrix, that matrix is interpreted
        as a rotating-frame Hamiltonian :math:`H_\\mathrm{rf}` and the operator is

        .. math::
            U_\\text{rot}(t) = e^{-i 2 \\pi t H_\\mathrm{rf}}.

        Args:
            t: Time point(s).
            only_diag: If ``True``, return only the diagonal of the operator.
            in_qubit_subspace: If ``True``, return the operator represented in the qubit subspace.
            inverse: If ``True``, return the inverse of the operator.

        Returns:
            Rotating-frame operator(s). Shape depends on the ``only_diag`` parameter and the type of ``t``.

        """

        if inverse:
            sign = -1
        else:
            sign = 1

        h_rot = self._get_rotating_frame_hamiltonian(in_qubit_subspace=in_qubit_subspace)

        if isinstance(self.rotating_frame_hamiltonian, str) and self.rotating_frame_hamiltonian == 'local_qubit_z':
            if only_diag:
                frame_operator_fn = _frame_operator_diag
            else:
                frame_operator_fn = _frame_operator

            omegas = 2 * unp_dynamic.pi * self.rotating_frame_frequencies

            if in_qubit_subspace:
                def operator_(t1):
                    return frame_operator_fn(sign * omegas * t1)
            else:
                sigma_zs = [spin.operator_qubit_subspace.z for spin in self.spins]

                def operator_(t1):
                    return frame_operator_fn(sign * omegas * t1, sigma_zs)
        else:
            def operator_(t1):
                operator = expm(-1j * sign * 2 * unp_dynamic.pi * t1 * h_rot)
                if only_diag:
                    return unp_dynamic.diag(operator)
                return operator

        t_array = unp_dynamic.asarray(t)
        if t_array.shape == ():
            return operator_(t_array.item())
        return unp_dynamic.array([operator_(t1) for t1 in t_array])

    def _get_rotating_frame_hamiltonian(self,
                                        in_qubit_subspace: bool = False) -> Optional[uarray]:
        """Return the resolved rotating-frame Hamiltonian.

        For ``rotating_frame_hamiltonian='local_qubit_z'`` this function returns ``None``, since that case is handled
        by a dedicated diagonal fast path in :meth:`rotating_frame_operator`.

        """

        h_rot = self.rotating_frame_hamiltonian

        if isinstance(h_rot, str):
            if h_rot == 'local_qubit_z':
                return None
            if h_rot != 'static_hamiltonian':
                raise SimphonyError(f'Unsupported rotating_frame_hamiltonian: {h_rot}')
            self._build('static_hamiltonian')
            h_rot = self._static_hamiltonian

        h_rot = unp_dynamic.asarray(h_rot)
        if h_rot.ndim != 2 or h_rot.shape[0] != h_rot.shape[1]:
            raise SimphonyError('rotating_frame_hamiltonian matrix must be square.')

        if in_qubit_subspace:
            h_rot = self.project_to_qubit_subspace(h_rot)

        return h_rot

    def virtual_phases_operator(self,
                                only_diag: bool = False,
                                in_qubit_subspace: bool = False,
                                inverse: bool = False) -> uarray:
        """Return the operator corresponding to the virtual z-phases:

        .. math::
            U_\\text{z-phase} = \\bigotimes_{i\\in\\text{spins}} e^{i \\phi_i \\sigma_{z,i}/2},

        where :math:`\\phi_i` are the virtual z-phases of the spins and :math:`\\sigma_{z,i}` are the Pauli
        z-operators. Phases are specified in the :attr:`BaseSpin.virtual_phase` attribute.

        Args:
            only_diag: If ``True``, return only the diagonal of the operator.
            in_qubit_subspace: If ``True``, return the operator represented in the qubit subspace.
            inverse: If ``True``, return the inverse of the operator.

        Returns:
            Operator corresponding to the virtual z-phases. Shape depends on the ``only_diag`` parameter.

        """

        if inverse:
            sign = -1
        else:
            sign = 1

        if only_diag:
            frame_operator_fn = _frame_operator_diag
        else:
            frame_operator_fn = _frame_operator

        if in_qubit_subspace:
            return frame_operator_fn(sign * self.virtual_phases)
        else:
            sigma_zs = [spin.operator_qubit_subspace.z for spin in self.spins]
            return frame_operator_fn(sign * self.virtual_phases, sigma_zs)

    def driving_field(self,
                      name: str) -> BaseDrivingField:
        """Return the instance of the driving field specified by ``name``.

        Args:
            name: Name of the driving field.

        Returns:
            Instance of the specified driving field.

        Raises:
            SimphonyError: If the given name is not found in :attr:`driving_fields`.

        """

        if name not in self.driving_field_names:
            raise SimphonyError(f'Invalid name, must be in {self.driving_field_names}')

        idx = self.driving_field_names.index(name)

        return self.driving_fields[idx]

    def _quantum_nums_from_dict(self,
                                quantum_nums: dict) -> tuple:
        """Return the quantum numbers in the proper order from a ``dict``."""

        if len(quantum_nums.keys()) != self.n_spins:
            raise SimphonyError(f'Expected {self.n_spins} quantum numbers, but got {len(quantum_nums.keys())}.')

        for spin_name, quantum_num in quantum_nums.items():
            if spin_name not in self.spin_names:
                raise SimphonyError(f'Invalid spin name: {spin_name}')

            if quantum_num not in self.spin(spin_name).quantum_nums:
                raise SimphonyError(
                    f'Invalid quantum number for {spin_name}: {quantum_num}, must be {self.spin(spin_name).quantum_nums}')

        return tuple([quantum_nums[spin_name] for spin_name in self.spin_names])

    def _quantum_nums_from_bitstring(self,
                                     bitstring: str) -> tuple:
        """Return the quantum numbers corresponding to a qubit-subspace bitstring."""

        if len(bitstring) != self.n_spins:
            raise SimphonyError(f'Expected bitstring of length {self.n_spins}, but got {len(bitstring)}.')

        if any(bit not in {'0', '1'} for bit in bitstring):
            raise SimphonyError("Bitstring quantum numbers must contain only '0' and '1'.")

        return tuple(self.spins[idx].qubit_subspace[int(bit)] for idx, bit in enumerate(bitstring))

    def productstate(self,
                     quantum_nums: Union[str, Dict[str, float], Tuple[float, ...]]) -> uarray:
        """Return the product basis state of the model corresponding to given quantum numbers of the spins.

        Args:
            quantum_nums: Quantum numbers. It can be specified by three different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.
                - By a bitstring such as ``"010"``, interpreted in the :attr:`spin_names` order using each spin's ``qubit_subspace``.

        Returns:
            Product basis state

        """

        self._build('static_hamiltonian')

        if isinstance(quantum_nums, str):
            quantum_nums = self._quantum_nums_from_bitstring(quantum_nums)
        elif isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        return unp_static.eye(self.dimension, dtype=complex)[self._map_quantum_nums_to_idx[quantum_nums]]

    def _build_eigensystem(self):
        """Populate the cached eigensystem of the static Hamiltonian."""

        E0s, v0s = unp_static.linalg.eig(self._static_hamiltonian)
        E0s = E0s.real
        v0s = v0s.transpose()

        overlaps = np.abs(np.asarray(v0s)) ** 2
        product_basis_indices, eigenstate_indices = linear_sum_assignment(-overlaps.T)

        eigenenergies = [0.0] * self.dimension
        eigenstates = [None] * self.dimension

        for product_basis_idx, eigenstate_idx in zip(product_basis_indices, eigenstate_indices):
            eigenenergies[product_basis_idx] = float(E0s[eigenstate_idx])
            eigenstates[product_basis_idx] = v0s[eigenstate_idx]
        eigenstates = unp_static.stack(eigenstates)

        self._eigenenergies = eigenenergies
        self._eigenbasis = eigenstates

        if self.warn_on_ambiguous_labels:
            dominant_product_state_overlap_threshold = 0.9
            ambiguous_eigenstate_labels = self._collect_labeling_overlap_details(
                overlap_threshold=dominant_product_state_overlap_threshold,
                overlaps='relevant',
            )

            if ambiguous_eigenstate_labels:
                details = self._format_labeling_overlap_details_pretty(ambiguous_eigenstate_labels)
                warn(
                    (
                        "Eigenstates are labelled via a one-to-one maximal-overlap assignment to "
                        "local-S_z product-basis states. "
                        f"{len(ambiguous_eigenstate_labels)} assigned labels have overlap below "
                        f"{dominant_product_state_overlap_threshold:.1f}:\n{details}"
                    ),
                    SimphonyWarning,
                    stacklevel=1,
                )

    def _collect_labeling_overlap_details(self,
                                          overlap_threshold: Optional[float] = None,
                                          overlaps: Union[str, int] = 'relevant') -> List[Dict[str, object]]:
        """Return eigenstate-labeling overlaps against local-:math:`S_z` product-basis states."""
        if overlap_threshold is not None and not 0.0 <= overlap_threshold <= 1.0:
            raise SimphonyError('overlap_threshold must be between 0 and 1.')
        if isinstance(overlaps, str):
            if overlaps not in {'all', 'relevant'}:
                raise SimphonyError("overlaps must be 'all', 'relevant', or a non-negative integer.")
        elif not isinstance(overlaps, int) or overlaps < 0:
            raise SimphonyError("overlaps must be 'all', 'relevant', or a non-negative integer.")

        diagnostics = []
        for eigenstate_idx, eigenstate_in_product_basis in enumerate(self._eigenbasis):
            product_basis_overlaps = np.abs(np.asarray(eigenstate_in_product_basis)) ** 2
            diagonal_overlap = float(product_basis_overlaps[eigenstate_idx])
            if overlap_threshold is not None and diagonal_overlap >= overlap_threshold:
                continue

            sorted_overlap_indices = np.argsort(product_basis_overlaps)[::-1]
            if overlaps == 'all':
                selected_overlap_indices = sorted_overlap_indices
            elif overlaps == 'relevant':
                cumulative_overlaps = np.cumsum(product_basis_overlaps[sorted_overlap_indices])
                cutoff_idx = int(np.searchsorted(cumulative_overlaps, 0.99, side='left')) + 1
                selected_overlap_indices = sorted_overlap_indices[:cutoff_idx]
            else:
                selected_overlap_indices = sorted_overlap_indices[:overlaps]

            diagnostics.append(
                {
                    'eigenstate': self._basis[eigenstate_idx],
                    'diagonal_overlap': diagonal_overlap,
                    'overlaps': [
                        {
                            'product_state': self._basis[product_state_idx],
                            'overlap': float(product_basis_overlaps[product_state_idx]),
                        }
                        for product_state_idx in selected_overlap_indices
                    ],
                }
            )

        diagnostics.sort(key=lambda item: item['diagonal_overlap'])
        return diagnostics

    def _plot_labeling_overlap_matrix(self,
                                      overlap_threshold: Optional[float] = None,
                                      plot_type: str = 'combined'):
        """Plot overlaps with local-:math:`S_z` product-basis states for the eigenstate labeling diagnostics."""

        if plot_type not in {'combined', 'diag_offdiag_separate'}:
            raise SimphonyError("plot_type must be 'combined' or 'diag_offdiag_separate'.")

        overlap_matrix = np.abs(np.asarray(self._eigenbasis)) ** 2
        diagonal_overlaps = np.diag(overlap_matrix)

        if overlap_threshold is None:
            selected_row_indices = np.arange(overlap_matrix.shape[0])
            title = 'overlap matrix'
        else:
            selected_row_indices = np.flatnonzero(diagonal_overlaps < overlap_threshold)
            title = (
                'overlap matrix '
                f'(diagonal overlap < {overlap_threshold:.1f})'
            )

        if selected_row_indices.size == 0:
            fig, ax = plt.subplots(figsize=(6, 1.8))
            ax.axis('off')
            ax.text(0.5, 0.5, 'No eigenstates selected.', ha='center', va='center')
            ax.set_title(title, fontsize=10)
            plt.tight_layout()
            plt.draw()
            return

        matrix_to_plot = overlap_matrix[selected_row_indices, :]
        row_states = [self._basis[idx] for idx in selected_row_indices]
        col_states = list(self._basis)
        formatted_state_labels = self._format_labeling_state_labels(row_states + col_states)
        row_labels = formatted_state_labels[:len(row_states)]
        col_labels = formatted_state_labels[len(row_states):]

        n_rows, n_cols = matrix_to_plot.shape
        fig_width = 0.5 + 0.3 * n_cols
        fig_height = 0.5 + 0.3 * n_rows

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if plot_type == 'combined':
            img = ax.imshow(
                matrix_to_plot,
                interpolation='none',
                cmap=plt.cm.binary,
                vmin=0,
                vmax=1,
                aspect='equal',
            )
            images = [img]
        else:
            diag_mask = np.eye(n_rows, n_cols, dtype=bool)
            diag_values = np.where(diag_mask, matrix_to_plot, np.nan)
            offdiag_values = np.where(diag_mask, np.nan, matrix_to_plot)
            finite_diag_values = diag_values[~np.isnan(diag_values)]
            finite_offdiag_values = offdiag_values[~np.isnan(offdiag_values)]
            diag_vmin = float(np.min(finite_diag_values)) if finite_diag_values.size else 0.0
            diag_vmax = 1.0
            offdiag_vmin = 0.0
            offdiag_vmax = float(np.max(finite_offdiag_values)) if finite_offdiag_values.size else 1.0
            if np.isclose(offdiag_vmax, offdiag_vmin):
                offdiag_vmax = offdiag_vmin + 1e-12
            if np.isclose(diag_vmax, diag_vmin):
                diag_vmin = max(0.0, diag_vmax - 1e-12)

            pale_red = (1.0, 0.82, 0.82, 1.0)
            black = (0.0, 0.0, 0.0, 1.0)
            pale_blue = (0.45, 0.62, 1.0, 1.0)
            diag_cmap = LinearSegmentedColormap.from_list(
                'labeling_diag_overlap',
                [pale_red, black],
            )
            offdiag_cmap = LinearSegmentedColormap.from_list(
                'labeling_offdiag_overlap',
                [(1.0, 1.0, 1.0, 1.0), pale_blue],
            )
            diag_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
            offdiag_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

            diag_img = ax.imshow(
                diag_values,
                interpolation='none',
                cmap=diag_cmap,
                vmin=diag_vmin,
                vmax=diag_vmax,
                aspect='equal',
            )
            offdiag_img = ax.imshow(
                offdiag_values,
                interpolation='none',
                cmap=offdiag_cmap,
                vmin=offdiag_vmin,
                vmax=offdiag_vmax,
                aspect='equal',
            )
            images = [diag_img, offdiag_img]

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('product basis state', fontsize=9)
        ax.set_ylabel('eigenstate', fontsize=9)
        ax.set_xticks(range(n_cols), labels=col_labels, fontfamily='monospace', fontsize=6, rotation=90)
        ax.set_yticks(range(n_rows), labels=row_labels, fontfamily='monospace', fontsize=6)

        for col_idx in range(n_cols - 1):
            ax.axvline(col_idx + 0.5, alpha=0.3, color='k', linestyle='-', linewidth=0.8)
        for row_idx in range(n_rows - 1):
            ax.axhline(row_idx + 0.5, alpha=0.3, color='k', linestyle='-', linewidth=0.8)

        if plot_type == 'combined':
            plt.tight_layout()
        else:
            fig.subplots_adjust(right=0.80)
        plt.draw()

        pos = ax.get_position()
        one_cell_width = pos.width / max(n_cols, 1)
        pad_norm = one_cell_width * 0.5

        cbar_left = pos.x1 + pad_norm
        cbar_bottom = pos.y0
        cbar_width = one_cell_width
        cbar_height = pos.height

        if plot_type == 'combined':
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            fig.colorbar(images[0], cax=cbar_ax)
        else:
            cbar_width = max(one_cell_width * 0.9, 0.025)
            cbar_gap = cbar_width * 1.6
            diag_cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            offdiag_cbar_ax = fig.add_axes([
                cbar_left + cbar_width + cbar_gap,
                cbar_bottom,
                cbar_width,
                cbar_height,
            ])
            fig.colorbar(images[0], cax=diag_cbar_ax)
            fig.colorbar(images[1], cax=offdiag_cbar_ax)
            diag_cbar_ax.tick_params(labelsize=6, pad=1)
            offdiag_cbar_ax.tick_params(labelsize=6, pad=1)
        plt.plot()

    def _labeling_overlap_matrix(self,
                                 overlap_threshold: Optional[float] = None) -> np.ndarray:
        """Return local-:math:`S_z` product-basis overlaps, optionally filtered by diagonal overlap."""

        if overlap_threshold is not None and not 0.0 <= overlap_threshold <= 1.0:
            raise SimphonyError('overlap_threshold must be between 0 and 1.')

        overlap_matrix = np.abs(np.asarray(self._eigenbasis)) ** 2
        if overlap_threshold is None:
            return overlap_matrix

        diagonal_overlaps = np.diag(overlap_matrix)
        selected_row_indices = np.flatnonzero(diagonal_overlaps < overlap_threshold)
        return overlap_matrix[selected_row_indices, :]

    def _format_labeling_overlap_details_pretty(self,
                                                diagnostics: List[Dict[str, object]],
                                                decimals: int = 3) -> str:
        """Format detailed labeling overlaps as grouped text with aligned overlap rows."""
        if not isinstance(decimals, int) or decimals < 0:
            raise SimphonyError('decimals must be a non-negative integer.')

        if not diagnostics:
            return 'none'

        all_states = [item['eigenstate'] for item in diagnostics] + [
            overlap_entry['product_state']
            for item in diagnostics
            for overlap_entry in item['overlaps']
        ]
        formatted_state_labels = self._format_labeling_state_labels(all_states)
        eigenstate_strings = formatted_state_labels[:len(diagnostics)]
        product_state_strings = formatted_state_labels[len(diagnostics):]
        overlap_strings = [
            f"{overlap_entry['overlap']:.{decimals}f}"
            for item in diagnostics
            for overlap_entry in item['overlaps']
        ]

        eigenstate_width = max(len(value) for value in eigenstate_strings)
        product_state_width = max(len(value) for value in product_state_strings)
        overlap_width = max(len(value) for value in overlap_strings)

        header = "eigenstate: - product state -> overlap"
        separator = "-" * max(
            len(header),
            len("eigenstate: ") + eigenstate_width,
            len("  - ") + product_state_width + len(" -> ") + overlap_width,
        )

        lines = [header, separator]
        for item, eigenstate_string in zip(diagnostics, eigenstate_strings):
            line_prefix = f"{eigenstate_string}: "
            continuation_prefix = " " * len(line_prefix)
            for overlap_idx, product_state_string, overlap_entry in zip(
                range(len(item['overlaps'])),
                product_state_strings[:len(item['overlaps'])],
                item['overlaps'],
            ):
                product_state_string = product_state_string.ljust(product_state_width)
                overlap_string = f"{overlap_entry['overlap']:.{decimals}f}".rjust(overlap_width)
                prefix = line_prefix if overlap_idx == 0 else continuation_prefix
                lines.append(f"{prefix}- {product_state_string} -> {overlap_string}")
            product_state_strings = product_state_strings[len(item['overlaps']):]
        return "\n".join(lines)

    @staticmethod
    def _format_labeling_state_labels(states: Sequence[Tuple[float, ...]]) -> List[str]:
        """Return consistently padded state labels for text output and plot axes."""

        if not states:
            return []

        quantum_num_width = max(len(str(value)) for state in states for value in state)
        return [
            f"({', '.join(str(value).rjust(quantum_num_width) for value in state)})"
            for state in states
        ]

    def eigenenergy(self,
                    quantum_nums: Union[str, Dict[str, float], Tuple[float, ...]]) -> float:
        """Return the eigenenergy of the static Hamiltonian associated with the eigenstate that has the largest
        overlap with the local-:math:`S_z` product-basis state described by ``quantum_nums``.

        Args:
            quantum_nums: Quantum numbers. It can be specified by three different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.
                - By a bitstring such as ``"010"``, interpreted in the :attr:`spin_names` order using each spin's ``qubit_subspace``.

        Returns:
            eigenenergy (frequency in :math:`\\text{MHz}`)

        """

        self._build('eigensystem')

        if isinstance(quantum_nums, str):
            quantum_nums = self._quantum_nums_from_bitstring(quantum_nums)
        elif isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        idx = self._map_quantum_nums_to_idx.get(quantum_nums)

        if idx is None:
            raise SimphonyError('Invalid quantum number provided.')

        return self._eigenenergies[idx]

    def eigenstate(self,
                   quantum_nums: Union[str, Dict[str, float], Tuple[float, ...]]) -> uarray:
        """Return the static-Hamiltonian eigenstate assigned to the local-:math:`S_z` product-basis state.

        Args:
            quantum_nums: Quantum numbers. It can be specified by three different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.
                - By a bitstring such as ``"010"``, interpreted in the :attr:`spin_names` order using each spin's ``qubit_subspace``.

        Returns:
            eigenstate (coefficient in the product basis)

        """

        self._build('eigensystem')

        if isinstance(quantum_nums, str):
            quantum_nums = self._quantum_nums_from_bitstring(quantum_nums)
        elif isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        idx = self._map_quantum_nums_to_idx.get(quantum_nums)

        if idx is None:
            raise SimphonyError('Invalid quantum number provided.')

        return self._eigenbasis[idx]

    def test_labeling(self,
                      output: str = 'print',
                      return_type: str = 'dict',
                      plot_type: str = 'combined',
                      overlap_threshold: Optional[float] = None,
                      overlaps: Union[str, int] = 'relevant',
                      decimals: int = 3) -> Optional[Union[List[Dict[str, object]], np.ndarray]]:
        """Inspect eigenstate labeling quality in the local-:math:`S_z` product basis.

        Computes ``np.abs(model.eigenbasis)**2`` row-wise and reports overlaps
        with local-:math:`S_z` product-basis states for each eigenstate. When
        ``overlap_threshold`` is provided, only eigenstates below that diagonal
        overlap are included.

        Args:
            output: Output mode. ``'return'`` returns data, ``'print'`` prints a
                formatted dict-style report, and ``'plot'`` visualizes the overlap
                matrix.
            return_type: Return payload used when ``output='return'``. ``'dict'`` returns
                structured overlap data and ``'matrix'`` returns the overlap matrix as a
                plain ``numpy.ndarray``.
            plot_type: Plot style used when ``output='plot'``. ``'combined'`` uses a
                shared grayscale colormap. ``'diag_offdiag_separate'`` colors diagonal
                and off-diagonal overlaps separately and uses independent scales.
            overlap_threshold: Optional dominant local-:math:`S_z` product-basis overlap threshold. ``None``
                reports all eigenstates.
            overlaps: Overlap selection used in ``return_type='dict'`` and ``output='print'``
                mode. ``'all'`` keeps all overlaps, an integer keeps the largest ``n``
                overlaps, and ``'relevant'`` keeps the smallest prefix whose cumulative
                overlap reaches ``0.99``.
            decimals: Number of digits printed after the decimal point for overlaps.

        Returns:
            If ``output='return'`` and ``return_type='dict'``, returns a list sorted by
            increasing diagonal overlap. Each entry contains ``eigenstate`` and
            ``overlaps``. The ``overlaps`` value is a list of dictionaries with
            ``product_state`` and ``overlap`` keys, sorted by decreasing overlap.
            If ``return_type='matrix'``, returns the overlap matrix as a ``numpy.ndarray``.
            Otherwise returns ``None``.
        """
        self._build('eigensystem')
        if output not in {'return', 'print', 'plot'}:
            raise SimphonyError("output must be 'return', 'print', or 'plot'.")
        if return_type not in {'dict', 'matrix'}:
            raise SimphonyError("return_type must be 'dict' or 'matrix'.")

        if output == 'return':
            if return_type == 'matrix':
                return self._labeling_overlap_matrix(overlap_threshold=overlap_threshold)
            diagnostics = self._collect_labeling_overlap_details(
                overlap_threshold=overlap_threshold,
                overlaps=overlaps,
            )
            for item in diagnostics:
                item.pop('diagonal_overlap', None)
            return diagnostics

        if output == 'print':
            diagnostics = self._collect_labeling_overlap_details(
                overlap_threshold=overlap_threshold,
                overlaps=overlaps,
            )
            for item in diagnostics:
                item.pop('diagonal_overlap', None)
            if diagnostics:
                details = self._format_labeling_overlap_details_pretty(diagnostics, decimals=decimals)
                print(details)
            else:
                if overlap_threshold is None:
                    print("none")
                else:
                    print("none")
            return None

        if output == 'plot':
            self._plot_labeling_overlap_matrix(
                overlap_threshold=overlap_threshold,
                plot_type=plot_type,
            )
            return None

    def state(self,
              quantum_nums: Union[str, Dict[str, float], Tuple[float, ...]],
              basis: str = 'product',
              coeffs_basis: str = 'product') -> uarray:
        """Return the state of the model corresponding to the given quantum numbers.

        Args:
            quantum_nums: Quantum numbers. It can be specified by three different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.
                - By a bitstring such as ``"010"``, interpreted in the :attr:`spin_names` order using each spin's ``qubit_subspace``.

            basis: Specifies that the quantum numbers correspond to the product-basis state (``'product'``) or the
                eigenstate (``'eigen'``) of the model.
            coeffs_basis: Basis in which the returned coefficients are expanded. Must be ``'product'`` or ``'eigen'``.

        Return:
            state (coefficients either in the product basis or in the eigenbasis)

        """

        if isinstance(quantum_nums, str):
            quantum_nums = self._quantum_nums_from_bitstring(quantum_nums)
        elif isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        if basis == 'eigen' or coeffs_basis == 'eigen':
            self._build('eigensystem')
        else:
            self._build('static_hamiltonian')

        if coeffs_basis == 'product':

            if basis == 'eigen':
                return self.eigenstate(quantum_nums)
            elif basis == 'product':
                return self.productstate(quantum_nums)
            else:
                raise SimphonyError("Invalid basis, must be 'product' or 'eigen'")

        elif coeffs_basis == 'eigen':

            if basis == 'eigen':
                return self.productstate(quantum_nums)
            elif basis == 'product':
                return self._eigenbasis @ self.productstate(quantum_nums)
            else:
                raise SimphonyError("Invalid basis, must be 'product' or 'eigen'")

        else:
            raise SimphonyError("Invalid coeffs_basis, must be 'product' or 'eigen'")

    def splitting(self,
                  spin_name: str,
                  quantum_nums: Tuple[float, float],
                  rest_quantum_nums: Dict[str, float]) -> float:
        """Return the energy splitting between the two eigenstates of the model. The two eigenstates differ only in a
        single quantum number, which is characterized by the ``spin_name`` and its two ``quantum_nums``. The quantum
        numbers of the remaining spins are provided in the ``rest_quantum_numbers``.

        Args:
            spin_name: Name of the spin.
            quantum_nums: Two different quantum numbers of ``spin_name``.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Returns:
            energy splitting (frequency in :math:`\\text{MHz}`)

        """

        if spin_name not in self.spin_names:
            raise SimphonyError("Invalid spin_name")

        idx = self.spin_names.index(spin_name)

        if isinstance(rest_quantum_nums, dict):

            if set(self.spin_names).difference([spin_name]) != set(rest_quantum_nums.keys()):
                raise SimphonyError('Invalid rest_quantum_nums.')

            rest_quantum_nums = [rest_quantum_nums[key] for key in self.spin_names if key in rest_quantum_nums.keys()]

        elif isinstance(rest_quantum_nums, tuple):

            rest_quantum_nums = list(rest_quantum_nums)

        quantum_nums_0 = tuple(rest_quantum_nums[:idx] + [quantum_nums[0]] + rest_quantum_nums[idx:])
        quantum_nums_1 = tuple(rest_quantum_nums[:idx] + [quantum_nums[1]] + rest_quantum_nums[idx:])

        return self.eigenenergy(quantum_nums_1) - self.eigenenergy(quantum_nums_0)

    def splitting_qubit(self,
                        spin_name: str,
                        rest_quantum_nums: Dict[str, float] = None) -> float:
        """Return the energy splitting between the two eigenstates of the model. The two eigenstates differ only in a
        single quantum number, which corresponds to the ``spin_name``. The quantum numbers of the remaining spins are
        provided in the ``rest_quantum_numbers``.

        Args:
            spin_name: Name of the spin.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Note:
            If ``rest_quantum_nums`` is not provided or incomplete, the splitting is averaged over all possible quantum
            numbers for the unspecified spins.


        Returns:
            energy splitting (frequency in :math:`\\text{MHz}`)

        """

        idx = self.spin_names.index(spin_name)
        quantum_nums = self.spins[idx].qubit_subspace

        rest_spin_names = self.spin_names[:idx] + self.spin_names[idx + 1:]
        if rest_quantum_nums is None:
            rest_quantum_nums = {}
        specified_spin_names = set(rest_quantum_nums.keys())
        unspecified_spin_names = [spin_name for spin_name in rest_spin_names if spin_name not in specified_spin_names]

        if not unspecified_spin_names:
            return self.splitting(spin_name, quantum_nums, rest_quantum_nums)
        else:
            unspecified_qubit_subspaces = [self.spin(spin_name).qubit_subspace for spin_name in unspecified_spin_names]
            unspecified_qubit_subspaces = product(*unspecified_qubit_subspaces)
            splittings = []
            for unspecified_quantum_nums in unspecified_qubit_subspaces:
                rest_quantum_nums_1 = rest_quantum_nums.copy()
                rest_quantum_nums_1.update(dict(zip(unspecified_spin_names, unspecified_quantum_nums)))
                splitting = self.splitting(spin_name, quantum_nums, rest_quantum_nums_1)
                splittings.append(splitting)
            return unp_dynamic.mean(unp_dynamic.array(splittings))

    def _effective_driving_operator(self,
                                   driving_field_name: str) -> uarray:
        """Combine one driving field's operator components into a single effective operator."""
        self._build('driving_operators')
        driving_field = self.driving_field(driving_field_name)
        driving_operators = self._driving_operators[driving_field_name]
        coefficients = driving_field._component_coefficients()

        if len(driving_operators) != len(coefficients):
            raise SimphonyError('Mismatch between driving operator components and driving-field coefficients.')

        operator = coefficients[0] * driving_operators[0]
        for coefficient, driving_operator in zip(coefficients[1:], driving_operators[1:]):
            operator = operator + coefficient * driving_operator
        return operator

    def matrix_element(self,
                       driving_field_name: str,
                       state1: uarray,
                       state2: uarray) -> complex:
        """Return the matrix element of the effective driving operator between two states.

        Args:
            driving_field_name: Name of the driving field.
            state1: First state.
            state2: Second state.

        Returns:
            matrix element (in :math:`\\text{MHz}`)

        """

        driving_operator = self._effective_driving_operator(driving_field_name)

        return unp_dynamic.linalg.multi_dot([state1.conjugate(),
                                             driving_operator,
                                             state2])

    def _rabi_helper(self,
                     driving_field_name: str,
                     spin_name: str,
                     quantum_nums: Union[dict, tuple],
                     rest_quantum_nums: Dict[str, float]) -> complex:
        """Return the transition matrix element used by the public Rabi helpers."""

        idx = self.spin_names.index(spin_name)

        if isinstance(rest_quantum_nums, dict):
            rest_quantum_nums = [rest_quantum_nums[key] for key in self.spin_names if key in rest_quantum_nums.keys()]
        elif isinstance(rest_quantum_nums, tuple):
            rest_quantum_nums = list(rest_quantum_nums)

        quantum_nums_0 = tuple(rest_quantum_nums[:idx] + [quantum_nums[0]] + rest_quantum_nums[idx:])
        quantum_nums_1 = tuple(rest_quantum_nums[:idx] + [quantum_nums[1]] + rest_quantum_nums[idx:])

        matrix_element = self.matrix_element(driving_field_name,
                                             self.eigenstate(quantum_nums_0),
                                             self.eigenstate(quantum_nums_1))

        return matrix_element

    def rabi_period(self,
                    driving_field_name: str,
                    amplitude: float,
                    spin_name: str,
                    quantum_nums: Union[dict, tuple],
                    rest_quantum_nums: Union[dict, tuple]) -> float:
        """Return the period time of a Rabi cycle for a resonant transition of the model under the influence of a
        constant-strength driving field. The driving field is described by its ``amplitude``, while the two eigenstates
        characterizing the transition are defined by ``quantum_nums`` of the ``spin_name`` and by the ``rest_quantum_nums``,
        i.e. the quantum numbers of the remaining spins.

        Args:
            driving_field_name: Name of the driving field.
            amplitude: Strength of the driving field (in :math:`\\text{T}`).
            spin_name: Name of the spin.
            quantum_nums: Two different quantum numbers of the ``spin_name``.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Returns:
            Period time of a Rabi cycle (in :math:`\\mu\\text{s}`)

        """

        matrix_element = self._rabi_helper(driving_field_name=driving_field_name,
                                           spin_name=spin_name,
                                           quantum_nums=quantum_nums,
                                           rest_quantum_nums=rest_quantum_nums)

        return 1 / (amplitude * unp_dynamic.abs(matrix_element))

    def rabi_amplitude(self,
                       driving_field_name: str,
                       period_time: float,
                       spin_name: str,
                       quantum_nums: Union[dict, tuple],
                       rest_quantum_nums: Dict[str, float]) -> float:
        """Return the required amplitude of a driving field for a single Rabi cycle assuming resonant transition of the
        model under the influence of a constant-strength driving field. Rabi cycle is described by the ``period_time``,
        while the two eigenstates characterizing the transition are defined by ``quantum_nums`` of the ``spin_name`` and
        by the ``rest_quantum_nums``, i.e. the quantum numbers of the remaining spins.

        Args:
            driving_field_name: Name of the driving field.
            period_time: Period time of a Rabi cycle (in :math:`\\mu\\text{s}`).
            spin_name: Name of the spin.
            quantum_nums: Two different quantum numbers of the ``spin_name``.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Returns:
            Amplitude of the driving field (in T)

        """

        matrix_element = self._rabi_helper(driving_field_name=driving_field_name,
                                           spin_name=spin_name,
                                           quantum_nums=quantum_nums,
                                           rest_quantum_nums=rest_quantum_nums)

        return 1 / (period_time * unp_dynamic.abs(matrix_element))

    def _rabi_qubit_helper(self,
                                 function,
                                 driving_field_name: str,
                                 spin_name: str,
                                 value: float,
                                 rest_quantum_nums: Dict[str, float] = None) -> float:
        """Evaluate a public qubit-level Rabi helper, averaging over unspecified spectator states."""

        idx = self.spin_names.index(spin_name)
        quantum_nums = self.spins[idx].qubit_subspace

        rest_spin_names = self.spin_names[:idx] + self.spin_names[idx + 1:]
        if rest_quantum_nums is None:
            rest_quantum_nums = {}
        specified_spin_names = set(rest_quantum_nums.keys())
        unspecified_spin_names = [spin_name for spin_name in rest_spin_names if spin_name not in specified_spin_names]

        if not unspecified_spin_names:
            return function(driving_field_name, value, spin_name, quantum_nums, rest_quantum_nums)
        else:
            unspecified_qubit_subspaces = [self.spin(spin_name).qubit_subspace for spin_name in unspecified_spin_names]
            unspecified_qubit_combinations = product(*unspecified_qubit_subspaces)
            results = []
            for unspecified_quantum_nums in unspecified_qubit_combinations:
                rest_quantum_nums_1 = rest_quantum_nums.copy()
                rest_quantum_nums_1.update(dict(zip(unspecified_spin_names, unspecified_quantum_nums)))
                result = function(driving_field_name, value, spin_name, quantum_nums, rest_quantum_nums_1)
                results.append(result)
            return unp_dynamic.mean(unp_dynamic.array(results))

    def rabi_period_qubit(self,
                              driving_field_name: str,
                              amplitude: float,
                              spin_name: str,
                              rest_quantum_nums: Dict[str, float] = None) -> float:
        """Return the period time of a Rabi cycle for a resonant transition of the model under the influence of a
        constant-strength driving field. The driving field is described by its ``amplitude``, while the two eigenstates
        characterizing the transition are defined by the qubit states of the ``spin_name`` and by the
        ``rest_quantum_nums``, i.e. the quantum number(s) of the remaining spin(s).

        Args:
            driving_field_name: Name of the driving field.
            amplitude: Strength of the driving field (in :math:`\\text{T}`).
            spin_name: Name of the spin.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Returns:
            Period time of a Rabi cycle (in :math:`\\mu\\text{s}`)

        """

        return self._rabi_qubit_helper(function=self.rabi_period,
                                       driving_field_name=driving_field_name,
                                       value=amplitude,
                                       spin_name=spin_name,
                                       rest_quantum_nums=rest_quantum_nums)

    def rabi_amplitude_qubit(self,
                             driving_field_name: str,
                             period_time: float,
                             spin_name: str,
                             rest_quantum_nums: Dict[str, float] = None) -> float:
        """Return the required amplitude of a driving field for a single Rabi cycle assuming resonant transition of the
        model under the influence of a constant-strength driving field. Rabi cycle is described by the ``period_time``,
        while the two eigenstates characterizing the transition are defined by the qubit states of the ``spin_name`` and
        by the ``rest_quantum_nums``, i.e. the quantum numbers of the remaining spin(s).

        Args:
            driving_field_name: Name of the driving field.
            period_time: Period time of a Rabi cycle (in :math:`\\mu\\text{s}`).
            spin_name: Name of the spin.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s) as a dictionary, in which keys are spin names,
                and values are quantum numbers.

        Returns:
            Amplitude of the driving field (in :math:`\\text{T}`)

        """

        return self._rabi_qubit_helper(function=self.rabi_amplitude,
                                       driving_field_name=driving_field_name,
                                       value=period_time,
                                       spin_name=spin_name,
                                       rest_quantum_nums=rest_quantum_nums)

    def rabi_cycle_amplitude_qubit(self,
                                   driving_field_name: str,
                                   period_time: float,
                                   spin_name: str,
                                   rest_quantum_nums: Dict[str, float] = None) -> float:
        """Compatibility alias for :meth:`rabi_amplitude_qubit`.

        Args:
            driving_field_name: Name of the driving field.
            period_time: Period time of a Rabi cycle (in :math:`\\mu\\text{s}`).
            spin_name: Name of the spin.
            rest_quantum_nums: Quantum number(s) of the remaining spin(s).

        """

        return self.rabi_amplitude_qubit(
            driving_field_name=driving_field_name,
            period_time=period_time,
            spin_name=spin_name,
            rest_quantum_nums=rest_quantum_nums,
        )

    def _build_static_hamiltonian(self):
        """Assemble and rotate the cached static Hamiltonian and basis metadata."""

        self._basis = list(product(*[spin.quantum_nums for spin in self.spins]))
        self._basis_qubit_subspace = list(product(*[spin.qubit_subspace for spin in self.spins]))

        self._static_hamiltonian = unp_static.zeros((self.dimension, self.dimension), dtype=complex)

        # spins
        for idx, spin in enumerate(self.spins):
            D_tensor = spin.anisotropy_tensor
            for i in range(3):
                for j in range(3):
                    ops = [s.operator.i for s in self.spins]
                    ops[idx] = spin.operator.vec[i] @ spin.operator.vec[j]
                    op = tensorprod(*ops, unp_inner=unp_static)

                    self._static_hamiltonian += D_tensor[i, j] * op

        # static_fields
        for static_field in self.static_fields:
            target_indices = self._field_target_indices(static_field.target_spins)

            for idx in target_indices:
                spin = self.spins[idx]
                zeeman_sign = float(spin.__class__.ZEEMAN_SIGN)
                signed_gyromagnetic_ratio = zeeman_sign * spin.gyromagnetic_ratio

                for i in range(3):
                    ops = [s.operator.i for s in self.spins]
                    ops[idx] = spin.operator.vec[i]
                    op = tensorprod(*ops, unp_inner=unp_static)
                    component = static_field.vec[i]

                    self._static_hamiltonian += signed_gyromagnetic_ratio * component * op

        # interactions
        for interaction in self.interactions:

            idx1 = self.spins.index(interaction.spin1)
            idx2 = self.spins.index(interaction.spin2)

            for i in range(3):
                for j in range(3):
                    ops = [spin.operator.i for spin in self.spins]
                    ops[idx1] = interaction.spin1.operator.vec[i]
                    ops[idx2] = interaction.spin2.operator.vec[j]

                    op = tensorprod(*ops, unp_inner=unp_static)
                    coeff = interaction.tensor[i][j]

                    self._static_hamiltonian += coeff * op

        self._map_quantum_nums_to_idx = {quantum_nums: idx for idx, quantum_nums in enumerate(self._basis)}

        self._build('quantization_transform')
        rot_op = self._quantization_rotation_operator
        rot_op_dag = self._quantization_rotation_operator_dagger
        self._static_hamiltonian = rot_op_dag @ self._static_hamiltonian @ rot_op


    def _build_driving_operators(self):
        """Assemble and rotate the cached driving operators for all configured driving fields."""

        empty = unp_static.zeros((self.dimension, self.dimension), dtype=complex)
        operator_groups: Dict[str, List[uarray]] = {}

        for driving_field in self.driving_fields:
            directions = driving_field._component_directions()
            field_operators = [empty.copy() for _ in directions]
            target_indices = self._field_target_indices(driving_field.target_spins)

            for spin_idx in target_indices:
                spin = self.spins[spin_idx]
                zeeman_sign = float(spin.__class__.ZEEMAN_SIGN)
                signed_gyromagnetic_ratio = zeeman_sign * spin.gyromagnetic_ratio

                for component_idx, direction in enumerate(directions):
                    for i in range(3):
                        ops = [s.operator.i for s in self.spins]
                        ops[spin_idx] = spin.operator.vec[i]
                        op = tensorprod(*ops)
                        field_operators[component_idx] += signed_gyromagnetic_ratio * direction[i] * op

            operator_groups[driving_field.name] = field_operators

        self._build('quantization_transform')
        rot_op = self._quantization_rotation_operator
        rot_op_dag = self._quantization_rotation_operator_dagger

        self._driving_operators = {
            name: [rot_op_dag @ operator @ rot_op for operator in field_operators]
            for name, field_operators in operator_groups.items()
        }

    def _build_noise_operators(self):
        """Assemble and rotate the cached quasistatic-noise operators."""

        self._local_quasistatic_noise_operators = []

        for idx, spin in enumerate(self.spins):
            operators_per_spin = []
            for i in range(3):
                ops = [spin.operator.i for spin in self.spins]
                ops[idx] = spin.operator.vec[i]
                op = tensorprod(*ops)
                operators_per_spin.append(op)
            self._local_quasistatic_noise_operators.append(operators_per_spin)

        self._build('quantization_transform')
        rot_op = self._quantization_rotation_operator
        rot_op_dag = self._quantization_rotation_operator_dagger

        for idx, ops_per_spin in enumerate(self._local_quasistatic_noise_operators):
            self._local_quasistatic_noise_operators[idx] = [rot_op_dag @ op @ rot_op for op in ops_per_spin]

    def _local_static_field_for_spin(self,
                                     spin_index: int) -> np.ndarray:
        """Return the summed static magnetic field acting on the spin indexed by ``spin_index``."""

        local_static_field = np.zeros(3, dtype=float)
        for static_field in self.static_fields:
            target_indices = self._field_target_indices(static_field.target_spins)
            if spin_index in target_indices:
                local_static_field += np.asarray(static_field.vec, dtype=float)
        return local_static_field

    def _resolved_quantization_axis(self,
                                    spin_index: int) -> np.ndarray:
        """Return a normalized lab-frame vector for the spin's quantization axis."""

        spin = self.spins[spin_index]
        axis_spec = spin.quantization_axis

        if isinstance(axis_spec, str):
            if axis_spec == 'global_z':
                axis = np.array([0.0, 0.0, 1.0], dtype=float)
            elif axis_spec == 'anisotropy_axis':
                axis = np.asarray(spin.anisotropy_axis.vec, dtype=float)
            elif axis_spec == 'local_static_field':
                axis = self._local_static_field_for_spin(spin_index)
            else:
                raise SimphonyError(
                    f"Unknown quantization axis specifier '{axis_spec}' for spin '{spin.name}'."
                )
        else:
            axis = np.asarray(getattr(axis_spec, 'vec', axis_spec), dtype=float)

        norm = np.linalg.norm(axis)
        if norm == 0:
            raise SimphonyError(
                f"Quantization axis for spin '{spin.name}' must be a non-zero vector."
            )

        return axis / norm

    def _build_quantization_transform(self):
        """Build the cached unitary mapping the global frame to the configured spin quantization axes."""

        if not self.spins:
            identity = unp_static.eye(self.dimension, dtype=complex)
            self._quantization_rotation_operator = identity
            self._quantization_rotation_operator_dagger = identity
            return

        per_spin_unitaries = []

        for spin_index, spin in enumerate(self.spins):

            target_dir = self._resolved_quantization_axis(spin_index)
            rotation_axis, angle = _axis_angle_from_local_z_to_direction(target_dir)

            if np.isclose(angle, 0.0):
                per_spin_unitaries.append(spin.operator.i)
                continue

            generator = (
                rotation_axis[0] * spin.operator.x
                + rotation_axis[1] * spin.operator.y
                + rotation_axis[2] * spin.operator.z
            )

            unitary = expm(-1j * angle * generator, unp_inner=unp_static)
            per_spin_unitaries.append(unitary)

        rotation = tensorprod(*per_spin_unitaries, unp_inner=unp_static)
        rotation_dagger = unp_static.conjugate(unp_static.transpose(rotation))
        self._quantization_rotation_operator = rotation
        self._quantization_rotation_operator_dagger = rotation_dagger

    def operator_from_string(self,
                             string: str,
                             in_qubit_subspace: bool = False) -> uarray:
        """Return tensor product operator from an extended operator string.

        Uppercase ``I/X/Y/Z`` refer to Pauli operators in qubit subspace, while lowercase ``x/y/z`` refer to full-space
        spin operators ``Sx/Sy/Sz``. ``M`` denotes the full-space projector onto the ``m_s = 0`` state.

        Args:
            string: Operator string with one character per spin.
            in_qubit_subspace: If ``True``, interpret the string in the qubit subspace.

        Raises:
            SimphonyError: If the string contains an unsupported character or has the wrong length.

        """

        if any(char not in {"I", "X", "Y", "Z", "1", "M", "x", "y", "z"} for char in string):
            raise SimphonyError("Unsupported character in the string")

        if len(string) != self.n_spins:
            raise SimphonyError(
                f"Operator string length must match number of spins ({self.n_spins}), got {len(string)}."
            )

        if in_qubit_subspace and any(char in {"1", "M", "x", "y", "z"} for char in string):
            bad_chars = ", ".join(sorted(set(char for char in string if char in {"1", "M", "x", "y", "z"})))
            raise SimphonyError(
                f"in_qubit_subspace=True is incompatible with full-space symbols in operator string: {bad_chars}. "
                "Use only I/X/Y/Z or set in_qubit_subspace=False."
            )

        operators = []
        if in_qubit_subspace:
            return unp_static.asarray(Pauli(string).to_matrix())
        else:
            for spin, char in zip(self.spins, string):
                if char == "1":
                    operators.append(spin.operator.i)
                elif char == "M":
                    if 0 not in spin.quantum_nums:
                        raise SimphonyError(f"Operator 'M' requires a spin with quantum number 0, got {spin.quantum_nums}.")
                    idx = spin.quantum_nums.index(0)
                    projector = unp_static.zeros((spin.dimension, spin.dimension), dtype=complex)
                    projector[idx, idx] = 1.0
                    operators.append(projector)
                elif char in {"x", "y", "z"}:
                    operators.append(getattr(spin.operator, char))
                else:
                    operators.append(getattr(spin.operator_qubit_subspace, char.lower()))

            return tensorprod(*operators, unp_inner=unp_static)

    def simulate_time_evolution(self,
                                n_shots: int = 1,
                                start: float = 0.,
                                end: Optional[float] = None,
                                apply_noise: bool = True,
                                random_seed: int = None,
                                simulation_method: str = 'single_sine_wave',
                                n_eval: int = 251,
                                n_split: int = 250,
                                time_grid_jitter: Optional[float] = None,
                                state_only: bool = False,
                                solver_method: Optional[str] = None,
                                verbose: bool = False,
                                test_convergence: Union[bool, int] = False,
                                time_segment_sequence = None,
                                batch_size: Optional[int] = None) -> SimulationResult:

        """Simulate the time evolution of the model due to the driving fields under the effect of the noise.

        Args:
            n_shots: Number of the shots.
            start: Start time of the simulation (in :math:`\\mu\\text{s}`, defaults to 0).
            end: End time of the simulation (in :math:`\\mu\\text{s}`, defaults to end of the last pulse).
            apply_noise: Whether to include noise or not.
            random_seed: Optional seed for the simulation-local random number generator used for noise
                randomization and time-grid jitter.
            simulation_method: It must be ``'basic'`` or ``'single_sine_wave'``. Latter simulates only a single sine
                wave in the case of a monochromatic excitation that could accelerate the simulation.
            n_eval: Number of evaluation points per time segments.
            n_split: Number of split points per time segments.
            time_grid_jitter: Optional jitter factor applied when subdividing time steps (None disables jitter; defaults to None).
            state_only: Whether to simulate the time evolution from an initial state or simulate the full propagators.
            solver_method: Method of the evolution solver used to simulate the time evolution (defaults to
                ``'numpy_expm'`` on CPU and ``'jax_expm_parallel'`` on GPU).
            batch_size: Optional number of shots to evolve in a single JAX batch. ``None`` processes all at once.
            verbose: Whether to verbose the details of the simulation or not.
            test_convergence: Whether to test the convergence of the final propagators by halving the time steps. If an
                ``int`` provided, it specifies the number of times the convergence test is repeated.

        Return:
            Results of the simulation, which contains the propagators/states of the simulation, a copy of the simulated
            model, the timestamps of the simulation and the noise realizations corresponding to the different shots.

        """

        self._build('static_hamiltonian', 'driving_operators', 'noise_operators')

        if end is None:
            end = self.last_pulse_end

        if not start < end:
            raise SimphonyError("end must be greater than start")

        if verbose:
            print(f'start = {start}\nend = {end}')

        state_only = self._validate_initial_conditions(state_only=state_only)

        # set default method
        solver_method = self._resolve_solver_method(solver_method)
        if verbose:
            print(f'solver_method = {solver_method}')

        # prepare initial state
        simulation_method, y0s = self._prepare_initial_states(state_only=state_only,
                                                              simulation_method=simulation_method,
                                                              n_shots=n_shots)
        self._y0s = y0s

        # prepare active driving fields
        active_driving_field_names, active_driving_operators = self._prepare_active_driving_terms()

        rng = np.random.default_rng(random_seed)

        # prepare noise coefficients
        active_noise_strengths, active_noise_operators, noise_coeffs = self._prepare_noise_signals(
            apply_noise=apply_noise,
            rng=rng,
            n_shots=n_shots,
        )
        self._noise_coeffs = noise_coeffs
        if verbose:
            print(f'number of simulated driving terms = {len(active_driving_operators)}')
            print(f'number of simulated noise terms = {len(active_noise_operators)}')

        # set the solver
        solver = Solver(static_hamiltonian=self._static_hamiltonian,
                        driving_operators=active_driving_operators or None,
                        noise_operators=active_noise_operators or None)
        self._solver = solver

        # time segments and time discretization
        time_segment_sequence = self._prepare_time_segment_sequence(
            start=start,
            end=end,
            simulation_method=simulation_method,
            active_driving_field_names=active_driving_field_names,
            n_eval=n_eval,
            n_split=n_split,
            time_grid_jitter=time_grid_jitter,
            time_segment_sequence=time_segment_sequence
        )
        self._time_segment_sequence = time_segment_sequence

        if verbose:
            print('-' * NUM_DECORATORS_VERBOSE)
        sequence_result = solve_time_segment_sequence(solver=solver,
                                                      y0s=y0s,
                                                      noise_coeffs=noise_coeffs,
                                                      time_segment_sequence=time_segment_sequence,
                                                      method=solver_method,
                                                      batch_size=batch_size,
                                                      verbose=verbose,
                                                      rng=rng)
        ts = sequence_result.ts
        solutions = sequence_result.ys

        self._ts = ts
        self._solutions = solutions

        # output the results
        model_copy = deepcopy(self)
        if state_only is True:
            simulation_result = SimulationResult(
                model=model_copy,
                ts=ts,
                time_evol_state=TimeEvolState(array=solutions, ts=ts, model=model_copy),
                shot_noises=noise_coeffs
            )
        else:
            simulation_result = SimulationResult(
                model=model_copy,
                ts=ts,
                time_evol_operator=TimeEvolOperator(array=solutions, ts=ts, model=model_copy),
                shot_noises=noise_coeffs
            )

        if model_copy.initial_state is not None:
            simulation_result.initial_state = model_copy.initial_state
        elif model_copy.initial_density_matrix is not None:
            simulation_result.initial_density_matrix = model_copy.initial_density_matrix

        # test the convergence
        if test_convergence:
            average_gate_infidelities = self._test_convergence(
                simulation_result=simulation_result,
                time_segment_sequence=time_segment_sequence,
                test_convergence=test_convergence,
                state_only=state_only,
                active_noise_strengths=active_noise_strengths,
                random_seed=random_seed,
                n_shots=n_shots,
                start=start,
                end=end,
                apply_noise=apply_noise,
                solver_method=solver_method,
                verbose=verbose,
                time_grid_jitter=time_grid_jitter,
                batch_size=batch_size,
            )

            if verbose:
                print('#' * NUM_DECORATORS_VERBOSE)
                print('test convergence report:')
                for idx, avg_gate_infidelity in enumerate(average_gate_infidelities):
                    _shot_str = f'shot {idx}'
                    print('{}{}'.format(_shot_str, '.' * (50 - len(_shot_str)) + str(avg_gate_infidelity)))

                print('-' * NUM_DECORATORS_VERBOSE)

        return simulation_result

    def _validate_initial_conditions(self,
                                     state_only: bool) -> bool:
        """Validate initial-state inputs and return the effective ``state_only`` mode."""

        if self.initial_state is not None and self.initial_density_matrix is not None:
            raise SimphonyError('initial_state and initial_density_matrix cannot both be set.')

        if self.initial_state is not None:
            initial_state = unp_static.asarray(self.initial_state)
            if initial_state.ndim != 1 or initial_state.shape[0] != self.dimension:
                raise SimphonyError(
                    f'initial_state must have shape ({self.dimension},), got {initial_state.shape}.'
                )

        if self.initial_density_matrix is not None:
            initial_density_matrix = unp_static.asarray(self.initial_density_matrix)
            if initial_density_matrix.ndim != 2 or initial_density_matrix.shape != (self.dimension, self.dimension):
                raise SimphonyError(
                    'initial_density_matrix must have shape '
                    f'({self.dimension}, {self.dimension}), got {initial_density_matrix.shape}.'
                )
            if state_only:
                warn(
                    'If initial_density_matrix is set, state_only is ignored and full propagators are simulated.',
                    SimphonyWarning,
                    stacklevel=1,
                )
                state_only = False

        if state_only and self.initial_state is None:
            raise SimphonyError('Initial state is not initialized')

        return state_only

    def _resolve_solver_method(self, solver_method: Optional[str]) -> str:
        """Resolve and validate solver method from platform or user input."""
        valid_methods = {'numpy_expm', 'jax_expm', 'jax_expm_parallel'}
        if solver_method is None:
            platform = Config.get_platform()
            autodiff_enabled = Config.get_autodiff_mode()
            if platform == 'cpu':
                return 'jax_expm' if autodiff_enabled else 'numpy_expm'
            elif platform == 'gpu':
                return 'jax_expm_parallel'
            else:
                raise SimphonyError("Invalid backend, must be 'cpu' or 'gpu'")
        if solver_method not in valid_methods:
            raise SimphonyError("Invalid solver_method, must be one of 'numpy_expm', 'jax_expm', or 'jax_expm_parallel'")
        return solver_method

    def _prepare_initial_states(self,
                                state_only: bool,
                                simulation_method: str,
                                n_shots: int) -> Tuple[str, List[uarray]]:
        """Prepare solver initial states and adjust the simulation mode when needed."""

        if state_only is True:
            if simulation_method != 'basic':
                warn("If state_only is True, simulation_segment_method is set to 'basic'", SimphonyWarning, stacklevel=1)
                simulation_method = 'basic'
            initial_state = unp_static.array(self.initial_state, dtype=complex, copy=True)
            y0s = [initial_state] * n_shots
        else:
            identity = unp_static.identity(self.dimension, dtype=complex)
            y0s = [identity] * n_shots
        return simulation_method, y0s

    def _prepare_active_driving_terms(self) -> Tuple[List[BaseDrivingField], List[uarray]]:
        """Return active driving fields and flattened operators for the solver."""

        fields: List[BaseDrivingField] = []
        operators: List[uarray] = []
        for driving_field in self.driving_fields:
            if driving_field.pulses:
                fields.append(driving_field)
                operators.extend(self.driving_operators[driving_field.name])
        return fields, operators

    def _prepare_noise_signals(self,
                               apply_noise: bool,
                               rng,
                               n_shots: int):
        """Sample shot-dependent quasistatic-noise coefficients and select active noise operators."""

        if apply_noise:
            all_noise_strengths = unp_dynamic.array([spin.local_quasistatic_noise.vec for spin in self.spins])
            idx = all_noise_strengths.nonzero()
            active_noise_strengths = all_noise_strengths[idx]
            if len(active_noise_strengths.tolist()) == 0:
                active_noise_operators = []
            else:
                active_noise_operators = list(unp_dynamic.array(self._local_quasistatic_noise_operators)[idx])
        else:
            active_noise_strengths = unp_dynamic.array([])
            active_noise_operators = []

        n_active_noise = len(active_noise_strengths)
        if n_active_noise == 0:
            noise_coeffs = np.zeros((n_shots, 0), dtype=float)
        else:
            noise_coeffs = rng.standard_normal((n_shots, n_active_noise)) * unp_dynamic.asarray(active_noise_strengths)
        return active_noise_strengths, active_noise_operators, noise_coeffs

    def _prepare_time_segment_sequence(self,
                                       start,
                                       end,
                                       simulation_method,
                                       active_driving_field_names,
                                       n_eval,
                                       n_split,
                                       time_grid_jitter: Optional[float] = None,
                                       time_segment_sequence=None):
        """Build or reuse TimeSegmentSequence and discretize segments for simulation."""

        if time_segment_sequence is None:
            time_segment_sequence = self.pulses.convert_to_time_segment_sequence(start=start,
                                                                                 end=end)
            time_segment_sequence.discretize_simulation(method=simulation_method,
                                                        included_driving_field_names=active_driving_field_names,
                                                        n_eval=n_eval,
                                                        n_split=n_split,
                                                        time_grid_jitter=time_grid_jitter)
        return time_segment_sequence

    def _test_convergence(self,
                          *,
                          simulation_result: SimulationResult,
                          time_segment_sequence,
                          test_convergence: Union[bool, int],
                          state_only: bool,
                          active_noise_strengths,
                          random_seed: Optional[int],
                          n_shots: int,
                          start: float,
                          end: float,
                          apply_noise: bool,
                          solver_method: str,
                          verbose: bool,
                          time_grid_jitter: Optional[float],
                          batch_size: Optional[int]) -> List[float]:
        """Perform optional convergence test by halving step sizes."""

        if state_only:
            raise SimphonyError('Convergence test is not compatible with state only mode')

        if len(active_noise_strengths) > 0 and random_seed is None:
            warn("Convergence test without setting random_seed", SimphonyWarning, stacklevel=1)

        current_operators = simulation_result.time_evol_operator.matrix()[:, -1]
        num_halvings = test_convergence if isinstance(test_convergence, int) else 1

        for _ in range(num_halvings):
            if verbose:
                print('#' * NUM_DECORATORS_VERBOSE)
                print('step-halving')
                print('-' * NUM_DECORATORS_VERBOSE)

            for time_segment in time_segment_sequence:
                time_segment.simulation.max_dt /= 2

            refined_result = self.simulate_time_evolution(
                n_shots=n_shots,
                start=start,
                end=end,
                apply_noise=apply_noise,
                random_seed=random_seed,
                solver_method=solver_method,
                verbose=verbose,
                time_grid_jitter=time_grid_jitter,
                test_convergence=False,
                time_segment_sequence=time_segment_sequence,
                batch_size=batch_size,
            )

            refined_operators = refined_result.time_evol_operator.matrix()[:, -1]
            average_gate_infidelities = [
                1 - average_gate_fidelity_from_unitaries(U1, U2)
                for U1, U2 in zip(current_operators, refined_operators)
            ]
            current_operators = refined_operators

        return average_gate_infidelities

    def plot_levels(self,
                    zoom_electron_state: Optional[List[Tuple[float, ...]]] = None,
                    *,
                    height: Optional[float] = None,
                    return_fig_ax: bool = False):
        """Plot the energy spectrum with optionally zoomed electron subpanels.

        Args:
            zoom_electron_state: Electron basis states that receive dedicated zoom panels.
                Each entry must be an iterable of quantum numbers (for example ``[(0,), (-1,), (1,)]``).
                When omitted and the system has a single electron spin, all of its basis states are zoomed.
            height: Total figure height in inches. If ``None``, a height is chosen automatically based on the
                number of spins (capped at 12 inches).
            return_fig_ax: Whether to return the created Matplotlib figure and main axis.

        Raises:
            SimphonyError: If the static Hamiltonian has not been calculated or the model contains more than three
                electron spins.

        Returns:
            Figure and main axis if ``return_fig_ax`` is ``True``.
        """

        self._build('eigensystem')

        # --- Prepare data ---
        energies = self._eigenenergies
        basis = self._basis
        qubit_subspace = self._basis_qubit_subspace

        electron_idx = [i for i, s in enumerate(self.spins) if isinstance(s, ElectronSpin)]
        nuclear_idx = [i for i, s in enumerate(self.spins) if isinstance(s, NuclearSpin)]

        if len(electron_idx) > 3:
            raise SimphonyError('Plotting more than 3 electrons is not supported.')

        basis_e = [tuple(row[i] for i in electron_idx) for row in basis]
        basis_n = [tuple(row[i] for i in nuclear_idx) for row in basis]

        if len(electron_idx) == 1 and zoom_electron_state is None and len(nuclear_idx) > 0:
            electron_spin = self.spins[electron_idx[0]]
            zoom_electron_state = [(q,) for q in electron_spin.quantum_nums]

        zoom_electron_state = zoom_electron_state if zoom_electron_state else []

        sorted_data = sorted(zip(energies, basis_e, basis_n, basis), key=lambda x: x[0])
        energies, basis_e, basis_n, basis = zip(*sorted_data)

        # --- Build energy dictionary ---
        energy_dict = {e: {} for e in basis_e}
        for state_e, state_n, energy, full_state in zip(basis_e, basis_n, energies, basis):
            in_qubit = tuple(full_state) in qubit_subspace
            energy_dict[state_e][state_n] = {
                "energy": float(energy),
                "in_qubit_subspace": in_qubit
            }


        # --- Layout parameters ---
        base_panel_width_inch = 1.3
        width_per_spin_inch = 0.275
        main_width_inch = base_panel_width_inch + width_per_spin_inch * max(0, len(electron_idx) - 1)
        zoom_width_inch = base_panel_width_inch + width_per_spin_inch * max(0, len(nuclear_idx) - 1)
        hpad_inch = 0.65
        tick_label_size = 8
        label_font_size = 7
        zoom_title_font_size = label_font_size + 3
        legend_width_inch = 1.2
        legend_min_height_inch = 0.7
        legend_pad_inch = 0.35
        legend_title_pad = 0.16
        legend_row_gap = 0.38
        legend_description_offset = 0.14
        legend_title_font_size = label_font_size + 4
        legend_ket_font_size = label_font_size + 3
        legend_description_font_size = label_font_size + 2
        legend_title_fontweight = "normal"
        electron_color_active = "#ff0000"
        electron_color_inactive = "#ff9999"
        nuclear_color_active = "#0000ff"
        nuclear_color_inactive = "#9999ff"
        level_length_inch = 0.5
        energy_margin_fraction = 0.1


        if height is None:
            electron_height = 2 ** len(electron_idx)
            nuclear_height = 2 ** len(nuclear_idx) if nuclear_idx else 1
            auto_height = max(3, electron_height, nuclear_height)
            auto_height_cap = 12
            main_height_inch = min(auto_height, auto_height_cap)
        else:
            main_height_inch = height

        n_zoom = len(zoom_electron_state)

        total_width_inch = main_width_inch + n_zoom * (zoom_width_inch + hpad_inch)
        include_legend_column = True
        if include_legend_column:
            total_width_inch += legend_pad_inch + legend_width_inch
        fig = plt.figure(figsize=(total_width_inch, main_height_inch))

        def _to_norm_rect(x_inch, y_inch, w_inch, h_inch):
            return [
                x_inch / total_width_inch,
                y_inch / main_height_inch,
                w_inch / total_width_inch,
                h_inch / main_height_inch
            ]

        # --- Main plot ---
        x_cursor = 0.0
        ax = fig.add_axes(_to_norm_rect(x_cursor, 0.0, main_width_inch, main_height_inch))
        x_cursor += main_width_inch
        if n_zoom > 0:
            x_cursor += hpad_inch

        # horizontal line parameters
        x_offset = 0.08
        label_x_offset = 0.18

        label_x_size_main = width_per_spin_inch * max(1, len(electron_idx))
        rest_span_main = label_x_offset + label_x_size_main + x_offset
        denom_main = max(main_width_inch - level_length_inch, 1e-6)
        main_line_width = rest_span_main * level_length_inch / denom_main
        x_max_main = main_line_width + rest_span_main

        ax.hlines(energies, 0, main_line_width, color="k", linewidth=1.0)
        ax.set_xlim(-x_offset, x_max_main)
        ax.set_ylabel("Energy (MHz)")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", labelsize=tick_label_size)

        energy_min_main = min(energies)
        energy_max_main = max(energies)
        energy_span_main = max(energy_max_main - energy_min_main, 1e-9)
        pad_main = energy_margin_fraction * energy_span_main
        ax.set_ylim(energy_min_main - pad_main, energy_max_main + pad_main)

        # label formatter
        def format_label(label):
            formatted = []
            for i in label:
                is_int = float(i).is_integer()
                num = int(2 * i) if not is_int else int(i)
                sign = " " if num == 0 else ("-" if num < 0 else "+")
                text = f"{sign}{abs(num)}" + ("/2" if not is_int else "")
                formatted.append(text.center(4))
            return "|" + ",".join(formatted) + "⟩"

        def _ket_template(spin_indices, default_prefix):
            if not spin_indices:
                return rf"$|m_{{{default_prefix}}}\rangle$"
            components = []
            counts = {}
            for pos, idx in enumerate(spin_indices, start=1):
                spin_name = self.spins[idx].name or ''
                sanitized = ''.join(ch if ch.isalnum() else '_' for ch in spin_name).strip('_')
                if not sanitized:
                    sanitized = f"{default_prefix}_{pos}"
                base = sanitized
                counts[base] = counts.get(base, 0) + 1
                count = counts[base]
                display_base = base.replace('_', r'\_')
                if count > 1:
                    display = rf"{display_base}_{{{count}}}"
                else:
                    display = display_base
                components.append(rf"m_{{{display}}}")
            ket_body = ', '.join(components)
            return rf"$|{ket_body}\rangle$"

        # --- Electron labels (main plot) ---
        labels = []
        labels_pos_y = []
        label_colors = []
        electron_active_used = False
        electron_inactive_used = False
        nuclear_active_used = False
        nuclear_inactive_used = False
        for state_e, energy_dict_n in energy_dict.items():
            energies_e = [entry["energy"] for entry in energy_dict_n.values()]
            energies_e_mean = sum(energies_e) / len(energies_e)
            labels_pos_y.append(energies_e_mean)
            labels.append(format_label(state_e))
            has_qubit_state = any(entry["in_qubit_subspace"] for entry in energy_dict_n.values())
            label_colors.append(electron_color_active if has_qubit_state else electron_color_inactive)
            if has_qubit_state:
                electron_active_used = True
            else:
                electron_inactive_used = True

        label_pos_x_main = main_line_width + label_x_offset
        texts = []
        for y, lbl, color in zip(labels_pos_y, labels, label_colors):
            t = ax.text(
                x=label_pos_x_main,
                y=y,
                s=lbl,
                va="center",
                ha="left",
                fontsize=label_font_size,
                family="monospace",
                color=color
            )
            texts.append(t)

        main_annotations = list(zip(texts, labels_pos_y))

        # Fine-tune the repulsion so labels can move without dropping too far.
        with _quiet_adjust_text():
            adjust_text(
                texts,
                only_move={'text': 'y'},
                expand_text=(1.0, 1.02),
                force_text=(0.03, 0.12)
            )

        # --- Electron zoom subplots ---
        zoom_annotations = []
        label_x_size_zoom = width_per_spin_inch * max(1, len(nuclear_idx))
        rest_span_zoom = label_x_offset + label_x_size_zoom + x_offset
        denom_zoom = max(zoom_width_inch - level_length_inch, 1e-6)
        zoom_line_width = rest_span_zoom * level_length_inch / denom_zoom
        x_max_zoom = zoom_line_width + rest_span_zoom
        label_pos_x_zoom = zoom_line_width + label_x_offset
        if n_zoom > 0:
            zoom_panels = []
            for electron_state_tuple in zoom_electron_state:
                if electron_state_tuple not in energy_dict:
                    raise SimphonyError(f"Electron state {electron_state_tuple} not found.")
                nuclear_levels = energy_dict[electron_state_tuple]
                nuclear_states = list(nuclear_levels.keys())
                level_records = [nuclear_levels[state] for state in nuclear_states]
                level_energies = [record["energy"] for record in level_records]
                qubit_membership = [record["in_qubit_subspace"] for record in level_records]
                energy_min = min(level_energies)
                energy_max = max(level_energies)
                energy_span = max(energy_max - energy_min, 1e-9)

                zoom_panels.append({
                    "electron_state": electron_state_tuple,
                    "nuclear_states": nuclear_states,
                    "energies": level_energies,
                    "qubit_membership": qubit_membership,
                    "energy_min": energy_min,
                    "energy_max": energy_max,
                    "energy_span": energy_span,
                    "has_qubit_state": any(qubit_membership)
                })

            max_panel_span = max(panel["energy_span"] for panel in zoom_panels)
            max_zoom_height_inch = 0.85 * main_height_inch
            inches_per_unit = max_zoom_height_inch / max_panel_span
            min_zoom_height_inch = 0.1 * main_height_inch
            min_effective_span = min_zoom_height_inch / inches_per_unit

            for panel in zoom_panels:
                effective_span = max(panel["energy_span"], min_effective_span)
                energy_center = 0.5 * (panel["energy_min"] + panel["energy_max"])
                panel["display_min"] = energy_center - 0.5 * effective_span
                panel["display_max"] = energy_center + 0.5 * effective_span
                panel["display_span"] = effective_span
                panel["height_inch"] = panel["display_span"] * inches_per_unit

            zoom_band_center_inch = 0.5 * main_height_inch

            for idx_panel, panel in enumerate(zoom_panels):
                panel_height_inch = panel["height_inch"]
                panel_bottom_inch = zoom_band_center_inch - 0.5 * panel_height_inch
                ax_zoom = fig.add_axes(_to_norm_rect(x_cursor, panel_bottom_inch, zoom_width_inch, panel_height_inch))

                # energy levels
                ax_zoom.hlines(panel["energies"], 0, zoom_line_width, color="k", linewidth=1.0)
                ax_zoom.set_xlim(-x_offset, x_max_zoom)

                pad_energy = energy_margin_fraction * panel["display_span"]
                ax_zoom.set_ylim(panel["display_min"] - pad_energy, panel["display_max"] + pad_energy)
                ax_zoom.tick_params(axis="x", bottom=False, labelbottom=False)
                ax_zoom.tick_params(axis="y", labelsize=tick_label_size)

                # Title with ket notation
                ax_zoom.set_title(
                    format_label(panel["electron_state"]),
                    fontsize=zoom_title_font_size,
                    pad=4,
                    color=electron_color_active if panel["has_qubit_state"] else electron_color_inactive
                )
                if panel["has_qubit_state"]:
                    electron_active_used = True
                else:
                    electron_inactive_used = True

                # --- Nuclear state labels with adjust_text ---
                zoom_texts = []
                for energy, nuclear_state, is_qubit in zip(panel["energies"], panel["nuclear_states"], panel["qubit_membership"]):
                    text_color = nuclear_color_active if is_qubit else nuclear_color_inactive
                    text = ax_zoom.text(
                        label_pos_x_zoom,
                        energy,
                        format_label(nuclear_state),
                        va="center",
                        ha="left",
                        fontsize=label_font_size,
                        family="monospace",
                        color=text_color
                    )
                    zoom_texts.append(text)
                    if is_qubit:
                        nuclear_active_used = True
                    else:
                        nuclear_inactive_used = True

                # Soften the forces to keep zoom labels separated while limiting downward drift.
                with _quiet_adjust_text():
                    adjust_text(
                        zoom_texts,
                        only_move={'text': 'y'},
                        expand_text=(1.0, 1.02),
                        force_text=(0.03, 0.12),
                        ax=ax_zoom
                    )

                zoom_annotations.append((ax_zoom, zoom_line_width, list(zip(zoom_texts, panel["energies"]))))

                x_cursor += zoom_width_inch
                if idx_panel < n_zoom - 1:
                    x_cursor += hpad_inch

        electron_template = _ket_template(electron_idx, "e")
        nuclear_template = _ket_template(nuclear_idx, "n")

        legend_entries = []
        if electron_active_used:
            legend_entries.append((electron_template, "electron state in qubit subspace", electron_color_active))
        if electron_inactive_used:
            legend_entries.append((electron_template, "electron state outside qubit subspace", electron_color_inactive))
        if nuclear_active_used:
            legend_entries.append((nuclear_template, "nuclear state in qubit subspace", nuclear_color_active))
        if nuclear_inactive_used:
            legend_entries.append((nuclear_template, "nuclear state outside qubit subspace", nuclear_color_inactive))

        if include_legend_column and legend_entries:
            # Add a dedicated legend column explaining the color coding.
            legend_height = max(legend_min_height_inch, legend_title_pad + legend_description_offset + len(legend_entries) * legend_row_gap)
            x_cursor += legend_pad_inch
            legend_ax = fig.add_axes(_to_norm_rect(x_cursor, main_height_inch - legend_height, legend_width_inch, legend_height))
            x_cursor += legend_width_inch
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, legend_height)
            legend_ax.axis("off")
            title_y = legend_height - legend_title_pad
            legend_ax.text(0.0, title_y, "Legend", fontsize=legend_title_font_size, fontweight=legend_title_fontweight)
            y_pos = title_y - legend_row_gap
            for ket_text, description, color in legend_entries:
                legend_ax.text(
                    0.0,
                    y_pos + legend_description_offset,
                    f"{description}:",
                    fontsize=legend_description_font_size,
                    color="#303030"
                )
                legend_ax.text(
                    0.0,
                    y_pos,
                    ket_text,
                    color=color,
                    fontsize=legend_ket_font_size,
                    family="monospace"
                )
                y_pos -= legend_row_gap

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        arrowprops = dict(arrowstyle='-', color='black', lw=0.5)

        def _connect_label(ax_obj, text_obj, y0, line_start_x):
            inv = ax_obj.transData.inverted()
            bbox = text_obj.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(inv)
            x1 = bbox_data.x0
            y1 = (bbox_data.y0 + bbox_data.y1) / 2.0
            ax_obj.annotate(  # link label to level
                
                '',
                xy=(x1, y1),
                xytext=(line_start_x, y0),
                arrowprops=arrowprops
            )

        for text_obj, y0 in main_annotations:
            _connect_label(ax, text_obj, y0, main_line_width)

        for ax_zoom, line_start_x, zoom_pairs in zoom_annotations:
            for text_obj, y0 in zoom_pairs:
                _connect_label(ax_zoom, text_obj, y0, line_start_x)

        plt.show()

        if return_fig_ax:
            return fig, ax

    def plot_driving_fields(self,
                            name: str = None,
                            function: str = 'full_waveform',
                            start: float = 0.0,
                            end: float = None):
        """Plot the pulses of the driving fields.

        Args:
            name: Name of the driving field. If not specified, all driving fields are plotted.
            function: Could be ``'full_waveform'`` or ``'complex_envelope'``.
            start: Start time of the plotted interval (in :math:`\\mu\\text{s}`).
            end: End time of the plotted interval (in :math:`\\mu\\text{s}`).

        """

        if end is None:
            end = self.last_pulse_end

        if name is None:
            name = self.driving_field_names

        if isinstance(name, str):
            name = [name]

        num_driving_fields = len(name)
        fig, ax = plt.subplots(num_driving_fields, 1, sharex=True, figsize=(10, 2.5 * num_driving_fields))

        if num_driving_fields == 1:
            ax = [ax]

        for ax1, name1 in zip(ax, name):
            self.driving_field(name1).plot_pulses(
                start=start,
                end=end,
                function=function,
                ax=ax1
            )
            ax1.tick_params(labelbottom=True)
        plt.show()


def _frame_operator(phases,
                    sigma_zs=None):
    if sigma_zs:
        s = [expm(1j * phase * unp_dynamic.asarray(sigma_z) / 2) for phase, sigma_z in zip(phases, sigma_zs)]
    else:
        sigma_z = unp_static.array([[1, 0], [0, -1]], dtype=complex)
        s = [expm(1j * phase * sigma_z / 2) for phase in phases]

    return tensorprod(*s)

def _frame_operator_diag(phases,
                         sigma_zs = None):
    if sigma_zs:
        s = [unp_dynamic.array([unp_dynamic.exp(1j * d * phase / 2) for d in unp_dynamic.diag(sigma_z)])
             for phase, sigma_z in zip(phases, sigma_zs)]
    else:
        s = [unp_dynamic.array([unp_dynamic.exp(1j * phase / 2), unp_dynamic.exp(-1j * phase / 2)])
             for phase in phases]

    return tensorprod(*s)
