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

from .config import Config, unp
from .exceptions import SimphonyError, SimphonyWarning
from .components import Spin, Interaction, StaticField, DrivingField, PulseList
from .utils import tensorprod, average_gate_fidelity_from_unitaries
from .simulationresult import SimulationResult, TimeEvolOperator, TimeEvolState

from typing import Union, List, Tuple, Dict, Optional
from warnings import warn

from copy import deepcopy
from itertools import product

import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm
from jax.scipy.linalg import expm as jexpm

import matplotlib.pyplot as plt

from qiskit_dynamics import Solver, Signal

NUM_DECORATORS_VERBOSE = 75

uarray = Union[np.ndarray, jnp.ndarray]

class Model:
    """Represent a coupled spin system.

    This class provides a high-level abstraction for simulating quantum systems consisting of multiple coupled spins.
    It supports adding spins, interactions, static and driving magnetic fields. The model allows calculation of energy
    levels and eigenstates. In addition, it provides methods for computing parameters required to define control pulses
    (e.g. splittings, Rabi-frequencies, ...). It enables the simulation of the system's time evolution under the
    influence of driving fields and additional noise. It is especially suitable for modeling systems like
    nitrogen-vacancy centers.

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

        self.spins: List[Spin] = []
        """Spins attached to the model."""

        self.interactions: List[Interaction] = []
        """Interactions attached to the model."""

        self.static_field: StaticField = StaticField()
        """Static field attached to the model."""

        self.driving_fields: List[DrivingField] = []
        """Driving fields attached to the model."""

        self.initial_state = None
        """Initial state of the simulations."""

        self.static_hamiltonian: Optional[np.ndarray] = None
        """Time-independent part of the Hamiltonian."""

        self.driving_operators: List[uarray] = []
        """Driving operators."""

        self.local_quasistatic_noise_operators: List[uarray] = []
        """Local quasistatic-noise operators."""

        self.basis: Optional[list] = None
        """Basis of the Hamiltonian."""

        self.basis_qubit_subspace: Optional[list] = None
        """Basis corresponding to the qubit subspace."""

        self.eigenbasis: Optional[np.ndarray] = None
        """Eigenbasis of the static Hamiltonian."""

        self._is_static_hamiltonian_calculated: bool = False

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'(spin_names={self.spin_names}, '
            f'driving_field_names={self.driving_field_names}, '
            f'dimension={self.dimension})'
        )

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
        return int(np.prod(self.subdimensions))

    @property
    def spin_names(self) -> List[str]:
        """Names of the spins."""
        return [spin.name for spin in self.spins]

    @property
    def driving_field_names(self) -> List[str]:
        """Names of the driving fields."""
        return [spin.name for spin in self.driving_fields]

    @property
    def rotating_frame_frequencies(self) -> uarray:
        """Rotating frame frequencies."""
        return unp.array([spin.rotating_frame_frequency for spin in self.spins])

    @property
    def virtual_phases(self)  -> uarray:
        """Virtual z-phases."""
        return unp.array([spin.virtual_phase for spin in self.spins])

    @property
    def initial_state(self) -> np.ndarray:
        """Initial state setter."""
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state):
        self._initial_state = initial_state

    @property
    def last_pulse_end(self):
        """End time of the latest pulse among all driving fields."""
        if not self.driving_fields:
            raise SimphonyError('No driving field added.')

        return unp.max(unp.array([driving_field.last_pulse_end for driving_field in self.driving_fields]))

    @property
    def pulses(self):
        """Pulses of all driving fields."""
        pulses = PulseList()
        for driving_field in self.driving_fields:
            pulses += driving_field.pulses
        return pulses

    def add_spin(self,
                 spin: Spin):
        """Add spin to the model.

        Raises:
            SimphonyError: If static Hamiltonian is already calculated.

        """

        if self._is_static_hamiltonian_calculated:
            raise SimphonyError('Static Hamiltonian is already calculated')

        if spin.name in self.spin_names:
            raise SimphonyError(f"A spin with name '{spin.name}' is already added to the model")

        self.spins += [spin]

    def add_interaction(self,
                        interaction: Interaction):
        """Add interaction to the model.

        Raises:
            SimphonyError: If static Hamiltonian is already calculated.

        """

        if self._is_static_hamiltonian_calculated:
            raise SimphonyError('Static Hamiltonian is already calculated')

        self.interactions += [interaction]

    def add_static_field(self,
                         static_field: StaticField):
        """Add static field to the model.

        Raises:
            SimphonyError: If static Hamiltonian is already calculated.

        """

        if self._is_static_hamiltonian_calculated:
            raise SimphonyError('Static Hamiltonian is already calculated')

        self.static_field = static_field

    def add_driving_field(self,
                          driving_field: DrivingField):
        """Add driving field to the model."""

        if driving_field.name in self.driving_field_names:
            raise SimphonyError(f"A driving field with name '{driving_field.name}' is already added to the model")

        self.driving_fields += [driving_field]

    def remove_all_pulses(self):
        """Remove all added pulses from all driving fields and reset the virtual z-phases to zero."""
        for driving_field in self.driving_fields:
            driving_field.remove_all_pulses()

        for spin in self.spins:
            spin.virtual_phase = 0.

    def spin(self,
             name: str) -> Spin:
        """Return the instance of the spin specified by ``name``.

        Args:
            name: Name of the spin.

        Returns:
            Spin: Instance of the specified spin.

        Raises:
            SimphonyError: If the given name is not found in :attr:`spin_names`.
        """
        if name not in self.spin_names:
            raise SimphonyError(f'Invalid name, must be in {self.spin_names}')

        idx = self.spin_names.index(name)

        return self.spins[idx]

    def project_to_qubit_subspace(self,
                                  unitary: jnp.array,
                                  only_diagonal: bool = False) -> jnp.array:
        """Project unitary to qubit subspace. Last two dimensions are used."""
        qubit_subspace_idxs = unp.array([self.basis.index(i) for i in self.basis_qubit_subspace])
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

    def rotating_frame_operator(self,
                                t: Union[float, List[float], uarray],
                                only_diag: bool = False,
                                in_qubit_subspace: bool = False,
                                inverse: bool = False) -> uarray:
        """Return the rotating frame operator:

        .. math::
            U_\\text{rot}(t) = \\prod_{i\\in\\text{spins}}e^{i 2 \\pi f_i t \\sigma_{z,i}/2},

        where :math:`f_i` are the rotating frame frequencies of the spins and :math:`\\sigma_{z,i}` are the Pauli
        z-operators. Frequencies are specified in the :attr:`Spin.rotating_frame_frequency` attribute.

        Args:
            t: Time point(s).
            only_diag: If ``True``, return only the diagonal of the operator.
            in_qubit_subspace: If ``True``, return only the operator projected to the qubit subspace.
            inverse: If ``True``, return the inverse of the operator.

        Returns:
            Rotating frame operator(s). Shape depends on the ``only_diag`` parameter and the type of the ``t`` parameter.

        """

        if inverse:
            sign = -1
        else:
            sign = 1

        if only_diag:
            rotating_frame_fn = _rotating_operator_diag_from_phases
        else:
            rotating_frame_fn = _rotating_operator_from_phases

        omegas = 2 * np.pi * self.rotating_frame_frequencies

        if in_qubit_subspace:
            def operator_(t1):
                return rotating_frame_fn(sign * omegas * t1)
        else:
            sigma_zs = [spin.operator_qubit_subspace.z for spin in self.spins]

            def operator_(t1):
                return rotating_frame_fn(sign * omegas * t1, sigma_zs)

        if isinstance(t, (float, int)):
            return operator_(t)
        else:
            return unp.array([operator_(t1) for t1 in t])

    def virtual_phases_operator(self,
                                only_diag: bool = False,
                                in_qubit_subspace: bool = False,
                                inverse: bool = False) -> uarray:
        """Return the operator corresponding the virtual z-phases:

        .. math::
            U_\\text{z-phase} = \\prod_{i\\in\\text{spins}}e^{i \\phi_i \\sigma_{z,i}/2},

        where :math:`\\phi_i` are the virtual z-phases of the spins and :math:`\\sigma_{z,i}` are the Pauli
        z-operators. Phases are specified in the :attr:`Spin.virtual_phase` attribute.

        Args:
            only_diag: If ``True``, return only the diagonal of the operator.
            in_qubit_subspace: If ``True``, return only the operator projected to the qubit subspace.
            inverse: If ``True``, return the inverse of the operator.

        Returns:
            Operator corresponding the virtual z-phases. Shape depends on the ``only_diag`` parameter.
        """

        if inverse:
            sign = -1
        else:
            sign = 1

        if only_diag:
            rotating_frame_fn = _rotating_operator_diag_from_phases
        else:
            rotating_frame_fn = _rotating_operator_from_phases

        if in_qubit_subspace:
            return rotating_frame_fn(sign * self.virtual_phases)
        else:
            sigma_zs = [spin.operator_qubit_subspace.z for spin in self.spins]
            return rotating_frame_fn(sign * self.virtual_phases, sigma_zs)

    def driving_field(self,
                      name: str) -> DrivingField:
        """Return the instance of the driving field specified by ``name``.

        Args:
            name: Name of the driving field.

        Returns:
            DrivingField: Instance of the specified driving field.

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
                raise SimphonyError('Invalid spin name: {spin_name}')

            if quantum_num not in self.spin(spin_name).quantum_nums:
                raise SimphonyError(
                    f'Invalid quantum number for {spin_name}: {quantum_num}, must be {self.spin(spin_name).quantum_nums}')

        return tuple([quantum_nums[spin_name] for spin_name in self.spin_names])

    def productstate(self,
                     quantum_nums: Union[Dict[str, float], Tuple[float, ...]]) -> np.ndarray:
        """Return the product basis state of the model corresponding to given quantum numbers of the spins.

        Args:
            quantum_nums: Quantum numbers. It can be specified by two different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.

        Returns:
            Product basis state

        """

        if isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        return np.identity(self.dimension, dtype=complex)[self._map_quantum_nums_to_idx[quantum_nums]]

    def calculate_eigenenergies_and_eigenstates(self):
        """Calculate the eigenenergies and eigenstates of the :attr:`static_hamiltonian`.

        Updates the values behind the :meth:`eigenenergy` and :meth:`eigenstate` methods.

        """

        E0s, v0s = np.linalg.eig(self.static_hamiltonian)
        E0s = E0s.real
        v0s = v0s.transpose()

        eigenenergies = []
        eigenstates = []

        for idx, quantum_nums in enumerate(self.basis):
            energy, state = max(zip(E0s, v0s), key=lambda energy_state: np.abs(energy_state[1][idx]))

            eigenenergies.append(energy)
            eigenstates.append(state)

        eigenstates = np.array(eigenstates)

        self._eigenenergies = eigenenergies
        self.eigenbasis = eigenstates
        self._map_quantum_nums_to_idx = {quantum_nums: idx for idx, quantum_nums in enumerate(self.basis)}

    def eigenenergy(self,
                    quantum_nums: Union[Dict[str, float], Tuple[float, ...]]) -> float:
        """Return the eigenenergy of the static Hamiltonian that associated with the eigenstate that has the largest
        overlap with the product basis state described by ``quantum_nums``.

        Args:
            quantum_nums: Quantum numbers. It can be specified by two different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.

        Returns:
            eigenenergy (frequency in :math:`\\text{MHz}`)

        """

        if isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        idx = self._map_quantum_nums_to_idx.get(quantum_nums)

        if idx is None:
            raise SimphonyError('Invalid quantum number provided.')

        return self._eigenenergies[idx]

    def eigenstate(self,
                   quantum_nums: Union[Dict[str, float], Tuple[float, ...]]) -> np.ndarray:
        """Return the eigenstate of the static Hamiltonian that has the largest overlap with the product basis state
        described by ``quantum_nums``.

        Args:
            quantum_nums: Quantum numbers. It can be specified by two different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.

        Returns:
            eigenstate (coefficient in the product basis)

        """

        if isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

        idx = self._map_quantum_nums_to_idx.get(quantum_nums)

        if idx is None:
            raise SimphonyError('Invalid quantum number provided.')

        return self.eigenbasis[idx]

    def state(self,
              quantum_nums: Union[Dict[str, float], Tuple[float, ...]],
              basis: str = 'product',
              coeffs_basis: str = 'product') -> np.ndarray:
        """Return the state of the model corresponding to the given quantum numbers.

        Args:
            quantum_nums: Quantum numbers. It can be specified by two different ways:

                - By a ``dict`` whose keys correspond to the spin names and values correspond to the quantum numbers.
                - By a ``tuple`` containing the quantum numbers in the same order as indicated by the :attr:`spin_names`.

            basis: Specifies that the quantum numbers correspond to the product-basis state (``'product'``) or the
                eigenstate (``'eigen'``) of the model.
            coeffs_basis: Basis in which the returned coefficients are expanded. Must be ``'product'`` or ``'eigen'``.

        Return:
            state (coefficients either in the product basis or in the eigenbasis)

        """

        if isinstance(quantum_nums, dict):
            quantum_nums = self._quantum_nums_from_dict(quantum_nums)

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
                return self.eigenbasis @ self.productstate(quantum_nums)
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
            return unp.mean(unp.array(splittings))

    def matrix_element(self,
                       driving_field_name: str,
                       state1: np.ndarray,
                       state2: np.ndarray) -> complex:
        """Return the matrix element of the time-independent operator associated with the driving field between
        two states.

        Args:
            driving_field_name: Name of the driving field.
            state1: First state.
            state2: Second state.

        Returns:
            matrix element (in :math:`\\text{MHz}`)

        """

        idx = self.driving_field_names.index(driving_field_name)
        driving_operator = self.driving_operators[idx]

        return unp.linalg.multi_dot([state1.conjugate(),
                                    driving_operator,
                                    state2])

    def _rabi_cycle_helper(self,
                           driving_field_name: str,
                           spin_name: str,
                           quantum_nums: Union[dict, tuple],
                           rest_quantum_nums: Dict[str, float]) -> complex:
        """Helper function for rabi_cycle_time and rabi_cycle_amplitude"""

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

    def rabi_cycle_time(self,
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

        matrix_element = self._rabi_cycle_helper(driving_field_name=driving_field_name,
                                                 spin_name=spin_name,
                                                 quantum_nums=quantum_nums,
                                                 rest_quantum_nums=rest_quantum_nums)

        return 1 / (amplitude * unp.abs(matrix_element))

    def rabi_cycle_amplitude(self,
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
        matrix_element = self._rabi_cycle_helper(driving_field_name=driving_field_name,
                                                 spin_name=spin_name,
                                                 quantum_nums=quantum_nums,
                                                 rest_quantum_nums=rest_quantum_nums)

        return 1 / (period_time * unp.abs(matrix_element))

    def _rabi_cycle_qubit_helper(self,
                                 function,
                                 driving_field_name: str,
                                 spin_name: str,
                                 value: float,
                                 rest_quantum_nums: Dict[str, float] = None) -> float:
        """Helper function for rabi_cycle_time_qubit and rabi_cycle_amplitude_qubit"""

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
            return unp.mean(unp.array(results))

    def rabi_cycle_time_qubit(self,
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

        return self._rabi_cycle_qubit_helper(function=self.rabi_cycle_time,
                                             driving_field_name=driving_field_name,
                                             value=amplitude,
                                             spin_name=spin_name,
                                             rest_quantum_nums=rest_quantum_nums)

    def rabi_cycle_amplitude_qubit(self,
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

        return self._rabi_cycle_qubit_helper(function=self.rabi_cycle_amplitude,
                                             driving_field_name=driving_field_name,
                                             value=period_time,
                                             spin_name=spin_name,
                                             rest_quantum_nums=rest_quantum_nums)

    def calculate_static_hamiltonian(self):
        """Calculate the static part of the Hamiltonian of the model and store it in :attr:`static_hamiltonian`. It
        corresponds to sum of the Zeeman-terms, zero-field splitting terms and intercation terms:

        .. math::
            H_\\text{static} = \\mathbf{B}_\\text{static}\\sum_{i\\in\\text{spins}}\\gamma_i\\mathbf{S}_i +
            \\sum_{i\\in\\text{spins}} D_i S_{z,i}^{2} +
            \\sum_{i,j\\in\\text{spins}}{\\mathbf{S}_{i}\\mathbf{A}\\mathbf{S}_{j}}.

        By calling the function, the tensor products of the spin operators are calculated, which fix the
        dimensionality of the model. After calling the function, adding further components is not allowed. But
        reseting or adding pulses to a specific ``driving_field`` is possible, as well as setting the strengths of the
        local quasistatic noises of the spins. The eigenenergies and eigenstates of the model are calculated here.

        """

        self.basis = list(product(*[spin.quantum_nums for spin in self.spins]))
        self.basis_qubit_subspace = list(product(*[spin.qubit_subspace for spin in self.spins]))

        self.static_hamiltonian = np.zeros((self.dimension, self.dimension), dtype=complex)

        # spins
        for idx, spin in enumerate(self.spins):
            if spin.zero_field_splitting != 0:
                ops = [spin.operator.i for spin in self.spins]
                ops[idx] = spin.operator.z ** 2

                op = tensorprod(*ops)
                strength = spin.zero_field_splitting

                self.static_hamiltonian += strength * op

        # static_fields
        for idx, spin in enumerate(self.spins):
            ops = [spin.operator.i for spin in self.spins]
            gyromagnetic_ratio = spin.gyromagnetic_ratio

            for i in range(3):
                ops[idx] = spin.operator.vec[i]
                op = tensorprod(*ops)
                component = self.static_field.vec[i]

                self.static_hamiltonian += gyromagnetic_ratio * component * op

        # interactions
        for interaction in self.interactions:

            idx1 = self.spins.index(interaction.spin1)
            idx2 = self.spins.index(interaction.spin2)

            for i in range(3):
                for j in range(3):
                    ops = [spin.operator.i for spin in self.spins]
                    ops[idx1] = interaction.spin1.operator.vec[i]
                    ops[idx2] = interaction.spin2.operator.vec[j]

                    op = tensorprod(*ops)
                    coeff = interaction.tensor[i][j]

                    self.static_hamiltonian += coeff * op

        self.calculate_eigenenergies_and_eigenstates()

        self._is_static_hamiltonian_calculated = True

    def calculate_driving_operators(self):
        """Calculate the operators of the driving fields. Driving operators are the time-independent operator parts of
        the driving Hamiltonian, whereas pulses are the time-dependent scalar parts:
        
        .. math::
            H_\\text{driving} = \\sum_{k\\in\\text{driving fields}}
            \\underbrace{B_{k}(t)}_{\\text{pulse}}
            \\underbrace{\\left[
            \\sum_{i\\in\\text{spins}}
            \\gamma_i\\hat{\\mathbf{u}}_{k}
            \\cdot
            \\mathbf{S}_{i}
            \\right]}_{\\text{driving operator}},


        where :math:`B_{k}(t)` and :math:`\\hat{\\mathbf{u}}_k` are the strength and direction vector of the
        :math:`k\\text{th}` driving field.

        """


        empty = np.zeros((self.dimension, self.dimension), dtype=complex)
        self.driving_operators = [empty.copy() for _ in self.driving_fields]

        for idx, driving_field in enumerate(self.driving_fields):

            for spin_idx, spin in enumerate(self.spins):
                ops = [spin.operator.i for spin in self.spins]
                gyromagnetic_ratio = spin.gyromagnetic_ratio

                for i in range(3):
                    ops[spin_idx] = spin.operator.vec[i]
                    op = tensorprod(*ops)
                    component = driving_field.vec[i]

                    self.driving_operators[idx] += gyromagnetic_ratio * component * op

    def calculate_local_quasistatic_noise_operators(self):
        """Calculate the operators associated with the local quasistatic noises. These operators represent the
        time-independent part of the noise Hamiltonian, while their strengths represent the shot-dependent scalar
        parts:

        .. math::
            H_\\text{noise} = \\sum_{i\\in\\text{spins}}
            \\overbrace{\\Delta\\omega_{i}}^{\\rlap{\\text{local quasistatic noise strength}}}
            \\underbrace{\\hat{\\mathbf{v}}_{i}
            \\cdot
            \\mathbf{S}_{i}}_{\\rlap{\\text{local quasistatic noise operator}}},

        where :math:`\\Delta\\omega_{i}` and :math:`\\hat{\\mathbf{v}}_{i}` are the strength and direction vector
        associated with the local quasistatic noise on the :math:`i\\text{th}` spin.

        """

        self.local_quasistatic_noise_operators = []

        for idx, spin in enumerate(self.spins):
            operators_per_spin = []
            for i in range(3):
                ops = [spin.operator.i for spin in self.spins]
                ops[idx] = spin.operator.vec[i]
                op = tensorprod(*ops)
                operators_per_spin.append(op)
            self.local_quasistatic_noise_operators.append(operators_per_spin)


    def calculate_hamiltonians(self):
        """Calculate the static Hamiltonian, the driving operators and the local quasistatic noise operators."""
        self.calculate_static_hamiltonian()
        self.calculate_driving_operators()
        self.calculate_local_quasistatic_noise_operators()

    def simulate_time_evolution(self,
                                n_shots: int = 1,
                                start: float = 0.,
                                end: Optional[float] = None,
                                apply_noise: bool = True,
                                random_seed: int = None,
                                simulation_method: str = 'single_sine_wave',
                                n_eval: int = 251,
                                n_split: int = 250,
                                state_only: bool = False,
                                solver_method: Optional[str] = None,
                                verbose: bool = False,
                                test_convergence: Union[bool, int] = False,
                                _time_segment_sequence = None) -> SimulationResult:

        """Simulate the time evolution of the model due to the driving fields under the effect of the noise.

        Args:
            n_shots: Number of the shots.
            start: Start time of the simulation (in :math:`\\mu\\text{s}`, defaults to 0).
            end: End time of the simulation (in :math:`\\mu\\text{s}`, defaults to end of the last pulse).
            apply_noise: Whether to include noise or not.
            random_seed: Seed of the random number generator for noise randomization.
            simulation_method: It must be ``'basic'`` or ``'single_sine_wave'``. Latter simulates only a single sine
                wave in the case of a monochromatic excitation that could accelerate the simulation.
            n_eval: Number of evaluation points per time segments.
            n_split: Number of split points per time segments.
            state_only: Whether to simulate the time evolution from an initial state or simulate the full propagators.
            solver_method: Method of the ``qiskit_dynamics`` solver that used to simulate the time evolution (defaults to
                ``'jax_expm'`` in the case of using CPU and ``'jax_expm_parallel'`` in the case of using GPU)
            verbose: Whether to verbose the details of the simulation or not.
            test_convergence: Whether to test the convergence of the final propagators by halving the time steps. If an
                ``int`` provided, it specifies the number of times the convergence test is repeated.

        Return:
            Results of the simulation, which contains the propagators/states of the simulation, a copy of the simulated
            model, the timestamps of the simulation and the noise realizations corresponding to the different shots.

        """

        if self._is_static_hamiltonian_calculated is False:
            raise SimphonyError("Static Hamiltonian is not calculated")

        if end is None:
            end = self.last_pulse_end

        if not start < end:
            raise SimphonyError("end must be greater than start")

        if verbose:
            print(f'start = {start}\nend = {end}')

        # set default method
        if solver_method is None:
            if Config.get_platform() == 'cpu':
                solver_method = 'jax_expm'
            elif Config.get_platform() == 'gpu':
                solver_method = 'jax_expm_parallel'
            else:
                raise SimphonyError("Invalid backend, must be 'cpu' or 'gpu'")
        elif solver_method not in ['jax_expm', 'jax_expm_parallel']:
            raise SimphonyError("Invalid solver_method, must be 'jax_expm' or 'jax_expm_parallel'")

        if verbose:
            print(f'solver_method = {solver_method}')

        # state only mode
        if state_only is True:
            if self.initial_state is None:
                raise SimphonyError('Initial state is not initialized')
            if simulation_method != 'basic':
                warn("If state_only is True, simulation_segment_method is set to 'basic'", SimphonyWarning, stacklevel=2)
                simulation_method = 'basic'
            y0s = list(np.repeat(np.array([self.initial_state], dtype=complex), repeats=n_shots, axis=0))
        else:
            y0s = list(np.repeat([np.identity(self.dimension, dtype=complex)], repeats=n_shots, axis=0))
        self._y0s = y0s

        # non-zero driving fields
        nonzero_driving_field_names = []
        nonzero_driving_operators = []
        for driving_field, driving_operator in zip(self.driving_fields, self.driving_operators):
            if len(driving_field.pulses) > 0:
                nonzero_driving_field_names.append(driving_field.name)
                nonzero_driving_operators.append(driving_operator)

        # non-zero noises
        if apply_noise:
            all_noise_strengths = unp.array([spin.local_quasistatic_noise.vec for spin in self.spins])
            idx = all_noise_strengths.nonzero()
            nonzero_noise_strengths = all_noise_strengths[idx]
            if len(nonzero_noise_strengths.tolist()) == 0:
                nonzero_noise_operators = []
            else:
                nonzero_noise_operators = list(unp.array(self.local_quasistatic_noise_operators)[idx])
        else:
            nonzero_noise_strengths = []
            nonzero_noise_operators = []

        if verbose:
            print(f'number of simulated driving terms = {len(nonzero_driving_operators)}', )
            print(f'number of simulated noise terms = {len(nonzero_noise_operators)}')

        # randomize noise for different shots
        np.random.seed(random_seed)
        shot_noises = np.random.randn(n_shots, len(nonzero_noise_strengths)) * nonzero_noise_strengths
        signals_noise = []
        for shot1_noises in shot_noises:
            signals1_noise = [Signal(unp.array(shot1_noise)) for shot1_noise in shot1_noises]
            signals_noise.append(signals1_noise)
        self._signals_noise = signals_noise

        # set the solver
        hamiltonian_operators = nonzero_driving_operators + nonzero_noise_operators
        self._hamiltonian_operators = hamiltonian_operators
        solver = Solver(
            static_hamiltonian=2 * np.pi * self.static_hamiltonian,
            hamiltonian_operators=[2 * np.pi * H_op for H_op in hamiltonian_operators] if hamiltonian_operators else None,
            array_library="jax"
        )
        self._solver = solver

        # time segments and time discretization
        if _time_segment_sequence is None:
            _time_segment_sequence = self.pulses.convert_to_time_segment_sequence(
                start=start,
                end=end
            )
            _time_segment_sequence.discretize_simulation(
                method=simulation_method,
                included_driving_field_names=nonzero_driving_field_names,
                n_eval=n_eval,
                n_split=n_split
            )

        self._time_segment_sequence = _time_segment_sequence

        if verbose:
            print('-' * NUM_DECORATORS_VERBOSE)

        ts = []
        solutions = []
        for idx, time_segment in enumerate(_time_segment_sequence):
            if time_segment.simulation_type == 'basic':
                simulate_time_segment = self._simulate_time_segment_basic
            elif time_segment.simulation_type == 'single_sine_wave':
                simulate_time_segment = self._simulate_time_segment_single_sine_wave
            else:
                raise SimphonyError('Unknown simulation_type')

            if verbose:
                _prec = max(-int(np.floor(np.log10(time_segment.duration))) + 2, 0)
                _start, _end = time_segment.simulation_t_span
                _max_dt = time_segment.simulation_max_dt
                _type = time_segment.simulation_type
                print(f'simulate time segment [{_start:.{_prec}f}, {_end:.{_prec}f}] with step size {_max_dt:.4g} (type: {_type})')

            solution, y0s = simulate_time_segment(
                solver=solver,
                time_segment=time_segment,
                signals_noise=signals_noise,
                method=solver_method,
                y0s=y0s
            )

            solutions.append(solution)
            ts.append(time_segment.simulation_t_eval)

        ts = unp.concatenate([elem[1:] if idx > 0 else elem for idx, elem in enumerate(ts)])
        solutions = unp.concatenate([elem[:, 1:] if idx > 0 else elem for idx, elem in enumerate(solutions)], axis=1)

        self._ts = ts
        self._solutions = solutions

        model_copy = deepcopy(self)
        if state_only is True:
            simulation_result = SimulationResult(
                model=model_copy,
                ts=ts,
                time_evol_state=TimeEvolState(array=solutions,
                                              ts=ts,
                                              model=model_copy),
                shot_noises=shot_noises
            )
        else:
            simulation_result = SimulationResult(
                model=model_copy,
                ts=ts,
                time_evol_operator=TimeEvolOperator(array=solutions,
                                                    ts=ts,
                                                    model=model_copy),
                shot_noises=shot_noises
            )

        # test the convergence
        if test_convergence:

            if state_only:
                raise SimphonyError('Convergence test is not compatible with state only mode')

            if len(nonzero_noise_strengths) > 0 and random_seed is None:
                warn("Convergence test without setting random_seed", SimphonyWarning, stacklevel=2)

            final_time_evol_operator_1 = simulation_result.time_evol_operator.matrix()[:, -1]

            num_halvings = test_convergence if isinstance(test_convergence, int) else 1
            for _ in range(num_halvings):

                if verbose:
                    print('#' * NUM_DECORATORS_VERBOSE)
                    print('step-halving')
                    print('-' * NUM_DECORATORS_VERBOSE)

                for time_segment in _time_segment_sequence:
                    time_segment.simulation_max_dt /= 2

                simulation_result_check = self.simulate_time_evolution(
                    n_shots=n_shots,
                    start=start,
                    end=end,
                    apply_noise=apply_noise,
                    random_seed=random_seed,
                    solver_method=solver_method,
                    verbose=verbose,
                    test_convergence=False,
                    _time_segment_sequence=_time_segment_sequence,
                )

                final_time_evol_operator_2 = simulation_result_check.time_evol_operator.matrix()[:, -1]
                average_gate_infidelities = [1 - average_gate_fidelity_from_unitaries(U1, U2) for U1, U2 in
                                             zip(final_time_evol_operator_1, final_time_evol_operator_2)]
                final_time_evol_operator_1 = final_time_evol_operator_2

                if verbose:
                    print('-' * NUM_DECORATORS_VERBOSE)

                print(f'infidelity of the worst shot after step-halving : {np.max(average_gate_infidelities)}')

        return simulation_result

    @staticmethod
    def _simulate_time_segment_basic(solver,
                                     time_segment,
                                     signals_noise,
                                     method,
                                     y0s):

        if solver.model.operators is None:
            signals = None
        else:
            signals = [time_segment.simulation_signals + noise_signals1 for noise_signals1 in signals_noise]

        sol = solver.solve(
            t_span=time_segment.simulation_t_span,
            y0=y0s,
            signals=signals,
            t_eval=time_segment.simulation_t_eval,
            method=method,
            max_dt=time_segment.simulation_max_dt
        )

        solution = unp.array([sol1.y for sol1 in sol])
        y0s = list(solution[:, -1])

        return solution, y0s

    @staticmethod
    def _simulate_time_segment_single_sine_wave(solver,
                                                time_segment,
                                                signals_noise,
                                                method,
                                                y0s):

        y0s = unp.array(y0s)
        y0s_identity = list(np.repeat([np.identity(y0s.shape[-1], dtype=complex)], repeats=y0s.shape[0], axis=0))

        sol = solver.solve(
            t_span=time_segment.simulation_t_span_projected_sorted,
            y0=y0s_identity,
            signals=[time_segment.simulation_signals + noise_signals1 for noise_signals1 in signals_noise],
            t_eval=time_segment.simulation_t_eval_projected_sorted,
            method=method,
            max_dt=time_segment.simulation_max_dt
        )

        solution_projected_sorted = unp.array([sol1.y for sol1 in sol])

        unsort_indices = time_segment.simulation_unsort_indices
        solution_projected = solution_projected_sorted[:, unsort_indices]
        U_single_sine = solution_projected_sorted[:, -1]

        cycle_indices = time_segment.simulation_cycle_indices
        U_single_sine_powers = unp.stack(unp.array([unp.linalg.matrix_power(U_single_sine, int(i)) for i in cycle_indices]), axis=1) # slow?

        solution = unp.einsum('stij,stjk,skl->stil', solution_projected, U_single_sine_powers, y0s)

        y0s = list(solution[:, -1])

        return solution, y0s

    def plot_levels(self,
                    height: float = 4,
                    width: float = 1):
        """Plot the energy levels of the model. The main plot shows the full spectrum, the subplots show the
        fine structure corresponding to the nuclear spins.

        Note:
            Currently it works only for one NV electron that must be the `first` spin of the model.
        
        Args:
            height: Height of the main plot.
            width: Width of the main plot.

        """

        def format_label(label):
            is_in_qubit_subspace = np.any(np.all(np.equal(np.array(self.basis_qubit_subspace), label), axis=1))
            out = []
            for i in label:
                if i.is_integer():
                    out.append(str(int(i)).rjust(2))
                else:
                    out.append(str(int(2 * i)).rjust(2) + '/2')

            return '|' + ','.join(out) + '⟩' + (' -' if is_in_qubit_subspace else '')

        energies = np.array(self._eigenenergies)
        energies_by_e = energies.reshape(3, -1)
        energies_by_e_max = np.max(energies_by_e, axis=1)
        labels_by_e = np.array(self.basis).reshape(3, -1, self.n_spins)
        labels_only_e = labels_by_e[:, 0, 0]

        if self.n_spins > 1:
            energies_ptps = np.array([np.ptp(i) for i in energies_by_e])
            ratios = energies_ptps / np.linalg.norm(energies_ptps)

        level_width = 1
        gap = 0.2
        label_width = (1 + width) * (2 * gap + level_width) - gap - level_width

        fig, ax = plt.subplots(figsize=(1, height))

        ax.plot([0, level_width], [energies, energies], c='k', linewidth=1)
        ax.set_xlim(-gap, level_width + gap)
        ax.set_ylabel('Energy (MHz)')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=8)
        for energy_by_e_max, label_only_e in zip(energies_by_e_max, labels_only_e):
            ax.text(level_width / 2,
                    energy_by_e_max,
                    format_label([label_only_e]),
                    va='bottom',
                    ha='center',
                    fontdict={'fontsize': 8})

        if self.n_spins > 1:

            sort_e_energies = np.argsort(np.mean(energies_by_e, axis=1))
            for idx, i in enumerate(sort_e_energies):
                axin = ax.inset_axes([2 + (2 + width) * idx, 0.5 - ratios[i] / 2, 1 + width, ratios[i]])
                axin.plot([0, level_width], [energies_by_e[i], energies_by_e[i]], c='k', linewidth=0.5)
                axin.set_xlim(-gap, label_width + level_width)
                axin.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                axin.tick_params(axis='y', labelsize=8)
                axin.set_title(format_label([labels_only_e[i]]))

                for energy, label in zip(energies_by_e[i], labels_by_e[i]):
                    axin.text(level_width + gap, energy, format_label(label), va='center', fontdict={'fontsize': 6})

        plt.show()

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


def _rotating_operator_from_phases(phases,
                                   sigma_zs = None):
    if sigma_zs:
        if unp.__name__ == 'numpy':
            s = [expm(1j * phase * sigma_z / 2) for phase, sigma_z in zip(phases, sigma_zs)]
        else:
            s = [jexpm(1j * phase * sigma_z / 2, max_squarings=64) for phase, sigma_z in zip(phases, sigma_zs)]
    else:
        sigma_z = unp.array([[1, 0], [0, -1]], dtype=complex)
        if unp.__name__ == 'numpy':
            s = [expm(1j * phase * sigma_z / 2) for phase in phases]
        else:
            s = [jexpm(1j * phase * sigma_z / 2, max_squarings=64) for phase in phases]

    return tensorprod(*s)

def _rotating_operator_diag_from_phases(phases,
                                        sigma_zs = None):
    if sigma_zs:
        s = [unp.array([unp.exp(1j * d * phase / 2) for d in unp.diag(sigma_z)]) for phase, sigma_z in zip(phases, sigma_zs)]
    else:
        s = [unp.array([unp.exp(1j * phase / 2), unp.exp(-1j * phase / 2)]) for phase in phases]

    return tensorprod(*s)