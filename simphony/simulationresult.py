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

"""Result of the simulation and objects for analysis."""
from numpy.matrixlib.defmatrix import matrix

from .config import unp
from .exceptions import SimphonyError
from .utils import partial_trace, average_gate_fidelity_from_unitaries, leakage_from_time_evolution_operator

from typing import Union, List, Optional

from itertools import product

import numpy as np
import jax.numpy as jnp

from qiskit.quantum_info import Pauli, pauli_basis, Operator as qiskit_Operator
from qiskit import QuantumCircuit

import matplotlib.pyplot as plt

uarray = Union[np.ndarray, jnp.ndarray]


class Operator:
    """Represent an operator.

    Initialize an instance of the ``Operator`` class.

    Args:
        operator: Must be

            - ``ndarray|Array``: The operator as a matrix itself.
            - ``str``: Pauli bitstring (sequence of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``).

    """

    def __init__(self,
                 operator: Union[uarray, str]):

        self.name: Optional[str] = None
        """Name of the operator."""

        self.matrix: uarray
        """Matrix representation of the operator."""

        if isinstance(operator, (np.ndarray, jnp.ndarray)):
            if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
                raise SimphonyError('operator must be a square array')
            self.matrix = unp.array(operator)

        elif isinstance(operator, str):
            self._operator = Pauli(operator)
            self.name = self._operator.to_label()
            self.matrix = unp.array(self._operator.to_matrix())
        else:
            raise SimphonyError('Invalid operator')

    def __repr__(self):
        if hasattr(self, 'name') and self.name is not None:
            return f"{type(self).__name__}(name='{self.name}')"
        else:
            return f"{type(self).__name__}(shape={self.matrix.shape})"

    def expectation_value(self,
                          state: uarray) -> uarray:
        """Return the expectation value of the operator with respect to the state. If multiple states are given, then
        return multiple expectation values (expectation values are calculated with respect to the last axis).

        Args:
            state: State or states.

        Returns:
            Expectation value or expectation values.

        """

        if state.shape[-1] != self.matrix.shape[0]:
            raise SimphonyError(f"Incompatible state and operator: state has dimension {state.shape[-1]}, but operator expects dimension {self.matrix.shape[0]}")

        exp_value = unp.real(unp.einsum('...i,ij,...j->...', state.conjugate(), self.matrix, state))

        return exp_value


class _TimeEvolBase:
    def __init__(self,
                 array,
                 ts,
                 model):
        self._array = unp.array(array)
        self.ts = unp.array(ts)
        self.model = model

    def __repr__(self):
        n_shots, n_ts, dimension = self._array.shape[:3]
        return f'{type(self).__name__}(dimension={dimension}, n_shots={n_shots}, n_ts={n_ts})'

    def _take_shot_and_t_idx(self, shot, t_idx):
        output = self._array
        ts = self.ts

        if shot != 'all':
            if isinstance(shot, str):
                raise SimphonyError("shot must be 'all', int(s)")
            if isinstance(shot, int):
                shot = [shot]
            output = unp.take(output, unp.array(shot), axis=0)

        if t_idx != 'all':
            if isinstance(t_idx, str):
                raise SimphonyError("t_idx must be 'all' or int(s)")
            if isinstance(t_idx, int):
                t_idx = [t_idx]
            t_idx = unp.array(t_idx)
            output = unp.take(output, t_idx, axis=1)
            ts = unp.take(ts, t_idx)

        return output, ts


class TimeEvolOperator(_TimeEvolBase):
    """Represent time evolution operator.

    Initialize an instance of the ``TimeEvolOperator`` class.

    Args:
        array: Time evolution operators for each shot and time point. Array of shape (`n_shots`, `n_times`, `dim`, `dim`).
        ts: Timestamps (in :math:`\\mu\\text{s}`).
        model: Spin model.

    Note:

        ``TimeEvolOperator`` must initialize in the product basis and the lab frame.

    """

    def __init__(self,
                 array: uarray,
                 ts: uarray,
                 model):
        super().__init__(array,
                         ts,
                         model)

    def matrix(self,
               shot: Union[str, int, List[int]] = 'all',
               t_idx: Union[str, int, List[int]] = 'all',
               basis: str = 'product',
               frame: str = 'lab',
               in_qubit_subspace: bool = False,
               remove_virtual_phase: bool = False) -> uarray:
        """Return the time evolution operator in a specific basis and frame.

        Args:
            shot: Specify the shot by its index or indices, or use ``'all'``.
            t_idx: Specify the time by its index or indices, or use ``'all'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            in_qubit_subspace: If ``True``, operator is projected to qubit subspace.
            remove_virtual_phase: If ``True``, virtual z-phases are transformed out from the operator.

        Return:
            Time evolution operators for shots and time points. Array of shape (`n_shots`, `n_times`, `dim`, `dim`).

        Hint:
            Use it, for example, as:

            .. code-block:: python

                result.time_evol_operator.matrix(shot=0, t_idx=[0,-1], basis='eigen', frame='rotating')

        """

        if basis not in ['product', 'eigen']:
            raise SimphonyError("basis must be 'product' or 'eigen'.")

        if frame not in ['lab', 'rotating']:
            raise SimphonyError("frame must be 'lab' or 'rotating'.")

        operator, ts = self._take_shot_and_t_idx(shot, t_idx)

        if basis == 'eigen':
            eigenbasis = self.model.eigenbasis
            operator = unp.einsum('ij,stjk,lk->stil', eigenbasis.conjugate(), operator, eigenbasis)

        if in_qubit_subspace:
            operator = self.model.project_to_qubit_subspace(operator)

        if frame == 'rotating':
            inv_rotating_frame_operator = self.model.rotating_frame_operator(ts,
                                                                             inverse=True,
                                                                             in_qubit_subspace=in_qubit_subspace)
            operator = unp.einsum('tij,stjk->stik', inv_rotating_frame_operator, operator)

        if remove_virtual_phase:
            inv_virtual_phases_operator = self.model.virtual_phases_operator(inverse=True,
                                                                             in_qubit_subspace=in_qubit_subspace)
            operator = unp.einsum('ij,stjk->stik', inv_virtual_phases_operator, operator)

        return operator


class TimeEvolState(_TimeEvolBase):
    """Represent time evolution of a state.

    Initialize an instance of the ``TimeEvolState`` class.

    Args:
        array: Time evolution states for each shot and time point. Array of shape (`n_shots`, `n_times`, `dim`).
        ts: Timestamps (in :math:`\\mu\\text{s}`).
        model: Spin model.

    """

    def __init__(self,
                 array: uarray,
                 ts: uarray,
                 model):
        super().__init__(array,
                         ts,
                         model)

    def vector(self,
               shot: Union[str, int, List[int]] = 'all',
               t_idx: Union[str, int, List[int]] = 'all',
               basis: str = 'product',
               frame: str = 'lab',
               in_qubit_subspace: bool = False) -> uarray:
        """Return the time-evolved state in a specific basis and frame.

        Args:
            shot: Specify the shot by its index or indices, or use ``'all'``.
            t_idx: Specify the time by its index or indices, or use ``'all'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            in_qubit_subspace: If ``True``, state is projected to qubit subspace.

        Return:
            Time evolution states for shots and time points. Array of shape (`n_shots`, `n_times`, `dim`).

        Hint:
            Use it, for example, as:

            .. code-block:: python

                result.time_evol_state.vector(shot=0, t_idx=[0,-1], basis='eigen', frame='rotating')

        """

        if basis not in ['product', 'eigen']:
            raise SimphonyError("basis must be 'product' or 'eigen'.")

        if frame not in ['lab', 'rotating']:
            raise SimphonyError("frame must be 'lab' or 'rotating'.")

        vector, ts = self._take_shot_and_t_idx(shot, t_idx)

        if basis == 'eigen':
            eigenbasis = self.model.eigenbasis
            vector = unp.einsum('ij,stj,j->sti', eigenbasis, vector)

        if in_qubit_subspace:
            vector = self.model.project_to_qubit_subspace(vector, only_diagonal=True)

        if frame == 'rotating':
            inv_rotating_frame_operator = self.model.rotating_frame_operator(
                ts,
                inverse=True,
                in_qubit_subspace=in_qubit_subspace
            )
            vector = unp.einsum('tij,stj->sti', inv_rotating_frame_operator, vector)

        return vector


class SimulationResult:
    """Represent the results of the simulation.

    Initialize a container that stores the results of a simulation.

    Args:
        model: Central-spin register model.
        ts: Timestamps of the stored simulation data (in :math:`\\mu\\text{s}`).
        time_evol_operator: Time evolution operator.
        time_evol_state: States corresponding the time evolution.
        shot_noises: Noise realizations of the simulation.

    """

    def __init__(self,
                 model,
                 ts: Union[List[float], uarray],
                 time_evol_operator: Optional[TimeEvolOperator] = None,
                 time_evol_state: Optional[TimeEvolState] = None,
                 shot_noises: uarray = None):

        self.model = model
        """Spin model. Recommended to be a deepcopy of the model at the time of simulation."""

        self.ts: uarray = unp.array(ts)
        """Timestamps of the stored simulation data (in :math:`\\mu\\text{s}`)."""

        self.n_ts = len(ts)
        """Number of the stored timestamps."""

        self.time_evol_operator: Optional[TimeEvolOperator] = time_evol_operator
        """Time evolution operator."""

        self.time_evol_state: Optional[TimeEvolState] = time_evol_state
        """States corresponding to the time evolution of an initial state. When set :attr:`initial_state` to a new value
        and the :attr:`time_evol_operator` is present, then it is updated automatically."""

        self.shot_noises: uarray = shot_noises
        """Shot noise realizations."""

        self.initial_state: Optional[uarray] = None

        self.n_shots: int
        """Number of the shots."""

        if shot_noises is not None:
            self.n_shots = len(shot_noises)
        else:
            self.n_shots = 1

        self.ideal: Optional[QuantumCircuit] = None

        self.ideal_unitary_matrix: Optional[uarray] = None

    def __repr__(self):
        dimension = self.model.dimension
        return f'{type(self).__name__}(model_dimension={dimension}, n_shots={self.n_shots}, n_ts={self.n_ts})'

    @property
    def initial_state(self):
        """Initial state. When set to a new value and the :attr:`time_evol_operator` is present, then it updates the
        :attr:`time_evol_state`.

        Args:
            initial_state (np.ndarray): Initial state.

        Hint:
            :meth:`productstate`, :meth:`eigenstate`, :meth:`state` methods of the :class:`model` can be used to set
            the initial states:

            .. code-block:: python

                >>> result.initial_state = model.state({'e': 0, 'N': 0, 'C': -1/2}, basis = 'eigen')
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self,
                      initial_state: np.ndarray):
        self._initial_state = initial_state

        if initial_state is not None and self.time_evol_operator is not None:
            states = np.einsum('stij,j->sti', self.time_evol_operator._array, initial_state)

            self.time_evol_state = TimeEvolState(array=states,
                                                 ts=self.ts,
                                                 model=self.model)

    @property
    def ideal(self):
        """Ideal circuit for calculating the average gate fidelity. It calculates the ideal unitary matrix stored in
        :attr:`ideal_unitary_matrix`.

        Args:
            ideal (:class:`qiskit.circuit.QuantumCircuit`): The ideal quantum circuit.

        Hint:
            Use the :class:`qiskit.circuit.QuantumCircuit` class, for example, as:

            .. code-block:: python

                >>> qc = QuantumCircuit(2)
                >>> qc.rx(np.pi, 0)
                >>> result.ideal = qc

        """

        return self._ideal

    @ideal.setter
    def ideal(self,
              ideal: Optional[QuantumCircuit]):
        if isinstance(ideal, QuantumCircuit):
            ideal_unitary_matrix = unp.array(qiskit_Operator(ideal.reverse_bits()).to_matrix())

            if ideal_unitary_matrix.shape[0] != 2**self.model.n_spins:
                raise SimphonyError('ideal circuit size is not compatible with the number of the spins')

            self._ideal = ideal
            self._ideal_unitary_matrix = ideal_unitary_matrix
        elif ideal is None:
            self._ideal = None
            self._ideal_unitary_matrix = None
        else:
            raise SimphonyError('ideal must be a QuantumCircuit')

    @property
    def ideal_unitary_matrix(self):
        """Ideal unitary matrix, defined in the qubit subspace, to calculate the average gate fidelity.

        Args:
            ideal_unitary_matrix: Ideal unitary matrix.

        Hint:
            Use the :attr:`ideal` to set it.

        """

        return self._ideal_unitary_matrix

    @ideal_unitary_matrix.setter
    def ideal_unitary_matrix(self,
              ideal_unitary_matrix: Optional[uarray]):
        if isinstance(ideal_unitary_matrix, (np.ndarray, jnp.ndarray)):
            if (
                    ideal_unitary_matrix.ndim != 2 or
                    ideal_unitary_matrix.shape[0] != ideal_unitary_matrix.shape[1] or
                    ideal_unitary_matrix.shape[0] != 2 ** self.model.n_spins
            ):
                raise SimphonyError('ideal_unitary_matrix must be a square matrix compatible with the dimension of the qubit subspace')

            self._ideal = None
            self._ideal_unitary_matrix = ideal_unitary_matrix
        elif ideal_unitary_matrix is None:
            self._ideal = None
            self._ideal_unitary_matrix = None
        else:
            raise SimphonyError('ideal_unitary_matrix must be a np.array or jnp.array')

    def expectation_value(self,
                          operator: Union[Operator, uarray, str, List[Union[Operator, uarray, str]]],
                          shot: Union[str, int, List[int]] = 'avg',
                          t_idx: Union[str, int, List[int]] = 'all',
                          basis: str = 'product',
                          frame: str = 'rotating',
                          in_qubit_subspace: bool = True) -> Union[uarray, List[uarray]]:
        """Return the expectation values of a given operator or operators.

        Args:
            operator: :class:`Operator` or input of :class:`Operator`. Could be a list.
            shot: Specify the shot by its index or indices, or use ``'avg'``.
            t_idx: Specify the time by its index or indices, or use ``'all'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            in_qubit_subspace: Specify whether the operator defined in the qubit subspace.

        Returns:
            Expectation values for shots and time points. Array of shape (`n_shots`, `n_times`, `n_operators`).

        Raises:
            SimphonyError: If the initial state is undefined.

        Note:
            If the operator is given as a Pauli string, :attr:`in_qubit_subspace` must be ``True``.

        """

        if self.initial_state is None:
            raise SimphonyError('Initial state is undefined.')

        if shot == 'avg':
            shot_ = 'all'
        else:
            shot_ = shot

        state = self.time_evol_state.vector(shot=shot_,
                                            t_idx=t_idx,
                                            basis=basis,
                                            frame=frame,
                                            in_qubit_subspace=in_qubit_subspace)

        if not isinstance(operator, list):
            operator = [operator]

        exp_value_list = []
        for operator1 in operator:
            if not isinstance(operator1, Operator):
                operator1 = Operator(operator1)
            exp_value = operator1.expectation_value(state)

            exp_value_list.append(exp_value)

        exp_value = unp.array(exp_value_list)
        exp_value = unp.moveaxis(exp_value, 0, -1)

        if shot == 'avg':
            exp_value = unp.mean(exp_value, axis=0, keepdims=True)

        return exp_value

    def plot_expectation_value(self,
                               operator: Union[Operator, uarray, str, List[Union[Operator, uarray, str]]],
                               shot: Union[str, int] = 'avg',
                               basis: str = 'product',
                               frame: str = 'rotating',
                               title: str = None,
                               ax: plt.axis = None,
                               in_qubit_subspace: bool = True):
        """Plot the expectation values of the operators corresponding the time evolved states from an initial state.

        Args:
            operator: :class:`Operator` or input of :class:`Operator`. Could be a list.
            shot: Specify the shot by its index, or use ``'avg'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            title: Title of the plot (optional)
            ax: Matplotlib axis for customization (optional)
            in_qubit_subspace: Specify whether the operator defined in the qubit subspace.

        Note:
            If the operator is given as Pauli string, :attr:`in_qubit_subspace` must be ``True``.

        """

        if isinstance(shot, str) and shot != 'avg':
            raise SimphonyError("shot must be an int or 'avg'")

        if not isinstance(operator, list):
            operator = [operator]

        exp_value = self.expectation_value(operator=operator,
                                           shot=shot,
                                           t_idx='all',
                                           basis=basis,
                                           frame=frame,
                                           in_qubit_subspace=in_qubit_subspace)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2))

        for idx in range(len(operator)):

            if isinstance(operator[idx], str):
                label = '$\\langle {} \\rangle$'.format(operator[idx])
            elif isinstance(operator[idx], Operator) and operator[idx].name:
                label = operator[idx].name
            else:
                label = 'operator_' + str(idx + 1)

            ax.plot(self.ts, exp_value[0,:,idx], label = label)

        ax.set_ylabel('Exp. value')
        ax.set_xlabel('t [$\mu$s]')
        ax.legend()

        if title is not None:
            ax.set_title(title)

        ax.grid()
        ax.set_ylim(-1.075, 1.075)

        plt.plot()

    def plot_Bloch_vectors(self,
                           spin_names: Union[str, List[str]] = None,
                           shot: Union[int, str] = 'avg',
                           basis: str = 'product',
                           frame: str = 'rotating'):
        """Plot the Bloch vectors corresponding the spins.

        Args:
            spin_names: Names of the spins (defaults to plot all spins).
            shot: Specify the shot by its index, or use ``'avg'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.

        """

        if spin_names is None:
            spin_names = self.model.spin_names
        elif isinstance(spin_names, str):
            spin_names = [spin_names]
            if not set(spin_names).issubset(set(self.model.spin_names)):
                raise SimphonyError(f'Invalid spin names, must be a subset of {self.model.spin_names}')

        n_spins = self.model.n_spins
        for spin_name in spin_names:
            idx = self.model.spin_names.index(spin_name)
            operator = ['I' * idx + op + 'I' * (n_spins - 1 - idx) for op in ['X', 'Y', 'Z']]
            self.plot_expectation_value(operator=operator,
                                        shot=shot,
                                        basis=basis,
                                        frame=frame,
                                        title=f'Components of the Bloch vector ({spin_name})')

    def process_matrix(self,
                       spin_names: Union[str, List[str]] = None,
                       shot: Union[str, int, List[int]] = 'avg',
                       t_idx: int = -1,
                       basis: str = 'product',
                       frame: str = 'rotating',
                       remove_virtual_phase: bool = True) -> uarray:
        """Return the process matrix corresponding to the time evolution operator.

        Args:
            spin_names: Names of the selected spins (defaults to all spins).
            shot: Specify the shot by its index or indices, or use ``'all'`` or ``'avg'``.
            t_idx: Specify the time by its index (defaults to last timestamp).
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            remove_virtual_phase: Specify to remove the virtual z-phases.

        Returns:
            Process matrix or process matrices.

        """

        if spin_names is None:
            spin_names = self.model.spin_names

        if self.time_evol_operator is None:
            raise SimphonyError("Missing time_evol_operator")

        if shot not in ['all', 'avg'] and shot not in range(self.n_shots) and not set(shot).issubset(
                set(range(self.n_shots))):
            raise SimphonyError(f"Invalid shot argument, must be 'all', 'avg' or in {range(self.n_shots)}")

        shot_ = 'all' if shot == 'avg' else shot
        operator = self.time_evol_operator.matrix(shot=shot_,
                                                  t_idx=t_idx,
                                                  basis=basis,
                                                  frame=frame,
                                                  in_qubit_subspace=True,
                                                  remove_virtual_phase=remove_virtual_phase)

        spin_idxs = self.model._spin_idx_from_name(spin_names)
        process_matrix = _calculate_process_matrix(operator, keep_idxs=spin_idxs)

        if isinstance(shot, int):
            process_matrix = process_matrix[0, 0]
        elif shot == 'avg':
            process_matrix = unp.mean(process_matrix[:, 0], axis=0)
        else:
            process_matrix = process_matrix[:, 0]

        return process_matrix

    def plot_process_matrix(self,
                            spin_names: Union[str, List[str]] = None,
                            value: str = 're-im',
                            shot: Union[int, str] = 'avg',
                            t_idx: int = -1,
                            basis: str = 'product',
                            frame: str = 'rotating',
                            remove_virtual_phase: bool = True):
        """Plot the process matrix corresponding to the time evolution propagator.

        Args:
            spin_names: Names of the selected spins (defaults to all spins).
            value: 'abs' or 're-im'.
            shot: Specify the shot by its index, or use ``'avg'``.
            t_idx: Specify the time by its index  (defaults to the last timestamp).
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            remove_virtual_phase: Specify to remove the virtual z-phases.

        """

        if shot != 'avg' and shot not in range(self.n_shots):
            raise SimphonyError(f"Invalid shot argument, must be 'avg' or in {range(self.n_shots)}")

        process_matrix = self.process_matrix(spin_names=spin_names,
                                             shot=shot,
                                             t_idx=t_idx,
                                             basis=basis,
                                             frame=frame,
                                             remove_virtual_phase=remove_virtual_phase)

        _plot_process_matrix(process_matrix=process_matrix, value=value)

    def average_gate_fidelity(self,
                              shot: Union[str, int, List[int]] = 'avg',
                              basis: str = 'product',
                              frame: str = 'rotating',
                              ancilla_state: dict = None):
        """Return the average gate fidelity. Compare the final time evolution operator in the qubit subspace in the
        given frame and bases, and the :attr:`ideal_unitary_matrix`.

        Args:
            shot: Specify the shot by its index or indices, or use ``'avg'`` or ``'all'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.
            frame: Specify the frame, must be either ``'lab'`` or ``'rotating'``.
            ancilla_state: Specify the spins used as ancilla qubits and their respective initial states.

        Return:
            Average gate fidelity or fidelities.

        Note:
            The average gate fidelity is calculated as

            .. math::
                F = \\frac{| \\text{Tr}(U_\\text{ideal}^{\\dagger} U_\\text{subspace}) |^2 / d + 1}{d + 1},

            where :math:`d = 2^N` is the dimension of the computational subspace, :math:`U_\\text{ideal}` is the ideal
            (target) unitary operation, and :math:`U_\\text{subspace}` is the actual evolution projected onto the
            computational subspace.

            If an ancilla qubit is present, the reduced operator is obtained by fixing the ancilla in its reference
            state (e.g., :math:`|0\\rangle`) and tracing it out:

            .. math::
                U_r = \\mathrm{Tr}_{\\text{ancilla}} \\Big[ (|0\\rangle\\langle 0| \\otimes \\mathbb{1}_{N-1}) \\, U \\, (|0\\rangle\\langle 0| \\otimes \\mathbb{1}_{N-1}) \\Big],

            where :math:`U` is the operator including the ancilla, and :math:`U_r` is the reduced operator acting only
            on the :math:`N-1` qubit subspace, which is used in the fidelity calculation.

        """

        if self.ideal_unitary_matrix is None:
            raise SimphonyError('ideal_unitary_matrix is missing.')

        if isinstance(shot, int):
            if not (0 <= shot < self.n_shots):
                raise SimphonyError(f"Invalid shot argument: {shot}. Must be between 0 and {self.n_shots - 1}.")
        elif isinstance(shot, list):
            if not all(isinstance(s, int) and 0 <= s < self.n_shots for s in shot):
                raise SimphonyError(
                    f"Invalid shot list: {shot}. Entries must be integers between 0 and {self.n_shots - 1}.")
        elif shot not in ['all', 'avg']:
            raise SimphonyError(f"Invalid shot argument: {shot}. Must be 'all', 'avg', or a valid shot index.")

        shot_ = 'all' if shot == 'avg' else shot
        operator = self.time_evol_operator.matrix(shot=shot_,
                                                  t_idx=-1,
                                                  basis=basis,
                                                  frame=frame,
                                                  in_qubit_subspace=True,
                                                  remove_virtual_phase=True)

        ideal_unitary_matrix = self.ideal_unitary_matrix

        if ancilla_state:
            spin_names = self.model.spin_names
            num_spins = len(spin_names)
            dim = 2 ** num_spins

            if not isinstance(ancilla_state, dict):
                raise SimphonyError('ancilla_state must be a dictionary')

            if any(k not in spin_names or v not in ['0', '1'] for k, v in ancilla_state.items()):
                raise SimphonyError(f"Invalid ancilla_state keys or values: {ancilla_state}")

            keep_indices = [i for i in range(dim) if all(
                format(i, f'0{num_spins}b')[spin_names.index(k)] == v for k, v in ancilla_state.items())]

            operator = operator[..., keep_indices, :][..., :, keep_indices]
            ideal_unitary_matrix = ideal_unitary_matrix[..., keep_indices, :][..., :, keep_indices]

        average_gate_fidelity = average_gate_fidelity_from_unitaries(ideal_unitary_matrix, operator)

        if isinstance(shot, int):
            average_gate_fidelity = average_gate_fidelity[0, 0]
        elif shot == 'avg':
            average_gate_fidelity = unp.mean(average_gate_fidelity[:, 0], axis=0)
        else:
            average_gate_fidelity = average_gate_fidelity[:, 0]

        return average_gate_fidelity

    def leakage(self,
                shot: Union[str, int, List[int]] = 'avg',
                t_idx: Union[str, int, List[int]] = -1,
                basis: str = 'product') -> uarray:
        """Return the leakage from the qubit subspace.

        Leakage is defined as the probability that a quantum operation results in the system leaving the qubit subspace.

        Args:
            shot: Specify the shot by its index or indices, or use ``'all'`` or ``'avg'``.
            t_idx: Specify the time by its index or indices, or use ``'all'``.
            basis: Specify the basis, must be either ``'product'`` or ``'eigen'``.

        Return:
            Leakage or leakages.

        """

        shot_ = 'all' if shot == 'avg' else shot

        time_evol_matrix = self.time_evol_operator.matrix(shot=shot_,
                                                          t_idx=t_idx,
                                                          basis=basis,
                                                          in_qubit_subspace=True)

        leakage = leakage_from_time_evolution_operator(time_evol_matrix)

        if isinstance(shot, int):
            leakage = leakage[..., 0]
        elif shot == 'avg':
            leakage = unp.mean(leakage, axis=0)

        if isinstance(t_idx, int):
            leakage = leakage[0] if leakage.ndim > 0 else leakage

        return leakage

    def ideal_process_matrix(self,
                             spin_names: Union[str, List[str]] = None) -> uarray:
        """Return the process matrix corresponding to the ideal circuit.


        Args:
            spin_names: Names of the selected spins (defaults to all spins).

        Returns:
            Ideal process matrix (in the qubit subspace).

        """

        if self.ideal is None:
            raise SimphonyError('ideal is missing.')

        if spin_names is None:
            spin_names = self.model.spin_names

        spin_idxs = self.model._spin_idx_from_name(spin_names)
        ideal_process_matrix = _calculate_process_matrix(self.ideal_unitary_matrix, keep_idxs=spin_idxs)

        return ideal_process_matrix

    def plot_ideal_process_matrix(self,
                                  spin_names: Union[str, List[str]] = None,
                                  value: str = 're-im'):
        """Plot the ideal process matrix corresponding to the ideal circuit.

        Args:
            spin_names: Names of the selected spins (defaults to all spins).
            value: 'abs' or 're-im'.

        Note:
            Currently only supported to plot max 3 spins.

        """

        ideal_process_matrix = self.ideal_process_matrix(spin_names=spin_names)

        _plot_process_matrix(process_matrix=ideal_process_matrix, value=value)


def _calculate_process_matrix(operator,
                              keep_idxs):
    """Return the process matrix from operator. The operator could have extra dimensions, last two used."""

    n_spins = np.log(operator.shape[-1]) / np.log(2)
    n_spins = int(n_spins)

    pauli_operators = unp.array(pauli_basis(n_spins).to_matrix())

    pauli_coeffs = unp.einsum('...ij,nji->...n', operator, pauli_operators)
    pauli_coeffs /= 2 ** n_spins

    process_matrix = unp.einsum('...i,...j->...ij', pauli_coeffs, pauli_coeffs.conjugate())

    if len(keep_idxs) < n_spins:
        process_matrix = partial_trace(process_matrix, sub_dims=[4] * n_spins, keep=keep_idxs)

    return process_matrix


def _plot_process_matrix(process_matrix: uarray,
                         value: str = 're-im'):
    n_spins = np.log(process_matrix.shape[0]) / np.log(4)

    if not n_spins.is_integer() or process_matrix.ndim != 2 or process_matrix.shape[0] != process_matrix.shape[1]:
        raise SimphonyError('Invalid process matrix. Shape must be (4**n,4**n)')

    n_spins = int(n_spins)

    if n_spins > 3:
        raise SimphonyError('Plotting process matrix corresponding more than 3 spins is not supported.')

    n_cells = 4 ** n_spins

    pauli_strings = [''.join(s) for s in product('IXYZ', repeat=n_spins)]

    fig_width = 0.5 + 0.3 * n_cells
    fig_height = fig_width

    if value == 're-im':
        fig, axs = plt.subplots(ncols=2, figsize=(fig_width, fig_height))
        axs[0].imshow(np.real(process_matrix),
                      interpolation='none',
                      cmap=plt.cm.bwr,
                      vmin=-1,
                      vmax=1)
        axs[0].set_title('$Re(\chi_{ij})$', fontsize=10)

        img = axs[1].imshow(np.imag(process_matrix),
                            interpolation='none',
                            cmap=plt.cm.bwr,
                            vmin=-1,
                            vmax=1)
        axs[1].set_title('$Im(\chi_{ij})$', fontsize=10)

    elif value == 'abs':
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        img = ax.imshow(np.abs(process_matrix),
                        interpolation='none',
                        cmap=plt.cm.binary,
                        vmin=0,
                        vmax=1)
        ax.set_title('$|\chi_{ij}|$', fontsize=10)
        axs = [ax]

    else:
        raise SimphonyError("Invalid value, must be 're-im' or 'abs'")

    for ax in axs:
        ax.set_xticks(range(0, n_cells),
                      labels=pauli_strings,
                      fontfamily='monospace',
                      fontsize=6,
                      rotation=90)
        ax.set_yticks(range(0, n_cells),
                      labels=pauli_strings,
                      fontfamily='monospace',
                      fontsize=6)
        ax.set_aspect('equal', 'box')

        for i in range(n_cells - 1):
            ax.axvline(i + 0.5, alpha=0.3, color='k', linestyle='-', linewidth=0.8)
            ax.axhline(i + 0.5, alpha=0.3, color='k', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.draw()

    pos = axs[-1].get_position()
    one_cell_width = pos.width / n_cells
    pad_norm = one_cell_width * 0.5

    cbar_left = pos.x1 + pad_norm
    cbar_bottom = pos.y0
    cbar_width = one_cell_width
    cbar_height = pos.height

    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    fig.colorbar(img, cax=cbar_ax)

    plt.plot()
