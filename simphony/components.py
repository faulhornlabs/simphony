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

"""Components like spin, interaction, static and driving field, furthermore pulses and time segments."""

from .config import unp
from .utils import fill_array_by_idx, Components1D, Components2D
from .exceptions import SimphonyError

from typing import List, Tuple, Union, Optional, Callable

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt

from qiskit_dynamics.signals import Signal, SignalSum

from jax.lax import stop_gradient

uarray = Union[np.ndarray, jnp.ndarray]


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


class Spin:
    """Represent a spin.

    Initialize a spin.

    Args:
        dimension: Dimension, must be 2 for spin-1/2, 3 for spin-1, and so on.
        name: Name of the spin.
        qubit_subspace: The quantum numbers that defines the qubit subspace.
        gyromagnetic_ratio: Gyromagnetic ratio of the spin (in :math:`\\text{MHz}/\\text{T}`).
        zero_field_splitting: Zero-field splitting (or quadrupole moment) of the spin (in :math:`\\text{MHz}`).
        local_quasistatic_noise: Local-quasistatic noise components of the spin (in :math:`\\text{MHz}`).

    Raises:
        SimphonyError: If an invalid dimension or qubit subspace is provided.

    Hint:
        For example, one can define an NV electron spin as:

        .. code-block:: python

            spin = Spin(
                dimension = 3,
                name = 'NV-e',
                qubit_subspace = (0, -1),
                gyromagnetic_ratio = 28020, # MHz
                zero_field_splitting = 2880, # MHz
                local_quasistatic_noise = [0, 0, 1] # MHz
            )

    """

    def __init__(self,
                 dimension: int,
                 name: str,
                 qubit_subspace: Tuple[float, float],
                 gyromagnetic_ratio: float,
                 zero_field_splitting: float = 0.,
                 local_quasistatic_noise: Optional[Components1D] = None):

        self.dimension: int = dimension
        """Dimension, it is 2 for spin-1/2, 3 for spin-1, and so on."""

        self.name: str = name
        """Name of the spin."""

        self.gyromagnetic_ratio: float = gyromagnetic_ratio
        """Gyromagnetic ratio of the spin (in :math:`\\text{MHz}/\\text{T}`)."""

        self.zero_field_splitting: float = zero_field_splitting
        """Zero-field splitting (or quadrupole moment) (in :math:`\\text{MHz}`)."""

        self.quantum_nums: Tuple[float]
        """Quantum numbers corresponding to spin operator."""

        self.rotating_frame_frequency = 0.
        """Rotating frame frequency corresponding to spin."""

        self.virtual_phase = 0.
        """Virtual z-phase corresponding to spin."""

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

        if not isinstance(dimension, int) or dimension < 2:
            raise SimphonyError("dimension must be an integer greater than 1.")

        s = (dimension - 1) / 2
        self.quantum_nums = tuple(s - i for i in range(dimension))

        sx = np.zeros((dimension, dimension), dtype=complex)
        sy = np.zeros((dimension, dimension), dtype=complex)
        sz = np.diag(self.quantum_nums).astype(complex)
        si = np.eye(dimension, dtype=complex)

        for i in range(dimension - 1):
            factor = np.sqrt((s + 1) * (2 * i + 2) - (i + 1) * (i + 2))
            sx[i, i + 1] = sx[i + 1, i] = 0.5 * factor
            sy[i, i + 1] = -0.5j * factor
            sy[i + 1, i] = 0.5j * factor

        self.operator.i = si
        self.operator.x = sx
        self.operator.y = sy
        self.operator.z = sz

        self.qubit_subspace = qubit_subspace

        self.local_quasistatic_noise: Components1D = Components1D(local_quasistatic_noise)
        """The local quasistatic noise components (in :math:`\\text{MHz}`).
        
        Tip:
            The vector components are stored as:
    
            .. code-block:: python
    
                >>> spin.local_quasistatic_noise.vec
                [0, 0, 0.1]
                
            Individual components (``x``, ``y`` or ``z``) can be accessed directly using their respective attributes:
    
            .. code-block:: python
    
                >>> spin.local_quasistatic_noise.z
                0.1
        """

    def __repr__(self):
        return (
            f'{type(self).__name__}'
            f'(dimension={self.dimension}, '
            f"name='{self.name}', "
            f'qubit_subspace={self.qubit_subspace}, '
            f'gyromagnetic_ratio={self.gyromagnetic_ratio}, '
            f'zero_field_splitting={self.zero_field_splitting}, '
            f'local_quasistatic_noise={self.local_quasistatic_noise.vec})'
        )

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
            array=np.array([[1, 0], [0, 1]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.x = fill_array_by_idx(
            array=np.array([[0, 1], [1, 0]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.y = fill_array_by_idx(
            array=np.array([[0, -1j], [1j, 0]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )
        self.operator_qubit_subspace.z = fill_array_by_idx(
            array=np.array([[1, 0], [0, -1]], dtype=complex),
            dimension=self.dimension,
            idxs=idxs
        )


class Interaction(Components2D):
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

    def __init__(self,
                 spin1: Spin,
                 spin2: Spin,
                 tensor: np.array = None):
        super().__init__(tensor)
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
        return (f'{type(self).__name__}('
                f"spin_name_1='{self.spin1.name}', "
                f"spin_name_2='{self.spin2.name}', "
                f'tensor={self.tensor}'
                ')'
                )


class StaticField(Components1D):
    """Represent a static magnetic field.

    Initialize the static magnetic field.

    Args:
        strengths: Strengths of the components (in :math:`\\text{T}`, defaults to zero vector).

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

    def __init__(self,
                 strengths: List[float] = None):
        super().__init__(strengths)

        self.vec: Union[List[float], np.array]
        """Vector corresponding the static magnetic field."""

        self.x: float
        """The :math:`x` components of the static magnetic field."""

        self.y: float
        """The :math:`y` components of the static magnetic field."""

        self.z: float
        """The :math:`z` components of the static magnetic field."""

    def __repr__(self):
        return f'{type(self).__name__}(strengths={self.vec})'


class DrivingField(Components1D):
    """Represent an AC magnetic field.

    The AC magnetic field is used to drive the spin system. At initialization, only the direction of the field is set.
    Time-dependence of the field can be set by adding a pulse with the :meth:`add_..._pulse` methods.

    Initialize a driving magnetic field.

    Args:
        name: Name of the driving field.
        direction: Direction of the driving field. If not a unit vector, it will be normalized.

    Hint:
        An x-directional microwave field can be initialized with:

        .. code-block:: python

            driving_field_MW = simphony.DrivingField(direction = [1, 0, 0], name = 'MW_x')
    """

    def __init__(self,
                 name: str,
                 direction: List[float]):
        direction = direction / np.linalg.norm(direction)
        direction.tolist()

        super().__init__(direction)

        self.name: str = name
        """Name of the driving field."""

        self.last_pulse_end: float = 0.
        """End time of the latest-ending pulse."""

        self.pulses: PulseList = PulseList()
        """Pulses added to the driving field."""

        self.vec: List[float]
        """Vector corresponding the unit vector of the magnetic field."""

        self.x: float
        """The :math:`x` components of the magnetic field's unit vector."""

        self.y: float
        """The :math:`y` components of the magnetic field's unit vector."""

        self.z: float
        """The :math:`z` components of the magnetic field's unit vector."""

    def __repr__(self):
        return f"{type(self).__name__}(name='{self.name}', direction={self.vec})"

    def remove_all_pulses(self):
        """Remove all previously added pulses from the driving field."""
        self.last_pulse_end = 0.
        self.pulses = PulseList()

    def add_pulse(self,
                  pulse):
        """ Add a general pulse to the driving field.

        Args:
            pulse (Pulse): Pulse object.

        """

        self.pulses.append(pulse)
        if pulse.driving_field is None:
            pulse.driving_field = self
        else:
            raise SimphonyError(f'Pulse is already added to {pulse.driving_field}')
        self.last_pulse_end = unp.max(unp.array([self.last_pulse_end, pulse.end]))

    def add_rectangle_pulse(self,
                            amplitude: float,
                            frequency: float,
                            phase: float,
                            duration: float,
                            start: Optional[float] = None,
                            rise_time: float = 0,
                            fall_time: float = 0):
        """Add a pulse to the driving field with rectangular envelope (constant-amplitude). The pulse is described by a
        ``frequency`` and a ``phase``, and its envelope is characterized by an ``amplitude`` and a ``duration``. By
        default, the pulse is turned on and off instantly, but by setting the ``rise_time`` and ``fall_time``
        parameters, the rise and fall of the pulse can be set to a finite duration.

        Args:
            amplitude: Amplitude of the pulse (in :math:`\\text{T}`).
            frequency: Frequency of the pulse (in :math:`\\text{MHz}`).
            phase: Phase of the pulse.
            duration: Duration of the pulse (in :math:`\\mu\\text{s}`).
            start: Trigger (start) time of the pulse (in :math:`\\mu\\text{s}`, defaults to end of the last pulse).
            rise_time: Rise time of the pulse (in :math:`\\mu\\text{s}`).
            fall_time: Fall time of the pulse (in :math:`\\mu\\text{s}`).

        Note:
            - The time-dependence of the rise and fall is described by a sinusoidal function, for instance:

            .. math::
                 f_\\text{rise}(t) = \\frac{A}{2}\\left(1+\\cos\\left[\\pi\\left(1-\\frac{t}{\\tau_\\text{rise}}\\right)
                 \\right]\\right),\\quad\\text{where } 0 \\le t \\le \\tau_\\text{rise}.

        """

        if duration <= 0:
            raise SimphonyError('duration must be positive')

        if rise_time < 0 or fall_time < 0:
            raise SimphonyError('rise_time and fall_time must be non-negative')

        if rise_time + fall_time > duration:
            raise SimphonyError('Sum of the rise_time and fall_time must be greater than or equal to the duration')

        if start is None:
            start = self.last_pulse_end

        if frequency < 0:
            frequency *= -1
            phase += np.pi

        if rise_time > 0:
            def rise(t):
                """This must be defined with jnp"""
                return jnp.exp(1j * phase) * 0.5 * amplitude * (1 + jnp.cos(jnp.pi * (1 - (start - t) / rise_time)))

            pulse = Pulse(start=start,
                          end=start + rise_time,
                          frequency=frequency,
                          complex_envelope=rise,
                          driving_field=self)
            self.pulses.append(pulse)

        pulse = Pulse(start=start + rise_time,
                      end=start + duration - fall_time,
                      frequency=frequency,
                      complex_envelope=amplitude * unp.exp(1j * phase),
                      driving_field=self)
        self.pulses.append(pulse)

        if fall_time > 0:
            def fall(t):
                """This must be defined with jnp"""
                return jnp.exp(1j * phase) * 0.5 * amplitude * (1 + jnp.cos(jnp.pi * (1 - (start + duration - t) / fall_time)))

            pulse = Pulse(start=start + duration - fall_time,
                          end=start + duration,
                          frequency=frequency,
                          complex_envelope=fall,
                          driving_field=self)
            self.pulses.append(pulse)

        self.last_pulse_end = unp.max(unp.array([self.last_pulse_end, start + duration]))

    def add_discrete_pulse(self,
                           samples: List[Union[float, complex]],
                           frequency: float,
                           dt: Union[float, List[float]],
                           start: Optional[float] = None):
        """Add a pulse to the driving field with piecewise-constant envelope. The pulse is described by its
        ``frequency`` and its envelope is characterized by ``samples`` and ``dt``.

        Args:
            samples: Array of samples (in :math:`\\text{T}`).
            frequency: Frequency of the pulse (in :math:`\\text{MHz}`).
            dt: Duration(s) of the pulse segments (in :math:`\\mu\\text{s}`).
            start: Trigger (start) time of the pulse (in :math:`\\mu\\text{s}`, defaults to end of the last pulse).

        """

        samples = unp.array(samples, dtype=complex)
        len_samples = samples.shape[0]

        if start is None:
            start = self.last_pulse_end

        # if hasattr(dt, '__len__'):
        #     dt = unp.array(dt)
        #     if len(dt) != len(samples):
        #         raise SimphonyError('samples and dt must have the same length')
        #     if unp.any(dt <= 0):
        #         raise SimphonyError('All dt must be positive')
        #
        #     ts = unp.concatenate([unp.array([start]), start + unp.cumsum(dt)])
        #
        # else:
        #     if dt <= 0:
        #         raise SimphonyError('dt must be positive')
        #
        #     ts = unp.linspace(start, start + dt * len(samples), len(samples) + 1)

        dt = unp.atleast_1d(unp.array(dt))
        len_dt = dt.shape[0]

        if len_dt == 1:
            if dt[0] <= 0:
                raise SimphonyError('dt must be positive')

            ts = unp.linspace(start, start + dt[0] * len_samples, len_samples + 1)

        elif len_dt == samples.shape[0]:
            if unp.any(dt <= 0):
                raise SimphonyError('All dt must be positive')
            
            ts = unp.concatenate([unp.array([start]), start + unp.cumsum(dt)])

        else:
            raise SimphonyError('ts must be a single value, or dt and samples must have the same length')

        if frequency < 0:
            frequency *= -1
            samples *= unp.exp(1j * np.pi)

        for sample, start, end in zip(samples, ts[:-1], ts[1:]):
            pulse = Pulse(start=start,
                          end=end,
                          frequency=frequency,
                          complex_envelope=sample,
                          driving_field=self)
            self.pulses.append(pulse)

        self.last_pulse_end = unp.max(unp.array([self.last_pulse_end, ts[-1]]))

    def add_wait(self,
                 duration: float):
        """Add an idle section to the driving field. It is characterized by the ``duration`` time.

        Args:
            duration: Duration of the waiting (in :math:`\\mu\\text{s}`).

        """
        if duration <= 0:
            raise SimphonyError('duration must be positive')
        self.last_pulse_end += duration

    def plot_pulses(self,
                    function: str = 'full_waveform',
                    start: float = 0.,
                    end: Optional[float] = None,
                    ax=None):
        """Plot the added pulse(s).

        Args:
            function: Could be ``full_waveform`` or ``complex_envelope``.
            start: Start time of the plotted interval (in :math:`\\mu\\text{s}`, defaults to 0).
            end: End time of the plotted interval (in :math:`\\mu\\text{s}`). If ``None``, it will be set to the end
                time of the last-ending pulse.
            ax: Axis to customize the plot.

        """

        if end is None:
            end = self.last_pulse_end

        if end <= start:
            raise SimphonyError('end must be greater than start')

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 2.5))

        pulses = self.pulses

        time_segment_sequence = pulses.convert_to_time_segment_sequence(start=start,
                                                                        end=end)

        self._time_segment_sequence = time_segment_sequence

        time_segment_sequence.plot(function=function, ax=ax)

        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel('$B_{ac}$ [T]')
        ax.grid()
        ax.set_title('{} ({})'.format(self.name, function))
        plt.tight_layout()


class Pulse:
    """Represents a single pulse.

    The class defines a finite-duration pulse characterized by a constant frequency and a specified complex envelope. It
    can be attached to a driving field to specify its time-dependent behavior.

    Note:
        :class:`Pulse` is a low-level object representing a pulse or a pulse fragment (e.g. rising edge of a smooth
        pulse). To add pulses to driving fields, use the appropriate :meth:`add_..._pulse` methods of the corresponding
        :class:`DrivingField` instances.

    Initialize a pulse.

    Args:
        start: Trigger (start) time of the pulse (in :math:`\\mu\\text{s}`).
        end: End time of the pulse (in :math:`\\mu\\text{s}`).
        frequency: Frequency of the pulse (in :math:`\\text{MHz}`).
        complex_envelope: Complex envelope of the pulse. It can be a complex number or a callable returning a
            complex-valued function (in :math:`\\text{T}`).
        driving_field: Driving field which the pulse is attached to.

    """

    def __init__(self,
                 start: float,
                 end: float,
                 frequency: float,
                 complex_envelope: Union[complex, Callable],
                 driving_field: DrivingField = None):

        self.start: float = start
        """Trigger (start) time of the pulse."""

        if end <= start:
            raise SimphonyError('end should be greater than start')

        self.end: float = end
        """End time of the pulse."""

        self.frequency: float = frequency
        """Frequency of the pulse."""

        self.complex_envelope: Union[complex, Callable] = complex_envelope
        """Complex envelope of the pulse."""

        self.driving_field = driving_field
        """Driving field which the pulse is attached to."""

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
        return not isinstance(self.complex_envelope, Callable)

    @property
    def signal(self) -> Signal:
        """Returns the pulse's signal as a :class:`qiskit_dynamics.Signal` instance."""
        if self.is_constant_envelope:
            def complex_envelope_fn(ts):
                return unp.where(ts < self.start, 0.,
                                unp.where(ts < self.end, self.complex_envelope, 0.))
        else:
            def complex_envelope_fn(ts):
                return unp.where(ts < self.start, 0.,
                                unp.where(ts < self.end, self.complex_envelope(ts), 0.))

        return Signal(envelope=complex_envelope_fn, carrier_freq=self.frequency)


class PulseList:
    """
    Represent multiple pulses.

    Tht class behaves similarly to a standard :class:`list`, it supports indexing, setting items, iteration, and
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

    def __getitem__(self, index) -> Pulse:
        return self._pulses[index]

    def __setitem__(self, index, value):
        self._pulses[index] = value

    def __repr__(self):
        return f"{type(self).__name__}(n_pulses={len(self)})"

    def __str__(self):
        lines = [f"{type(self).__name__}(["]
        for pulse in self._pulses:
            lines.append(f"   {repr(pulse)},")
        lines.append("])")
        return "\n".join(lines)

    def append(self,
               new_pulse: Pulse):
        """Append a pulse."""
        self._pulses.append(new_pulse)

    def __add__(self, other):
        new = self._pulses + other._pulses
        return self.__class__(new)

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
                                         end: Optional[float] = None):
        """Convert the to :class:`TimeSegmentSequence`.

        Args:
            start: Start time of the inspected time interval (defaults to the start of the first pulse).
            end: End time of the inspected time interval (defaults to the end of the last pulse).

        Returns:
            TimeSegmentSequence: A sequence of time segments, each containing pulses that overlap with the respective time interval.
        """
        division_points = _calculate_division_points(
            self,
            start=start,
            end=end
        )

        time_segment_sequence = []
        for segment_start, segment_end in zip(division_points, division_points[1:]):
            pulses = PulseList(
                [pulse for pulse in self if pulse.start <= segment_start and pulse.end >= segment_end]
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

    division_points = [t for pulse in pulse_list for t in (pulse.start, pulse.end) if
                       (start is None or t > start) and (end is None or t < end)]
    if start is not None:
        division_points.insert(0, start)
    if end is not None:
        division_points.append(end)
    division_points = unp.array(division_points, dtype=unp.float64)

    division_points = unp.sort(division_points)

    tol = 1e-10
    idxs = unp.diff(division_points) > tol
    idxs = unp.insert(idxs, 0, True)
    division_points = division_points[idxs]

    return division_points


class TimeSegment(PulseList):
    """Represents a time segment associated with a list of pulses.

    A ``TimeSegment`` is a time interval and includes all pulses that are active in the interval. It based on the
    :class:`PulseList` and provides additional metadata relevant for plotting and simulation.

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
        super().__init__(pulses)

        self.start: float = start
        """Start time of the segment."""

        self.end: float = end
        """End time of the segment."""

        self.plot_ts: Optional[uarray] = None
        """Time points for plotting."""

        self.plot_ps: Optional[uarray, List[uarray]] = None
        """Values for plotting."""

        self.simulation_type: Optional[str] = None
        """Type of the simulation."""

        self.simulation_t_span: Optional[uarray] = None
        """Time interval for Qiskit Dynamics's ``expm`` solver."""

        self.simulation_t_eval: Optional[uarray] = None
        """Evaluation time points for Qiskit Dynamics's ``expm`` solver."""

        self.simulation_max_dt: Optional[float] = None
        """Maximal step size for the Qiskit Dynamics's ``expm`` solver."""

        self.simulation_signals: Optional[Union[Signal, SignalSum]] = None
        """:attr:`qiskit_dynamics.Signal` for Qiskit Dynamics's ``expm`` solver."""

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'start={self.start}, '
                f'end={self.end}, '
                f'n_pulses={self.n_pulses}, '
                f"simulation_type={'None' if self.simulation_type is None else repr(self.simulation_type)}"
                ')'
                )

    def append(self,
               new_pulse: Pulse):
        """Appending a pulse is disabled for TimeSegment instances."""
        raise SimphonyError('Appending a pulse is disabled for TimeSegment instances.')

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
            return self[0].frequency
        else:
            return max(pulse.frequency for pulse in self)

    def signal(self,
               driving_field_name: Optional[str] = None) -> Union[Signal,SignalSum]:
        """Return the signal to be used in simulation.

        Note:
            The signal is not clipped to the segment’s time interval. However, it is intended to be used only
            within this interval. This reduces unnecessary computations and improves simulation efficiency.

        Args:
            driving_field_name: If specified, only pulses matching this driving field name are included in the resulting
                signal. If ``None``, all pulses in the segment are used.

        Returns:
            The signal constructed from the pulses in this time segment.

        """

        signal = Signal(0.)
        for pulse in self:
            if not driving_field_name or pulse.driving_field.name == driving_field_name:
                signal += Signal(carrier_freq=pulse.frequency, envelope=pulse.complex_envelope)
        return signal

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
                self.plot_ts = unp.array([self.start, self.end])
                self.plot_ps = unp.array([0., 0.])
            else:
                max_frequency = self.max_frequency
                duration = self.duration
                n_points = int(duration * max_frequency + 1) * 251
                max_n_points = 10000
                if n_points > max_n_points:
                    n_points = max_n_points
                self.plot_ts = unp.linspace(self.start, self.end, n_points)
                signal = self.signal()
                self.plot_ps = signal(self.plot_ts)

        elif function == 'complex_envelope':
            if self.n_pulses == 0:
                self.plot_ts = unp.array([self.start, self.end])
                self.plot_ps = [unp.array([0., 0.])]
            elif self.is_all_constant_envelope:
                self.plot_ts = unp.array([self.start, self.end])
                self.plot_ps = [unp.array([pulse.complex_envelope for t in self.plot_ts]) for pulse in self]
            else:
                n_points = 251
                self.plot_ts = unp.linspace(self.start, self.end, n_points)
                self.plot_ps = [unp.array(
                    [pulse.complex_envelope if pulse.is_constant_envelope else pulse.complex_envelope(t) for t in
                     self.plot_ts]) for pulse in self]

        else:
            raise SimphonyError("Invalid function, must be 'complex_envelope' or 'full_waveform'")

    def discretize_simulation(self,
                              included_driving_field_names: List[str],
                              method: str,
                              n_eval: int = 251,
                              n_split: int = 250):
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
            included_driving_field_names: Specify which signals are stored in :attr:`simulation_signals`.
            method: Could be ``'basic'`` or ``'single_sine_wave'``.
            n_eval: Number of evaluation points at which the simulation results (time evolution operator or
                wave function) will be computed and returned.
            n_split: Number of time steps used discretize to one period of the highest-frequency component.

        """

        if method == 'basic':
            self._discretize_simulation_basic(n_eval=n_eval,
                                              n_split=n_split)
        elif method == 'single_sine_wave':
            self._discretize_simulation_single_sine_wave(n_eval=n_eval,
                                                         n_split=n_split)
        else:
            raise SimphonyError("method must be 'basic' or 'single_sine_wave'")

        self.simulation_signals = [self.signal(name) for name in included_driving_field_names]

    def _discretize_simulation_basic(self,
                                     n_eval: int = 251,
                                     n_split: int = 250):
        if self.n_pulses == 0:
            max_dt = self.duration
        else:
            max_frequency = self.max_frequency
            if self.is_all_constant_envelope:
                max_dt = 1. / max_frequency / n_split
            else:
                max_dt = unp.min(unp.array([1. / max_frequency, self.duration])) / n_split

        self.simulation_type = 'basic'
        self.simulation_t_span = unp.array([self.start, self.end])
        self.simulation_t_eval = unp.linspace(self.start, self.end, n_eval)
        self.simulation_max_dt = max_dt

    def _discretize_simulation_single_sine_wave(self,
                                                n_eval: int = 251,
                                                n_split: int = 250):
        if self.n_pulses == 1 and self[0].is_constant_envelope and 1. / self[0].frequency <= self.duration:
            periodic_time = 1. / self[0].frequency
            max_dt = periodic_time / n_split

            t_eval = unp.linspace(self.start, self.end, n_eval)
            cycle_indices, t_eval_projected = unp.divmod(t_eval - self.start, periodic_time)
            cycle_indices = cycle_indices.astype(int)
            t_eval_projected += self.start

            sort_indices = unp.argsort(t_eval_projected)
            t_eval_projected_sorted = t_eval_projected[sort_indices]
            unsort_indices = unp.argsort(sort_indices)

            t_end_first_sine_wave = self.start + periodic_time
            t_eval_projected_sorted = unp.append(t_eval_projected_sorted, t_end_first_sine_wave)

            self.simulation_type = 'single_sine_wave'
            self.simulation_t_span = unp.array([self.start, self.end])
            self.simulation_t_eval = t_eval
            self.simulation_max_dt = max_dt
            self.simulation_t_span_projected_sorted = unp.array([self.start, t_end_first_sine_wave])
            self.simulation_t_eval_projected_sorted = t_eval_projected_sorted
            self.simulation_unsort_indices = unsort_indices
            self.simulation_cycle_indices = cycle_indices

        else:
            self._discretize_simulation_basic(n_eval=n_eval, n_split=n_split)


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

    def __len__(self):
        return len(self._time_segments)

    def __getitem__(self, index):
        return self._time_segments[index]

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
                ts.append(time_segment.plot_ts)
                ps.append(time_segment.plot_ps)
            ts = unp.concatenate(ts)
            ps = unp.concatenate(ps)

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

                plot_ts = stop_gradient(time_segment.plot_ts)
                plot_ps = stop_gradient(time_segment.plot_ps)

                for ps in plot_ps:
                    ps_start_real.append(ps[0].real)
                    ps_start_imag.append(ps[0].imag)
                    ps_end_real.append(ps[-1].real)
                    ps_end_imag.append(ps[-1].imag)
                t_start = plot_ts[0]

                ax.plot([t_start, t_start], [min(ps_start_real), max(ps_start_real)], c=c_real)
                ax.plot([t_start, t_start], [min(ps_start_imag), max(ps_start_imag)], '--', c=c_imag)

                for ps in plot_ps:
                    ax.plot(plot_ts, unp.real(ps), c=c_real)
                    ax.plot(plot_ts, unp.imag(ps), '--', c=c_imag)

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
                              n_split: int = 250):
        """Discretize the time segment list for simulation.

        Args:
            included_driving_field_names: Specify which driving field' signals are included in
                :attr:`simulation_signals`.
            method: Specify the method. It could be ``'basic'`` or ``'single_sine_wave'``.
            n_eval: Number of evaluation points.
            n_split: Number of splitting points for non-constant envelope and for single sine wave.
        """

        for time_segment in self:
            time_segment.discretize_simulation(included_driving_field_names=included_driving_field_names,
                                               method=method,
                                               n_eval=n_eval,
                                               n_split=n_split)
