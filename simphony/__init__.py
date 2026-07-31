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

"""Package simulating dynamics of coupled spin systems."""

# Prevent 'jax' from allocating 75% of the memory of the gpu by default, and enable to allocate 95% of the memory
from os import environ as _environ
_environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
_environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.95')
_environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Enable x64 for jax
import jax as _jax
_jax.config.update('jax_enable_x64', True)

# Import classes and functions from submodules
from .config import Config, unp_static, unp_dynamic, unp_analysis
from .defaults import default_nv_model, default_multi_nv_model, default_nv_hyperfine_database
from .utils import (
    fill_array_by_idx, is_unitary, leakage_from_process_matrix, leakage_from_time_evolution_operator, tensorprod,
    Components1D, Components2D, average_gate_fidelity_from_unitaries, average_gate_fidelity_from_process_matrices,
    partial_trace
)
from .components import (
    BaseSpin, ElectronSpin, NuclearSpin,
    Interaction, StaticField, BaseDrivingField, LinearDrivingField, CircularDrivingField,
    Pulse, PulseList, TimeSegment, TimeSegmentSequence,
    _calculate_division_points
)
from .model import Model, RotatingFrameSetter
from .analysis import SimulationResult, TimeEvolState, TimeEvolOperator, TimeEvolDensityMatrix, Operator
from .exceptions import SimphonyError, SimphonyWarning

# qiskit_dynamics is no longer required; internal solvers are used.

# Defaults are initialized when importing ``simphony.config``.
