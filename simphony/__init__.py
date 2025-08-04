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
_environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
_environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
_environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable x64 for jax
import jax as _jax
_jax.config.update('jax_enable_x64', True)

# Check the hyperfine database and download it if not available
from urllib.request import urlretrieve as _urlretrieve
from urllib.error import HTTPError as _HTTPError
from os.path import dirname as _dirname, join as _join, exists as _exists
from os import makedirs as _makedirs

_current_dir = _dirname(__file__)
_file = _join(_current_dir, 'data/hyperfine/nv.csv')
_url = 'https://faulhornlabs.com/simphony/nv.csv'

if not _exists(_file):
    print('Hyperfine database not found.')
    try:
        _makedirs(_dirname(_file), exist_ok=True)
        _urlretrieve(_url, _file)
        print('Hyperfine database has been downloaded.')

    except _HTTPError:
        print('Hyperfine database could not be found on the online server. The carbon_atom_indices argument of the default_nv_model function will not work.')

# Import classes and functions from submodules
from .config import Config
from .defaults import default_nv_model, default_rotating_frame
from .utils import (
    fill_array_by_idx, is_unitary, leakage_from_process_matrix, leakage_from_time_evolution_operator, tensorprod,
    Components1D, Components2D, average_gate_fidelity_from_unitaries, average_gate_fidelity_from_process_matrices,
    partial_trace
)
from .components import (
    Spin, Interaction, StaticField, DrivingField, Pulse, PulseList, TimeSegment, TimeSegmentSequence,
    _calculate_division_points
)
from .model import Model
from .simulationresult import SimulationResult, TimeEvolState, TimeEvolOperator, Operator
from .exceptions import SimphonyError, SimphonyWarning

# Patch qiskit_dynamics 0.5.1
from . import _qiskit_dynamics_patch

# Set the computation platform to 'cpu' by default
Config.set_platform('cpu')

# Set gpu to 0th in CUDA_VISIBLE_DEVICES by default
Config.set_gpu(0)

# Set the autodiff_mode to False by default
Config.set_autodiff_mode(False)