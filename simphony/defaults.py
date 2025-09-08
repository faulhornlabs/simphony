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

"""Default models, pulses and pulse sequences."""

from typing import Optional, Union, List, Tuple, Dict
Num = Union[int, float]

from os.path import dirname as _dirname, join as _join

import pandas as pd

from .exceptions import SimphonyError
from .components import Spin, Interaction, StaticField, DrivingField
from .model import Model

# Constants (references in the `default_nv_model` docstring)
ELECTRON_GYROMAGNETIC_RATIO = 28033.1  # MHz/T
ELECTRON_ZERO_FIELD_SPLITTING = 2872.  # MHz

NITROGEN14_GYROMAGNETIC_RATIO = -3.07771  # MHz/T
NITROGEN14_QUADRUPOLE_SPLITTING = -5.01  # MHz
NITROGEN14_HYPERFINE_PARALLEL = -2.14  # MHz
NITROGEN14_HYPERFINE_PERPENDICULAR = -2.70  # MHz

NITROGEN15_GYROMAGNETIC_RATIO = 4.31727  # MHz/T
NITROGEN15_HYPERFINE_PARALLEL = 3.03  # MHz
NITROGEN15_HYPERFINE_PERPENDICULAR = 3.65  # MHz

CARBON13_GYROMAGNETIC_RATIO = -10.7084  # MHz/T


def default_nv_model(nitrogen_isotope: Optional[int] = None,
                     carbon_atom_indices: Optional[List[tuple]] = None,
                     static_field_strength: float = 0.05,
                     interaction: bool = True,
                     **kwargs: dict) -> Model:
    """Create a default central spin model based on a nitrogen-vacancy center. The electron spin of the nitrogen-vacancy
    center is added by default as ``'e'``. :math:`^{13}\\text{C}` nuclear spins can be added via the
    ``carbon_atom_indices`` argument. The positions of the nuclear spins are used to determine the hyperfine
    tensors, based on the `Ivády Group's hyperfine dataset <https://ivadygroup.elte.hu/hyperfine/nv/index.html>`_.

    Args:
        nitrogen_isotope: Atomic number of the nitrogen isotope (optional, should be ``14`` or ``15``).
        carbon_atom_indices: List of (:math:`n_1`, :math:`n_2`, :math:`n_3`, :math:`n_4`) tuples.
        static_field_strength: Strength of the static field (in :math:`\\text{T}`)
        interaction: Whether interaction is presented or not.
        **kwargs: Additional keyword arguments passed to the ``default_rotating_frame()`` function, which is
            responsible for defining the rotating frame.

    Return:
        Default model.

    Raise:
        SimphonyError: If invalid nitrogen isotope is given. If invalid carbon nuclear spin indices are given.

    .. Note::

        The Hamiltonian describes the default NV model (*note that our convention for nuclear spin gyromagnetic ratios
        is different from the standard convention*):

        .. math::
            H =&
            \\underbrace{\\gamma_{e} \\mathbf{B} \\cdot \\mathbf{S} + D S_z^2}_{\\text{electron}}
            +
            \\underbrace{\\gamma_{N} \\mathbf{B} \\cdot \\mathbf{I}_{N} + P I_{N,z}^2 + \\mathbf{S} \\cdot \\mathbf{A}_{N} \\cdot \\mathbf{I}_{N}}_{\\text{nitrogen}}
            \\\\
            &+\\underbrace{\\sum_{i}{\\gamma_{C} \\mathbf{B} \\cdot \\mathbf{I}_{C}^{(i)} + \\mathbf{S} \\cdot \\mathbf{A}_{C}^{(i)} \\cdot \\mathbf{I}_{C}^{(i)}}}_{\\text{carbon(s)}},

        where the parameters are the follows:

        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Spin                        | Parameter               | Symbol                 | Value                               |
        +=============================+=========================+========================+=====================================+
        | Electron (:math:`S=1`)      | Gyromagnetic ratio      | :math:`\gamma_e`       | :math:`28.0331\,\\text{GHz/T}` [1]   |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Zero-field splitting    | :math:`D`              | :math:`2.872\,\\text{GHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Nitrogen-14 (:math:`I=1`)   | Gyromagnetic ratio      | :math:`\gamma_{N}`     | :math:`-3.07771\,\\text{MHz/T}` [2]  |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Quadrupole splitting    | :math:`P`              | :math:`-5.01\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine perpendicular | :math:`A_{N\perp}`     | :math:`-2.70\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine parallel      | :math:`A_{N\parallel}` | :math:`-2.14\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Nitrogen-15 (:math:`I=1/2`) | Gyromagnetic ratio      | :math:`\gamma_{N}`     | :math:`4.31727\,\\text{MHz/T}` [2]   |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine perpendicular | :math:`A_{N\perp}`     | :math:`3.65\,\\text{MHz}` [1]        |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine parallel      | :math:`A_{N\parallel}` | :math:`3.03\,\\text{MHz}` [1]        |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Carbon-13 (:math:`I=1/2`)   | Gyromagnetic ratio      | :math:`\gamma_C`       | :math:`-10.7084\,\\text{MHz/T}` [2]  |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+

        | References:
        | [1] Felton et al., Phys. Rev. B 79, 075203 (2009)
        | [2] CRC Handbook of Chemistry and Physics, sec. 11-4  (97th edition)


    .. Hint::

        The carbon atom indices specify the position of a carbon atom as:

        .. math::

            \\text{carbon atom position} = n_1 \\cdot \\mathbf{a}_1 + n_2 \\cdot \\mathbf{a}_2 +
            n_3 \\cdot \\mathbf{a}_3 + n_4 \\cdot \\boldsymbol{\\tau},

        where :math:`\\mathbf{a}_1`, :math:`\\mathbf{a}_2` and :math:`\\mathbf{a}_3` are the primitive lattice vectors,
        :math:`\\mathbf{0}` and :math:`\\boldsymbol{\\tau}` are the positions of the atoms inside the primitive cell,
        furthermore :math:`n_1`, :math:`n_2`, :math:`n_3` (integers) and :math:`n_4` (:math:`0` or :math:`1`) are the
        carbon nuclear spin indices. Our convention is:

        .. math::

            \\mathbf{a}_1 &= a_\\text{CC}\\cdot ( 0, 2\\sqrt{2}/3, 4/3 ),

            \\mathbf{a}_2 &= a_\\text{CC}\\cdot ( -\\sqrt{6}/3, -\\sqrt{2}/3, 4/3 ),

            \\mathbf{a}_3 &= a_\\text{CC}\\cdot ( \\sqrt{6}/3, -\\sqrt{2}/3, 4/3 ),

            \\boldsymbol{\\tau} &= (\\mathbf{a}_1 + \\mathbf{a}_2 + \\mathbf{a}_3) / 4 = a_\\text{CC}\\cdot ( 0, 0, 1 )

        where :math:`a_\\text{CC} = 0.1545 \\text{ nm}` is the carbon-carbon distance. The nitrogen occupies the
        :math:`\\boldsymbol{\\tau}` position, while the missing carbon atom corresponds to the :math:`\\mathbf{0}`
        lattice point.

        .. _default_nv_model:

    """

    static_field = StaticField()
    static_field.z = static_field_strength

    driving_field_MW = DrivingField(direction=[1, 0, 0], name='MW_x')
    driving_field_RF = DrivingField(direction=[1, 0, 0], name='RF_x')

    model = Model(name='default')
    model.add_static_field(static_field)
    model.add_driving_field(driving_field_MW)
    model.add_driving_field(driving_field_RF)

    spin_e = Spin(dimension=3,
        name='e',
        qubit_subspace=(0, -1),
        gyromagnetic_ratio=ELECTRON_GYROMAGNETIC_RATIO,
        zero_field_splitting=ELECTRON_ZERO_FIELD_SPLITTING
    )
    model.add_spin(spin_e)

    if nitrogen_isotope == 14 or nitrogen_isotope == 15 or nitrogen_isotope is None:

        if nitrogen_isotope == 14:
            spin_N = Spin(
                dimension=3,
                name='N',
                qubit_subspace=(0, -1),
                gyromagnetic_ratio=NITROGEN14_GYROMAGNETIC_RATIO,
                zero_field_splitting=NITROGEN14_QUADRUPOLE_SPLITTING
            )
            hyperfine_N = Interaction(spin_e, spin_N)
            hyperfine_N.xx = NITROGEN14_HYPERFINE_PERPENDICULAR
            hyperfine_N.yy = NITROGEN14_HYPERFINE_PERPENDICULAR
            hyperfine_N.zz = NITROGEN14_HYPERFINE_PARALLEL

        if nitrogen_isotope == 15:
            spin_N = Spin(
                dimension=2,
                name='N',
                qubit_subspace=(-1 / 2, 1 / 2),
                gyromagnetic_ratio=NITROGEN15_GYROMAGNETIC_RATIO,
                zero_field_splitting=0.
            )
            hyperfine_N = Interaction(spin_e, spin_N)
            hyperfine_N.xx = NITROGEN15_HYPERFINE_PERPENDICULAR
            hyperfine_N.yy = NITROGEN15_HYPERFINE_PERPENDICULAR
            hyperfine_N.zz = NITROGEN15_HYPERFINE_PARALLEL

        if nitrogen_isotope == 14 or nitrogen_isotope == 15:
            model.add_spin(spin_N)

            if interaction is True:
                model.add_interaction(hyperfine_N)

    else:
        raise SimphonyError('Invalid nitrogen isotope, should be 14 or 15 or None')

    if carbon_atom_indices is not None:

        current_dir = _dirname(__file__)
        file_database = _join(current_dir, './data/hyperfine/nv.csv')

        hyperfine_df = pd.read_csv(file_database, index_col=[0])

        for idx, indices in enumerate(carbon_atom_indices):

            hyperfine_row = hyperfine_df[(hyperfine_df['n1'] == indices[0]) &
                                         (hyperfine_df['n2'] == indices[1]) &
                                         (hyperfine_df['n3'] == indices[2]) &
                                         (hyperfine_df['n4'] == indices[3])
                                         ]

            if len(hyperfine_row) == 0:
                raise SimphonyError('Invalid carbon atom indices.')

            C_idx = idx+1 if len(carbon_atom_indices) > 1 else ''

            spin_C = Spin(
                dimension=2,
                name='C{}'.format(C_idx),
                qubit_subspace=(-1 / 2, 1 / 2),
                gyromagnetic_ratio=CARBON13_GYROMAGNETIC_RATIO,
                zero_field_splitting=0.
            )
            model.add_spin(spin_C)

            hyperfine_C = Interaction(spin_e, spin_C)

            hyperfine_C.xx = float(hyperfine_row.Axx.values[0])
            hyperfine_C.xy = float(hyperfine_row.Axy.values[0])
            hyperfine_C.xz = float(hyperfine_row.Axz.values[0])
            hyperfine_C.yx = float(hyperfine_row.Axy.values[0])
            hyperfine_C.yy = float(hyperfine_row.Ayy.values[0])
            hyperfine_C.yz = float(hyperfine_row.Ayz.values[0])
            hyperfine_C.zx = float(hyperfine_row.Axz.values[0])
            hyperfine_C.zy = float(hyperfine_row.Ayz.values[0])
            hyperfine_C.zz = float(hyperfine_row.Azz.values[0])

            if interaction is True:
                model.add_interaction(hyperfine_C)

    model.calculate_hamiltonians()
    model = default_rotating_frame(model, **kwargs)

    return model


def default_rotating_frame(model: Model,
                           electron_spin_name: str = 'e',
                           electron_spin_state: str = '1',
                           nuclear_spin_state: str = 'avg') -> Model:
    """Set the rotating frame frequencies for all spins in the model.

    The rotating frame frequency of the electron spin is set to its qubit splitting, either:

      - averaged over all nuclear spin configurations (if ``nuclear_spin_state == 'avg'``), or
      - conditioned on the nuclear spin qubit(s) state(s) (``'0'`` or ``'1'``).

    The rotating frame frequencies of all nuclear spins are set to their respective qubit splittings,
    assuming the electron spin is in the qubit state ``'0'`` or ``'1'``, and:

      - averaged over all other nuclear spins (if ``nuclear_spin_state == 'avg'``), or
      - conditioned on the nuclear qubit state ``'0'`` or ``'1'`` for the remaining nuclear spins.

    Args:
        model: The model to which the rotating frame frequencies are assigned.
        electron_spin_name: The name of the electron spin in the model.
        electron_spin_state: ``'0'`` or ``'1'``, specifying the electron qubit state to condition on.
        nuclear_spin_state: ``'0'``, ``'1'``, or ``'avg'`` — how to treat nuclear spin states when conditioning.

    Returns:
        The updated model with the assigned rotating frame frequencies.

    Raises:
        SimphonyError: If ``electron_spin_state`` or ``nuclear_spin_state`` is invalid.

    """

    if electron_spin_state in ['0', '1']:
        q = int(electron_spin_state)
        electron_spin_state = model.spin(electron_spin_name).qubit_subspace[q]
    else:
        raise SimphonyError("electron_spin_state must be '0' or '1'.")

    if nuclear_spin_state in ['0', '1']:
        q_n = int(nuclear_spin_state)
    elif nuclear_spin_state == 'avg':
        pass
    else:
        raise SimphonyError("nuclear_spin_state must be '0', '1' or 'avg'.")

    for spin in model.spins:
        if spin.name == electron_spin_name:
            rest_quantum_nums = {}
            if nuclear_spin_state != 'avg':
                for nuclear_spin in model.spins:
                    if nuclear_spin.name != electron_spin_name:
                        rest_quantum_nums[nuclear_spin.name] = nuclear_spin.qubit_subspace[q_n]

            spin.rotating_frame_frequency = model.splitting_qubit(spin_name=spin.name,
                                                                  rest_quantum_nums=rest_quantum_nums)

        else:
            rest_quantum_nums = {electron_spin_name: electron_spin_state}
            if nuclear_spin_state != 'avg':
                for rest_nuclear_spin in model.spins:
                    if rest_nuclear_spin != spin and rest_nuclear_spin.name != electron_spin_name:
                        rest_quantum_nums[rest_nuclear_spin.name] = rest_nuclear_spin.qubit_subspace[q_n]

            spin.rotating_frame_frequency = model.splitting_qubit(spin_name=spin.name,
                                                                  rest_quantum_nums=rest_quantum_nums)

    return model

# def default_rotating_frame(model: Model,
#                            electron_spin_name: str = 'e') -> Model:
#     """Set the default rotating frame for a model.
#
#     The rotating frame frequency of the electron spin is set to its splitting, averaged over all nuclear spin
#     configurations. The nuclear spin rotating frame frequencies are set to their respective splittings, assuming the
#     electron spin is in its qubit state :math:`\\ket{1}` and averaging over all other nuclear spin configurations.
#
#     Arguments:
#         model: The model to which the rotating frame frequencies are assigned.
#         electron_spin_name: The name of the electron spin.
#
#     Returns:
#         The updated model with the assigned rotating frame frequencies.
#
#     """
#
#     electron_spin_qubit_1 = model.spin(electron_spin_name).qubit_subspace[1]
#     for spin in model.spins:
#         if spin.name == electron_spin_name:
#             spin.rotating_frame_frequency = model.splitting_qubit(spin_name=spin.name)
#         else:
#             spin.rotating_frame_frequency = model.splitting_qubit(spin_name=spin.name, rest_quantum_nums={electron_spin_name: electron_spin_qubit_1})
#
#     return model