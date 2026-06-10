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

"""Default NV-center models and packaged lattice and hyperfine data."""

import csv
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Union, List, Tuple, Sequence
Num = Union[int, float]

from os.path import dirname as _dirname, join as _join

import numpy as np

from .exceptions import SimphonyError
from .components import ElectronSpin, NuclearSpin, Interaction, StaticField, LinearDrivingField
from .model import Model, RotatingFrameSetter
from .utils import _axis_angle_from_local_z_to_direction, _rotation_matrix_from_axis_angle

# Constants (references in the `default_nv_model` docstring)
ELECTRON_GYROMAGNETIC_RATIO = 28033.1  # MHz/T
ELECTRON_ZERO_FIELD_SPLITTING = 2872.  # MHz

NITROGEN14_GYROMAGNETIC_RATIO = 3.07771  # MHz/T
NITROGEN14_QUADRUPOLE_SPLITTING = -5.01  # MHz
NITROGEN14_HYPERFINE_PARALLEL = -2.14  # MHz
NITROGEN14_HYPERFINE_PERPENDICULAR = -2.70  # MHz

NITROGEN15_GYROMAGNETIC_RATIO = -4.31727  # MHz/T
NITROGEN15_HYPERFINE_PARALLEL = 3.03  # MHz
NITROGEN15_HYPERFINE_PERPENDICULAR = 3.65  # MHz

CARBON13_GYROMAGNETIC_RATIO = 10.7084  # MHz/T


@dataclass(frozen=True)
class NVLatticeSite:
    """Single NV lattice-site entry.

    The entry stores the lattice indices, the ideal and DFT positions, and the
    associated hyperfine tensor data for a single site.

    Args:
        type: Site type (`'C'`, `'N'`, or `'V'`).
        n1: First lattice index.
        n2: Second lattice index.
        n3: Third lattice index.
        n4: Basis-site index within the primitive cell.
        x_lattice: Ideal lattice x-coordinate in nanometers.
        y_lattice: Ideal lattice y-coordinate in nanometers.
        z_lattice: Ideal lattice z-coordinate in nanometers.
        x_dft: Relaxed DFT x-coordinate in nanometers.
        y_dft: Relaxed DFT y-coordinate in nanometers.
        z_dft: Relaxed DFT z-coordinate in nanometers.
        displacement: Euclidean distance between lattice and DFT positions in nanometers.
        distance_from_lattice_center: Distance from the ideal NV lattice center in nanometers.
        Axx: Hyperfine tensor xx-component in MHz.
        Ayy: Hyperfine tensor yy-component in MHz.
        Azz: Hyperfine tensor zz-component in MHz.
        Axy: Hyperfine tensor xy-component in MHz.
        Axz: Hyperfine tensor xz-component in MHz.
        Ayz: Hyperfine tensor yz-component in MHz.
    """

    type: str
    n1: int
    n2: int
    n3: int
    n4: int
    x_lattice: float
    y_lattice: float
    z_lattice: float
    x_dft: float
    y_dft: float
    z_dft: float
    displacement: float
    distance_from_lattice_center: float
    Axx: float
    Ayy: float
    Azz: float
    Axy: float
    Axz: float
    Ayz: float

    def __repr__(self) -> str:
        def _fmt(value: float) -> str:
            return "nan" if np.isnan(value) else f"{value:.6g}"

        has_hyperfine = not np.isnan(self.Axx)
        lines = [
            "NVLatticeSite(",
            f"  type={self.type!r},",
            f"  index=({self.n1}, {self.n2}, {self.n3}, {self.n4}),",
        ]
        if not has_hyperfine:
            lines.append("  hyperfine=nan,")
        else:
            lines.extend(
                [
                    "  hyperfine=(",
                    f"    ({_fmt(self.Axx)}, {_fmt(self.Axy)}, {_fmt(self.Axz)}),",
                    f"    ({_fmt(self.Axy)}, {_fmt(self.Ayy)}, {_fmt(self.Ayz)}),",
                    f"    ({_fmt(self.Axz)}, {_fmt(self.Ayz)}, {_fmt(self.Azz)}),",
                    "  ),",
                ]
            )
        lines.extend(
            [
                f"  lattice_positions=({self.x_lattice:.6g}, {self.y_lattice:.6g}, {self.z_lattice:.6g}),",
                f"  dft_positions=({self.x_dft:.6g}, {self.y_dft:.6g}, {self.z_dft:.6g}),",
                f"  distance_from_lattice_center={self.distance_from_lattice_center:.6g},",
                ")",
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class NVLatticeDatabase(SequenceABC[NVLatticeSite]):
    """NV lattice-site database.

    This object provides sequence-style access to the packaged default NV
    lattice database.

    Args:
        sites: Tuple containing the packaged :class:`NVLatticeSite` entries.
    """

    sites: tuple[NVLatticeSite, ...]

    def __getitem__(self, item):
        return self.sites[item]

    def __len__(self) -> int:
        return len(self.sites)

    def __repr__(self) -> str:
        preview_count = 5

        def _site_summary(site: NVLatticeSite) -> str:
            has_hyperfine = not np.isnan(site.Axx)
            return (
                "NVLatticeSite("
                f"type={site.type!r}, "
                f"index=({site.n1}, {site.n2}, {site.n3}, {site.n4}), "
                f"has_hyperfine={has_hyperfine}, "
                f"distance_from_lattice_center={site.distance_from_lattice_center:.6g}"
                ")"
            )

        preview_lines = [f"  {_site_summary(site)}," for site in self.sites[:preview_count]]
        preview = "\n".join(preview_lines)
        remaining = len(self.sites) - preview_count
        if remaining > 0:
            if preview:
                preview = f"{preview}\n  ...\n  {remaining} more\n  ..."
            else:
                preview = f"  ...\n  {remaining} more\n  ..."
        return f"NVLatticeDatabase(\n{preview}\n)"


def _apply_secular_approximation_to_interaction(interaction: Interaction) -> Interaction:
    """Reduce a builder-created interaction tensor to its secular ``zz`` term."""

    secular_tensor = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, float(interaction.zz)],
    ]
    interaction.tensor = secular_tensor
    return interaction


def _validate_carbon_atom_index(indices: Sequence[Num]) -> Tuple[int, int, int, int]:
    """Validate and unpack a carbon hyperfine database index tuple."""
    if not isinstance(indices, SequenceABC) or isinstance(indices, (str, bytes)):
        raise SimphonyError(
            'Each carbon atom index must be a 4-component sequence (n1, n2, n3, n4).'
        )
    if len(indices) != 4:
        raise SimphonyError(
            'Each carbon atom index must be a 4-component sequence (n1, n2, n3, n4).'
        )

    n1, n2, n3, n4 = indices
    if not all(isinstance(value, (int, np.integer)) for value in (n1, n2, n3, n4)):
        raise SimphonyError('Carbon atom indices n1, n2, n3, and n4 must be integers.')
    if n4 not in (0, 1):
        raise SimphonyError('Carbon atom index n4 must be 0 or 1.')

    return int(n1), int(n2), int(n3), int(n4)


def _validate_nitrogen_isotope(isotope: Optional[int], *, nv_idx: Optional[int] = None) -> Optional[int]:
    """Validate a nitrogen isotope selector used by the default builders."""
    if isotope in (14, 15, None):
        return isotope
    if nv_idx is None:
        raise SimphonyError('Invalid nitrogen isotope, should be 14 or 15 or None')
    raise SimphonyError(f'Invalid nitrogen isotope for NV {nv_idx}: {isotope}. Expected 14, 15, or None.')


def _parse_optional_float(value: str) -> float:
    """Parse a numeric CSV field, keeping blank values as ``nan``."""
    if value.strip() == '':
        return float('nan')
    return float(value)


@lru_cache(maxsize=1)
def _load_full_nv_lattice_sites() -> tuple[NVLatticeSite, ...]:
    """Load the full NV lattice CSV into a cached immutable row sequence."""
    current_dir = _dirname(__file__)
    file_database = _join(current_dir, 'data/hyperfine/nv.csv')
    entries: list[NVLatticeSite] = []

    with open(file_database, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(
                NVLatticeSite(
                    type=(row.get('type') or 'C').strip() or 'C',
                    n1=int(row['n1']),
                    n2=int(row['n2']),
                    n3=int(row['n3']),
                    n4=int(row['n4']),
                    x_lattice=_parse_optional_float(row.get('x_lattice', '')),
                    y_lattice=_parse_optional_float(row.get('y_lattice', '')),
                    z_lattice=_parse_optional_float(row.get('z_lattice', '')),
                    x_dft=_parse_optional_float(row.get('x_dft', '')),
                    y_dft=_parse_optional_float(row.get('y_dft', '')),
                    z_dft=_parse_optional_float(row.get('z_dft', '')),
                    displacement=_parse_optional_float(row.get('displacement', '')),
                    distance_from_lattice_center=_parse_optional_float(
                        row.get('distance_from_lattice_center', '')
                    ),
                    Axx=_parse_optional_float(row['Axx']),
                    Ayy=_parse_optional_float(row['Ayy']),
                    Azz=_parse_optional_float(row['Azz']),
                    Axy=_parse_optional_float(row['Axy']),
                    Axz=_parse_optional_float(row['Axz']),
                    Ayz=_parse_optional_float(row['Ayz']),
                )
            )

    return tuple(entries)


@lru_cache(maxsize=1)
def _load_nv_hyperfine_database() -> dict[Tuple[int, int, int, int], dict[str, float]]:
    """Load the carbon-only NV hyperfine lookup used by the default builders."""
    database: dict[Tuple[int, int, int, int], dict[str, float]] = {}
    for site in _load_full_nv_lattice_sites():
        if site.type != 'C':
            continue
        database[(site.n1, site.n2, site.n3, site.n4)] = {
            'Axx': float(site.Axx),
            'Axy': float(site.Axy),
            'Axz': float(site.Axz),
            'Ayy': float(site.Ayy),
            'Ayz': float(site.Ayz),
            'Azz': float(site.Azz),
        }
    return database


@lru_cache(maxsize=1)
def _load_nv_hyperfine_database_entries() -> NVLatticeDatabase:
    """Cache the public NV lattice-site view separately from the runtime lookup."""
    return NVLatticeDatabase(_load_full_nv_lattice_sites())


def default_nv_hyperfine_database() -> NVLatticeDatabase:
    """Return the full default NV lattice/hyperfine database.

    The returned object is an :class:`NVLatticeDatabase`, which behaves like an
    immutable sequence of :class:`NVLatticeSite` entries. Each entry contains
    the lattice indices, DFT and ideal lattice positions, the distance from
    the ideal lattice center, and the associated hyperfine tensor data.

    Unlike the internal carbon-only lookup used by :func:`default_nv_model` and
    :func:`default_multi_nv_model`, this public database also includes the
    nitrogen and vacancy rows present in the packaged CSV. Those non-carbon
    entries are returned with ``hyperfine=nan`` in their representation and
    ``nan`` tensor components in the underlying dataclass fields.

    The returned :class:`NVLatticeDatabase` can be indexed and iterated like a
    sequence:

    .. code-block:: python

        db = simphony.default_nv_hyperfine_database()
        first_site = db[0]
        carbon_sites = [site for site in db if site.type == 'C']

    Returns:
        Full packaged NV lattice database, including carbon, nitrogen, and
        vacancy entries.
    """
    return _load_nv_hyperfine_database_entries()


def _rotate_tensor_from_local_frame_to_lab_frame(direction: Sequence[Num],
                                                 tensor_local: Sequence[Sequence[Num]]) -> np.ndarray:
    """Rotate a local-frame tensor into the lab frame aligned with ``direction``.

    The local frame is assumed to use ``z`` as its symmetry or reference axis.
    This helper computes the rotation that maps that local ``z`` direction onto
    the supplied lab-frame ``direction`` and applies it as ``R @ tensor @ R.T``.

    Args:
        direction: Target direction of the local ``z`` axis in the lab frame.
        tensor_local: A 3x3 tensor expressed in the local frame.

    Returns:
        The corresponding 3x3 tensor expressed in the lab frame.
    """

    tensor_local = np.asarray(tensor_local, dtype=float)
    if tensor_local.shape != (3, 3):
        raise SimphonyError('tensor_local must be a 3x3 tensor.')

    rotation_axis, angle = _axis_angle_from_local_z_to_direction(direction)
    if np.isclose(angle, 0.0):
        return tensor_local

    rotation = _rotation_matrix_from_axis_angle(rotation_axis, angle)
    return rotation @ tensor_local @ rotation.T


def _rotate_axial_tensor_from_local_frame_to_lab_frame(direction: Sequence[Num],
                                                       parallel: float,
                                                       perpendicular: float) -> np.ndarray:
    """Build and rotate an axially symmetric local-frame tensor into the lab frame.

    The local tensor is constructed as ``diag(perpendicular, perpendicular, parallel)``,
    where the local ``z`` axis is the symmetry axis. That tensor is then rotated
    into the lab frame so that the local ``z`` axis points along ``direction``.

    Args:
        direction: Target direction of the local symmetry axis in the lab frame.
        parallel: Tensor component along the local symmetry axis.
        perpendicular: Tensor component perpendicular to the local symmetry axis.

    Returns:
        The 3x3 tensor expressed in the lab frame.
    """

    tensor_local = np.diag([perpendicular, perpendicular, parallel]).astype(float)
    return _rotate_tensor_from_local_frame_to_lab_frame(direction, tensor_local)


def _is_quantization_axis_vector(value: Any) -> bool:
    if isinstance(value, str):
        return False

    try:
        axis = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False

    return axis.shape == (3,) and np.all(np.isfinite(axis))


def _resolve_quantization_axes(quantization_axis: Any, n_spins: int) -> List[Any]:
    if isinstance(quantization_axis, str) or _is_quantization_axis_vector(quantization_axis):
        return [quantization_axis] * n_spins

    if isinstance(quantization_axis, (SequenceABC, np.ndarray)):
        axes = list(quantization_axis)
        if len(axes) != n_spins:
            raise SimphonyError(
                "quantization_axis must be a single axis or contain exactly "
                f"{n_spins} per-spin axes."
            )
        return axes

    raise SimphonyError("quantization_axis must be a string, a 3-component vector, or a per-spin axis list.")


def default_nv_model(nitrogen_isotope: Optional[int] = None,
                     carbon_atom_indices: Optional[List[tuple]] = None,
                     static_field_strength: float = 0.05,
                     static_field_direction: Sequence[Num] = (0.0, 0.0, 1.0),
                     add_default_driving_fields: bool = True,
                     interaction: bool = True,
                     secular_approximation: bool = False,
                     quantization_axis: Any = 'global_z',
                     qubit_subspace_electron: Tuple[Num, Num] = (0, -1),
                     qubit_subspace_nitrogen: Optional[Tuple[Num, Num]] = None,
                     qubit_subspace_carbon: Tuple[Num, Num] = (-1 / 2, 1 / 2),
                     warn_on_ambiguous_labels: bool = True,
                     rotating_frame: Optional[RotatingFrameSetter] = None) -> Model:
    """Create a default central spin model based on a nitrogen-vacancy center. The electron spin of the nitrogen-vacancy
    center is added by default as ``'e'``. :math:`^{13}\\text{C}` nuclear spins can be added via the
    ``carbon_atom_indices`` argument. The positions of the nuclear spins are used to determine the hyperfine
    tensors, based on the `Ivády Group's hyperfine dataset <https://ivadygroup.elte.hu/hyperfine/nv/index.html>`_.
    In this default model, the NV axis is aligned with the global :math:`z` direction.

    Args:
        nitrogen_isotope: Atomic number of the nitrogen isotope (optional, should be ``14`` or ``15``).
        carbon_atom_indices: List of (:math:`n_1`, :math:`n_2`, :math:`n_3`, :math:`n_4`) tuples.
        static_field_strength: Strength of the static field (in :math:`\\text{T}`), default ``0.05``.
        static_field_direction: Direction of the static field as a 3-component vector, default ``(0, 0, 1)``.
        add_default_driving_fields: If ``True`` (default), add the default linear driving fields
            ``MW_x`` and ``RF_x``. If ``False``, leave the model without driving fields.
        interaction: Whether interaction is presented or not.
        secular_approximation: If ``True``, keep only the builder-added interaction ``zz`` terms.
        quantization_axis: Quantization axis applied to every spin, or a list with one axis per created spin.
            Each axis accepts ``'global_z'``, ``'anisotropy_axis'``, ``'local_static_field'`` or an explicit vector.
        qubit_subspace_electron: Qubit subspace for the NV electron spin. Default is ``(0, -1)``.
        qubit_subspace_nitrogen: Qubit subspace for the nitrogen spin. If ``None`` (default), keeps the current
            isotope-specific defaults: ``(0, -1)`` for :math:`^{14}N` and ``(-1/2, 1/2)`` for :math:`^{15}N`.
        qubit_subspace_carbon: Qubit subspace for carbon-13 spins. Default is ``(-1/2, 1/2)``.
        warn_on_ambiguous_labels: If ``True`` (default), warn when eigenstate labels have low overlap with their
            assigned local-:math:`S_z` product-basis states.
        rotating_frame: Optional rotating-frame setter installed on the returned model.
            If omitted, the builder installs the default ``RotatingFrameSetter()``.

    Return:
        Default model.

    Raise:
        SimphonyError: If invalid nitrogen isotope is given. If invalid carbon nuclear spin indices are given.

    .. Note::

        The Hamiltonian describes the default NV model (nuclear spins enter with a negative sign in the Zeeman term):

        .. math::
            H =& \\underbrace{\\gamma_{e} \\mathbf{B} \\cdot \\mathbf{S} + D S_z^2}_{\\text{electron}} \\\\
            & \\underbrace{-\\gamma_{N} \\mathbf{B} \\cdot \\mathbf{I}_{N} + P I_{N,z}^2 + \\mathbf{S} \\cdot \\mathbf{A}_{N} \\cdot \\mathbf{I}_{N}}_{\\text{nitrogen}} \\\\
            & \\underbrace{-\\sum_{i}{\\gamma_{C} \\mathbf{B} \\cdot \\mathbf{I}_{C}^{(i)} + \\sum_{i}{\\mathbf{S} \\cdot \\mathbf{A}_{C}^{(i)} \\cdot \\mathbf{I}_{C}^{(i)}}}}_{\\text{carbon(s)}},

        where the parameters are the follows:

        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Spin                        | Parameter               | Symbol                 | Value                               |
        +=============================+=========================+========================+=====================================+
        | Electron (:math:`S=1`)      | Gyromagnetic ratio      | :math:`\gamma_e`       | :math:`28.0331\,\\text{GHz/T}` [1]   |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Zero-field splitting    | :math:`D`              | :math:`2.872\,\\text{GHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Nitrogen-14 (:math:`I=1`)   | Gyromagnetic ratio      | :math:`\gamma_{N}`     | :math:`3.07771\,\\text{MHz/T}` [2]   |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Quadrupole splitting    | :math:`P`              | :math:`-5.01\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine perpendicular | :math:`A_{N\perp}`     | :math:`-2.70\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine parallel      | :math:`A_{N\parallel}` | :math:`-2.14\,\\text{MHz}` [1]       |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Nitrogen-15 (:math:`I=1/2`) | Gyromagnetic ratio      | :math:`\gamma_{N}`     | :math:`-4.31727\,\\text{MHz/T}` [2]  |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine perpendicular | :math:`A_{N\perp}`     | :math:`3.65\,\\text{MHz}` [1]        |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        |                             | Hyperfine parallel      | :math:`A_{N\parallel}` | :math:`3.03\,\\text{MHz}` [1]        |
        +-----------------------------+-------------------------+------------------------+-------------------------------------+
        | Carbon-13 (:math:`I=1/2`)   | Gyromagnetic ratio      | :math:`\gamma_C`       | :math:`10.7084\,\\text{MHz/T}` [2]   |
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

    direction_array = np.asarray(static_field_direction, dtype=float)
    if direction_array.shape != (3,):
        raise SimphonyError("static_field_direction must be a 3-component vector.")
    if not np.all(np.isfinite(direction_array)):
        raise SimphonyError("static_field_direction must contain only finite values.")
    direction_norm = np.linalg.norm(direction_array)
    if direction_norm == 0:
        raise SimphonyError("static_field_direction must be a non-zero vector.")
    unit_direction = direction_array / direction_norm
    static_field = StaticField((static_field_strength * unit_direction).tolist())

    model = Model(name='default')
    model.warn_on_ambiguous_labels = bool(warn_on_ambiguous_labels)
    model.add_static_field(static_field)
    if add_default_driving_fields:
        model.add_driving_field(LinearDrivingField(direction=[1.0, 0.0, 0.0], name="MW_x"))
        model.add_driving_field(LinearDrivingField(direction=[1.0, 0.0, 0.0], name="RF_x"))

    nitrogen_isotope = _validate_nitrogen_isotope(nitrogen_isotope)
    n_spins = 1
    if nitrogen_isotope in (14, 15):
        n_spins += 1
    if carbon_atom_indices is not None:
        n_spins += len(carbon_atom_indices)
    quantization_axes = _resolve_quantization_axes(quantization_axis, n_spins)
    spin_idx = 0

    spin_e = ElectronSpin(
        dimension=3,
        name='e',
        qubit_subspace=qubit_subspace_electron,
        gyromagnetic_ratio=ELECTRON_GYROMAGNETIC_RATIO,
        zero_field_splitting=ELECTRON_ZERO_FIELD_SPLITTING,
        quantization_axis=quantization_axes[spin_idx],
        anisotropy_axis=[0., 0., 1.],
    )
    model.add_spin(spin_e)
    spin_idx += 1

    if nitrogen_isotope == 14 or nitrogen_isotope == 15 or nitrogen_isotope is None:

        if nitrogen_isotope == 14:
            spin_N = NuclearSpin(
                dimension=3,
                name='N',
                qubit_subspace=(qubit_subspace_nitrogen if qubit_subspace_nitrogen is not None else (0, -1)),
                gyromagnetic_ratio=NITROGEN14_GYROMAGNETIC_RATIO,
                quadrupole_splitting=NITROGEN14_QUADRUPOLE_SPLITTING,
                quantization_axis=quantization_axes[spin_idx],
                anisotropy_axis=[0., 0., 1.],
            )
            hyperfine_N = Interaction(spin_e, spin_N)
            hyperfine_N.xx = NITROGEN14_HYPERFINE_PERPENDICULAR
            hyperfine_N.yy = NITROGEN14_HYPERFINE_PERPENDICULAR
            hyperfine_N.zz = NITROGEN14_HYPERFINE_PARALLEL

        if nitrogen_isotope == 15:
            spin_N = NuclearSpin(
                dimension=2,
                name='N',
                qubit_subspace=(qubit_subspace_nitrogen if qubit_subspace_nitrogen is not None else (-1 / 2, 1 / 2)),
                gyromagnetic_ratio=NITROGEN15_GYROMAGNETIC_RATIO,
                quadrupole_splitting=0.,
                quantization_axis=quantization_axes[spin_idx],
                anisotropy_axis=[0., 0., 1.],
            )
            hyperfine_N = Interaction(spin_e, spin_N)
            hyperfine_N.xx = NITROGEN15_HYPERFINE_PERPENDICULAR
            hyperfine_N.yy = NITROGEN15_HYPERFINE_PERPENDICULAR
            hyperfine_N.zz = NITROGEN15_HYPERFINE_PARALLEL

        if nitrogen_isotope == 14 or nitrogen_isotope == 15:
            model.add_spin(spin_N)
            spin_idx += 1

            if interaction is True:
                if secular_approximation:
                    hyperfine_N = _apply_secular_approximation_to_interaction(hyperfine_N)
                model.add_interaction(hyperfine_N)

    if carbon_atom_indices is not None:
        hyperfine_db = _load_nv_hyperfine_database()

        for idx, indices in enumerate(carbon_atom_indices):
            n1, n2, n3, n4 = _validate_carbon_atom_index(indices)
            hyperfine_row = hyperfine_db.get((n1, n2, n3, n4))
            if hyperfine_row is None:
                raise SimphonyError(f'Invalid carbon atom indices: {indices}.')

            C_idx = idx+1 if len(carbon_atom_indices) > 1 else ''

            spin_C = NuclearSpin(
                dimension=2,
                name='C{}'.format(C_idx),
                qubit_subspace=qubit_subspace_carbon,
                gyromagnetic_ratio=CARBON13_GYROMAGNETIC_RATIO,
                quadrupole_splitting=0.,
                quantization_axis=quantization_axes[spin_idx],
                anisotropy_axis=[0., 0., 1.],
            )
            model.add_spin(spin_C)
            spin_idx += 1

            hyperfine_C = Interaction(spin_e, spin_C)

            hyperfine_C.xx = hyperfine_row['Axx']
            hyperfine_C.xy = hyperfine_row['Axy']
            hyperfine_C.xz = hyperfine_row['Axz']
            hyperfine_C.yx = hyperfine_row['Axy']
            hyperfine_C.yy = hyperfine_row['Ayy']
            hyperfine_C.yz = hyperfine_row['Ayz']
            hyperfine_C.zx = hyperfine_row['Axz']
            hyperfine_C.zy = hyperfine_row['Ayz']
            hyperfine_C.zz = hyperfine_row['Azz']

            if interaction is True:
                if secular_approximation:
                    hyperfine_C = _apply_secular_approximation_to_interaction(hyperfine_C)
                model.add_interaction(hyperfine_C)

    model.rotating_frame = RotatingFrameSetter() if rotating_frame is None else rotating_frame
    model.sync_rotating_frame(force=True)

    return model

def _dipolar_tensor(r1, r2):
    """Return the electron-electron dipolar interaction tensor in MHz.

    Args:
        r1: Position of the first spin in nanometers.
        r2: Position of the second spin in nanometers.

    Returns:
        Symmetric traceless 3x3 dipolar tensor expressed in the lab frame.
    """

    mu0_over_4pi = 1e-7  # N/A^2
    h = 6.62607015e-34  # Js
    gamma_e_Hz_per_T = ELECTRON_GYROMAGNETIC_RATIO * 1e6  # MHz/T → Hz/T
    m3_to_nm3 = 1e27
    Hz_to_MHz = 1e-6

    DIPOLAR_PREFACTOR = (
            mu0_over_4pi
            * h
            * gamma_e_Hz_per_T ** 2
            * m3_to_nm3
            * Hz_to_MHz
    )

    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    r_vec = r1 - r2
    dist = np.linalg.norm(r_vec)
    if dist == 0:
        raise SimphonyError("positions for different NV centers must be distinct.")

    D0 = DIPOLAR_PREFACTOR / dist**3
    n = r_vec / dist
    I = np.eye(3)
    J = D0 * (I - 3 * np.outer(n, n))
    return J

# --------------------------------------------------------------------------
# Allowed NV crystal orientations (in crystal frame):
# These correspond to the four tetrahedral directions of NV centers in diamond.
# --------------------------------------------------------------------------
ALLOWED_NV_ORIENTATIONS = {
    ( 1,  1,  1): ( 0.000000,  0.000000,  1.000000),
    ( 1, -1, -1): ( 0.000000,  0.942809, -0.333333),
    (-1,  1, -1): (-0.816497, -0.471405, -0.333333),
    (-1, -1,  1): ( 0.816497, -0.471405, -0.333333),
}

def default_multi_nv_model(orientations: List[Tuple[float, float, float]],
                           positions: List[Tuple[float, float, float]],
                           nitrogen_isotopes: Optional[List[Optional[int]]] = None,
                           carbon_atom_indices: Optional[Sequence[Optional[Sequence[Tuple[int, int, int, int]]]]] = None,
                           static_field_strengths: Union[Num, Sequence[Num]] = 0.05,
                           static_field_directions: Optional[Union[Sequence[Num], Sequence[Sequence[Num]]]] = None,
                           add_default_driving_fields: bool = True,
                           include_hyperfine: bool = True,
                           include_dipolar: bool = True,
                           secular_approximation: bool = False,
                           quantization_axis: Any = 'global_z',
                           qubit_subspace_electron: Tuple[Num, Num] = (0, -1),
                           qubit_subspace_nitrogen: Optional[Tuple[Num, Num]] = None,
                           qubit_subspace_carbon: Tuple[Num, Num] = (-1 / 2, 1 / 2),
                           warn_on_ambiguous_labels: bool = True,
                           rotating_frame: Optional[RotatingFrameSetter] = None) -> Model:
    """Create a model containing multiple NV centers with arbitrary orientations and positions.

    Parameters shared with :func:`default_nv_model` are documented there; this docstring focuses on the
    multi-NV-specific extensions.

    Args:
        orientations: List of anisotropy axes for each NV center. Each orientation should be one of ``(1, 1, 1)``,
            ``(1, -1, -1)``, ``(-1, 1, -1)``, or ``(-1, -1, 1)``, representing the possible crystallographic axes of NV
            centers in diamond.
        positions: List of positions (x, y, z) in nanometers for each NV center.
        nitrogen_isotopes: Optional list of nitrogen isotopes (14 or 15) for each NV.
        carbon_atom_indices: Optional per-NV list of carbon-index lists. Each outer element corresponds to one NV
            center and contains the carbon tuples accepted by :func:`default_nv_model`. For a single NV, a flat list
            of tuples is also accepted for convenience.
        static_field_strengths: Scalar or iterable of static field magnitudes (T). A single value creates one global
            :class:`~simphony.components.StaticField` shared by every spin; an iterable must match ``orientations`` and
            yields one field per NV centre (automatically targeting the corresponding electron and nitrogen spins).
        static_field_directions: ``None`` (uses the lab-frame :math:`\hat{z}` axis), a single 3-component direction
            applied to the broadcast field, or ``len(orientations)`` direction vectors aligned with the per-NV strengths.
        add_default_driving_fields: If ``True`` (default), add the default linear driving fields
            ``MW_x`` and ``RF_x``. If ``False``, leave the model without driving fields.
        include_hyperfine: Whether to include the e–N hyperfine coupling.
        include_dipolar: Whether to include e–e dipolar coupling.
        secular_approximation: If ``True``, keep only the builder-added interaction ``zz`` terms.
        quantization_axis: Quantization axis applied to every spin, or a list with one axis per created spin.
            Each axis accepts ``'global_z'``, ``'anisotropy_axis'``, ``'local_static_field'`` or an explicit vector.
        qubit_subspace_electron: Qubit subspace for all NV electron spins. Default is ``(0, -1)``.
        qubit_subspace_nitrogen: Qubit subspace for nitrogen spins. If ``None`` (default), keeps the current
            isotope-specific defaults: ``(0, -1)`` for :math:`^{14}N` and ``(-1/2, 1/2)`` for :math:`^{15}N`.
        qubit_subspace_carbon: Qubit subspace for all carbon-13 spins.
        warn_on_ambiguous_labels: If ``True`` (default), warn when eigenstate labels have low overlap with their
            assigned local-:math:`S_z` product-basis states.
        rotating_frame: Optional rotating-frame setter installed on the returned model.
            If omitted, the builder installs the default ``RotatingFrameSetter()``.

    Returns:
        Multi-NV spin model with orientation and spatially resolved dipolar
        interactions.
    """

    n_nv = len(orientations)
    if len(positions) != n_nv:
        raise SimphonyError("positions must have the same length as orientations.")
    if nitrogen_isotopes is not None and len(nitrogen_isotopes) != n_nv:
        raise SimphonyError("nitrogen_isotopes must have the same length as orientations.")
    if nitrogen_isotopes is not None:
        nitrogen_isotopes = [
            _validate_nitrogen_isotope(isotope, nv_idx=idx + 1)
            for idx, isotope in enumerate(nitrogen_isotopes)
        ]

    if carbon_atom_indices is None:
        carbons_per_nv: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(n_nv)]
    else:
        raw_carbon_indices = list(carbon_atom_indices)
        if n_nv == 1 and raw_carbon_indices and isinstance(raw_carbon_indices[0], tuple):
            carbons_per_nv = [list(raw_carbon_indices)]
        else:
            if len(raw_carbon_indices) != n_nv:
                raise SimphonyError("carbon_atom_indices must have the same length as orientations when provided per NV.")
            carbons_per_nv = []
            for per_nv in raw_carbon_indices:
                if per_nv is None:
                    carbons_per_nv.append([])
                else:
                    carbons_per_nv.append(list(per_nv))

    carbon_names_per_nv = []
    for nv_idx, carbon_indices_for_nv in enumerate(carbons_per_nv, start=1):
        names = []
        for carbon_idx, _ in enumerate(carbon_indices_for_nv, start=1):
            names.append(f"C_{nv_idx}_{carbon_idx}")
        carbon_names_per_nv.append(names)

    strengths_are_sequence = isinstance(static_field_strengths, Sequence)
    if strengths_are_sequence:
        strengths_raw = list(static_field_strengths)
        if not strengths_raw:
            raise SimphonyError("static_field_strengths must contain at least one value.")
        if len(strengths_raw) != n_nv:
            raise SimphonyError("static_field_strengths must contain as many values as orientations or be provided as a scalar.")
        strengths_per_nv = [float(value) for value in strengths_raw]
        scalar_strength = None
    else:
        scalar_strength = float(static_field_strengths)
        strengths_per_nv = [scalar_strength] * n_nv

    if static_field_directions is None:
        directions_are_single = True
        single_direction = np.asarray((0.0, 0.0, 1.0), dtype=float)
        directions_per_nv = [single_direction.copy() for _ in range(n_nv)]
    elif isinstance(static_field_directions, Sequence):
        direction_items = list(static_field_directions)
        if not direction_items:
            raise SimphonyError("static_field_directions must contain at least one vector.")
        first_item = direction_items[0]
        if isinstance(first_item, Sequence):
            if len(direction_items) != n_nv:
                raise SimphonyError("static_field_directions must contain as many vectors as orientations or be provided as a single direction.")
            directions_are_single = False
            directions_per_nv = [np.asarray(vec, dtype=float) for vec in direction_items]
            single_direction = None
        else:
            directions_are_single = True
            single_direction = np.asarray(direction_items, dtype=float)
            directions_per_nv = [single_direction.copy() for _ in range(n_nv)]
    else:
        directions_are_single = True
        single_direction = np.asarray(static_field_directions, dtype=float)
        directions_per_nv = [single_direction.copy() for _ in range(n_nv)]

    static_fields = []
    if not strengths_are_sequence and directions_are_single:
        norm = np.linalg.norm(single_direction)
        if norm == 0:
            raise SimphonyError("static_field_directions must contain non-zero vectors.")
        field_vec = (scalar_strength * single_direction / norm).tolist()
        static_fields.append(StaticField(field_vec))
    else:
        if directions_are_single:
            norm = np.linalg.norm(single_direction)
            if norm == 0:
                raise SimphonyError("static_field_directions must contain non-zero vectors.")
            unit_direction = single_direction / norm
            directions_per_nv = [unit_direction for _ in range(n_nv)]
        else:
            normalized = []
            for direction in directions_per_nv:
                norm = np.linalg.norm(direction)
                if norm == 0:
                    raise SimphonyError("static_field_directions must contain non-zero vectors.")
                normalized.append(direction / norm)
            directions_per_nv = normalized

        for idx in range(n_nv):
            field_vec = (strengths_per_nv[idx] * directions_per_nv[idx]).tolist()
            field = StaticField(field_vec)
            targets = [f"e_{idx + 1}"]
            if nitrogen_isotopes is not None:
                isotope = nitrogen_isotopes[idx]
                if isotope == 14:
                    targets.append(f"N14_{idx + 1}")
                elif isotope == 15:
                    targets.append(f"N15_{idx + 1}")
            targets.extend(carbon_names_per_nv[idx])
            field.target_spins = tuple(targets)
            static_fields.append(field)

    model = Model(name="multi_NV")
    model.warn_on_ambiguous_labels = bool(warn_on_ambiguous_labels)
    for static_field in static_fields:
        model.add_static_field(static_field)
    if add_default_driving_fields:
        model.add_driving_field(LinearDrivingField(direction=[1.0, 0.0, 0.0], name="MW_x"))
        model.add_driving_field(LinearDrivingField(direction=[1.0, 0.0, 0.0], name="RF_x"))

    spins_e = []
    hyperfine_db = None
    if any(carbons for carbons in carbons_per_nv):
        hyperfine_db = _load_nv_hyperfine_database()

    n_spins = n_nv
    if nitrogen_isotopes is not None:
        n_spins += sum(isotope in (14, 15) for isotope in nitrogen_isotopes)
    n_spins += sum(len(carbon_indices_for_nv) for carbon_indices_for_nv in carbons_per_nv)
    quantization_axes = _resolve_quantization_axes(quantization_axis, n_spins)
    spin_idx = 0

    for idx, (orientation, position) in enumerate(zip(orientations, positions)):
        # Ensure orientation is valid
        orientation_key = tuple(int(x) for x in orientation)
        if orientation_key not in ALLOWED_NV_ORIENTATIONS:
            raise SimphonyError(
                f"Invalid NV orientation {orientation_key}. "
                f"Allowed values are: {list(ALLOWED_NV_ORIENTATIONS.keys())}"
            )

        anisotropy_axis = ALLOWED_NV_ORIENTATIONS[orientation_key]

        spin_e = ElectronSpin(
            dimension=3,
            name=f"e_{idx+1}",
            qubit_subspace=qubit_subspace_electron,
            gyromagnetic_ratio=ELECTRON_GYROMAGNETIC_RATIO,
            zero_field_splitting=ELECTRON_ZERO_FIELD_SPLITTING,
            anisotropy_axis=anisotropy_axis,
            quantization_axis=quantization_axes[spin_idx],
        )
        model.add_spin(spin_e)
        spins_e.append(spin_e)
        spin_idx += 1

        isotope = None if nitrogen_isotopes is None else nitrogen_isotopes[idx]

        if isotope == 14:
            spin_N = NuclearSpin(
                dimension=3,
                name=f"N14_{idx+1}",
                qubit_subspace=(qubit_subspace_nitrogen if qubit_subspace_nitrogen is not None else (0, -1)),
                gyromagnetic_ratio=NITROGEN14_GYROMAGNETIC_RATIO,
                quadrupole_splitting=NITROGEN14_QUADRUPOLE_SPLITTING,
                anisotropy_axis=anisotropy_axis,
                quantization_axis=quantization_axes[spin_idx],
            )
            model.add_spin(spin_N)
            spin_idx += 1
            if include_hyperfine:
                hf_tensor = _rotate_axial_tensor_from_local_frame_to_lab_frame(
                    anisotropy_axis,
                    parallel=NITROGEN14_HYPERFINE_PARALLEL,
                    perpendicular=NITROGEN14_HYPERFINE_PERPENDICULAR,
                )
                hf = Interaction(spin_e, spin_N, tensor=hf_tensor)
                if secular_approximation:
                    hf = _apply_secular_approximation_to_interaction(hf)
                model.add_interaction(hf)

        elif isotope == 15:
            spin_N = NuclearSpin(
                dimension=2,
                name=f"N15_{idx+1}",
                qubit_subspace=(qubit_subspace_nitrogen if qubit_subspace_nitrogen is not None else (-0.5, 0.5)),
                gyromagnetic_ratio=NITROGEN15_GYROMAGNETIC_RATIO,
                quadrupole_splitting=0.,
                anisotropy_axis=anisotropy_axis,
                quantization_axis=quantization_axes[spin_idx],
            )
            model.add_spin(spin_N)
            spin_idx += 1
            if include_hyperfine:
                hf_tensor = _rotate_axial_tensor_from_local_frame_to_lab_frame(
                    anisotropy_axis,
                    parallel=NITROGEN15_HYPERFINE_PARALLEL,
                    perpendicular=NITROGEN15_HYPERFINE_PERPENDICULAR,
                )
                hf = Interaction(spin_e, spin_N, tensor=hf_tensor)
                if secular_approximation:
                    hf = _apply_secular_approximation_to_interaction(hf)
                model.add_interaction(hf)

        for carbon_name, indices in zip(carbon_names_per_nv[idx], carbons_per_nv[idx]):
            n1, n2, n3, n4 = _validate_carbon_atom_index(indices)
            hyperfine_row = hyperfine_db.get((n1, n2, n3, n4))
            if hyperfine_row is None:
                raise SimphonyError(f'Invalid carbon atom indices for NV {idx + 1}: {indices}.')

            spin_C = NuclearSpin(
                dimension=2,
                name=carbon_name,
                qubit_subspace=qubit_subspace_carbon,
                gyromagnetic_ratio=CARBON13_GYROMAGNETIC_RATIO,
                quadrupole_splitting=0.,
                anisotropy_axis=anisotropy_axis,
                quantization_axis=quantization_axes[spin_idx],
            )
            model.add_spin(spin_C)
            spin_idx += 1

            hyperfine_tensor_local = np.array([
                [hyperfine_row['Axx'], hyperfine_row['Axy'], hyperfine_row['Axz']],
                [hyperfine_row['Axy'], hyperfine_row['Ayy'], hyperfine_row['Ayz']],
                [hyperfine_row['Axz'], hyperfine_row['Ayz'], hyperfine_row['Azz']],
            ])
            hyperfine_tensor_lab = _rotate_tensor_from_local_frame_to_lab_frame(
                anisotropy_axis,
                hyperfine_tensor_local,
            )

            hyperfine_C = Interaction(spin_e, spin_C, tensor=hyperfine_tensor_lab)
            if include_hyperfine:
                if secular_approximation:
                    hyperfine_C = _apply_secular_approximation_to_interaction(hyperfine_C)
                model.add_interaction(hyperfine_C)

    if include_dipolar and len(spins_e) > 1:
        for i in range(len(spins_e)):
            for j in range(i + 1, len(spins_e)):
                dipol_dipol_tensor = _dipolar_tensor(positions[i], positions[j])
                dip = Interaction(spins_e[i], spins_e[j], tensor=dipol_dipol_tensor)
                if secular_approximation:
                    dip = _apply_secular_approximation_to_interaction(dip)
                model.add_interaction(dip)

    model.rotating_frame = RotatingFrameSetter() if rotating_frame is None else rotating_frame
    model.sync_rotating_frame(force=True)

    return model
