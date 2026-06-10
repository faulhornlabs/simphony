==============
Basic concepts
==============

Introduction
------------

At the highest level, a Simphony :ref:`model <simphony_model_Model>` is a coupled-spin system assembled from a small
number of component types. :ref:`Spins <simphony_components_BaseSpin>` define the individual quantum objects and their
qubits. :ref:`Static fields <simphony_components_StaticField>` describe static external fields, while
:ref:`driving fields <simphony_components_BaseDrivingField>` describe time-dependent control fields.
:ref:`Interactions <simphony_components_Interaction>` couple different spins to one another. From these ingredients,
Simphony constructs the model, derives quantities such as the eigensystem and driving operators, and simulates time
evolution under pulse sequences.

Coordinate system and axis conventions
--------------------------------------

Simphony represents the full model in a global laboratory frame. Quantities attached to a
:ref:`spin <simphony_components_BaseSpin>`, :ref:`static field <simphony_components_StaticField>`, or
:ref:`driving field <simphony_components_BaseDrivingField>`, as well as :ref:`interactions
<simphony_components_Interaction>`, are interpreted in that same frame.
Each spin may additionally define an :ref:`anisotropy axis <simphony_components_BaseSpin_anisotropy_axis>` and a
:ref:`quantization axis <simphony_components_BaseSpin_quantization_axis>`. The anisotropy axis fixes the principal
direction of local terms such as zero-field splitting or quadrupole splitting, while the quantization axis sets which
direction is treated as the spin's local z-axis when Simphony builds operators and states for that spin.

For nitrogen-vacancy (NV) models, the z-axis is typically chosen along the N-V axis, and the static magnetic field is
often aligned with it as well. In the packaged single-NV and multi-NV builders, the electron and nuclear anisotropy
axes and the hyperfine tensors are provided in conventions consistent with the chosen NV orientations, and multi-NV
orientations are rotated into the common lab frame before the full model is assembled. For the exact packaged
conventions, see :ref:`default NV model <simphony_defaults_default_nv_model>` and
:ref:`default multi-NV model <simphony_defaults_default_multi_nv_model>`.

Spin-Hamiltonian
----------------

The coupled-spin system is described by a Hamiltonian of the form

.. math::

    \begin{aligned}
    H(t) &= \sum_i \boldsymbol{S}_i^\mathsf{T} \boldsymbol{D}_i \boldsymbol{S}_i
         + \sum_{i < j} \boldsymbol{S}_i^\mathsf{T} \boldsymbol{A}_{ij} \boldsymbol{S}_j \\
         &\quad + \sum_{i \in \text{electron spins}}
    \gamma_i \boldsymbol{S}_i \!\left[\boldsymbol{B}_\text{static} + \boldsymbol{B}_\text{drive}(t)\right]
         - \sum_{i \in \text{nuclear spins}}
    \gamma_i \boldsymbol{S}_i \!\left[\boldsymbol{B}_\text{static} + \boldsymbol{B}_\text{drive}(t)\right]
    \end{aligned}

where

    * :math:`\boldsymbol{S}_i = (S_{x,i}, S_{y,i}, S_{z,i})` are the spin operators,
    * :math:`\boldsymbol{D}_i` is the zero-field splitting (electron spin) or quadrupole tensor (nuclear spin),
    * :math:`\boldsymbol{A}_{ij}` describes the interaction tensor between spins :math:`i` and :math:`j`.
    * :math:`\gamma_i` are the gyromagnetic ratios,
    * :math:`\boldsymbol{B}_\text{static}` is the static magnetic field,
    * :math:`\boldsymbol{B}_\text{drive}(t)` is the time-dependent driving magnetic field,

The first term contains single-spin anisotropy terms such as zero-field splitting and nuclear quadrupole
contributions. The second term contains pairwise couplings such as hyperfine and dipolar interactions. The third and
fourth terms describe Zeeman coupling to the static and driving fields for electron and nuclear spins, respectively;
the nuclear-spin contribution carries the opposite sign convention used by Simphony.

Quantum numbers and qubit subspace
----------------------------------

Each spin carries an ordered set of :ref:`quantum numbers <simphony_components_BaseSpin_quantum_nums>`. Simphony uses
these quantum numbers to build the full product basis of the model, to label product states and eigenstates, and to
accept state specifications such as tuples or dictionaries of per-spin quantum numbers.

Simphony uses :ref:`qubit subspace <simphony_components_BaseSpin_qubit_subspace>` where needed to select or order the
two quantum numbers that define the computational qubit. For :math:`S = 1/2` systems, it fixes the ordering of the two
basis states. For :math:`S \geq 1` systems, it selects two levels that act as the logical :math:`|0\rangle` and
:math:`|1\rangle` states.

Units
-----

Hamiltonian matrix elements and energy splittings are expressed in MHz, while time is measured in
:math:`\mu\text{s}`. This corresponds to the unit system :math:`h = 1` (not :math:`\hbar = 1`) in the Hamiltonian.
Magnetic-field amplitudes are expressed in Tesla (T), and gyromagnetic ratios in MHz/T.

Bases and frames
----------------

The same simulated dynamics can be described in different bases and reference frames. In Simphony, this matters mainly
for state labeling, operator representation, and gate analysis.

Bases:
    * ``product``: Tensor-product basis built from the basis states of the individual spins.
    * ``eigen``: Eigenbasis of the full static Hamiltonian, including mixing from non-secular terms.

Frames:
    * ``lab``: The laboratory frame defined by the original time-dependent Schrödinger equation.
    * ``rotating``: A frame rotating with respect to the lab frame, often used to simplify the interpretation of driven dynamics.

By default, Simphony labels states in the ``product`` basis, while rotating-frame operators can be used when analyzing
or interpreting the simulated evolution.

Eigenstate labeling
-------------------

Simphony labels each eigenstate by the local-:math:`S_z` product-basis state
with which it has the largest overlap. In practice, this means that a tuple of
quantum numbers is used as a convenient label for the eigenstate whose dominant
component comes from that basis state.

This labeling convention is usually intuitive when one local-:math:`S_z`
product-basis state clearly dominates an eigenstate. In strongly mixed,
degenerate, or nearly degenerate cases, however, the interpretation of the
assigned label can be less clear. For those cases, Simphony provides
:ref:`analysis tools <simphony_model_Model_test_labeling>` for inspecting the
local-:math:`S_z` product-basis overlap structure of the eigensystem in more
detail.


Pulse segments
--------------

Simphony simulates pulse sequences by dividing the total time evolution into segments separated by pulse boundaries.
Within each segment, it determines which driving fields are active, discretizes the segment as needed, and propagates
the system accordingly. Certain common cases can be accelerated substantially, for example segments with no active pulse
or segments containing a single pulse with fixed frequency and constant envelope, where Simphony can use its
:ref:`single-sine-wave method <simphony_solvers_solve_time_segment_single_sine_wave>`.


Rotating frame
--------------

The rotating frame is useful for interpreting driven spin dynamics and comparing the simulated evolution to ideal gate
operations. In Simphony, time evolution is constructed from the lab-frame Hamiltonian, while the
:ref:`rotating-frame operator <simphony_model_Model_rotating_frame_operator>` can be used to transform and analyze the
resulting dynamics.

In Simphony, the operator corresponding to the rotating frame is:

.. math::

    U_\text{rotating}(t) = \bigotimes_{i \in \text{spins}} e^{i 2 \pi f_i t \sigma_{z,i} / 2},

where :math:`f_i` is the rotation frequency associated with spin :math:`i`, and :math:`\sigma_{z,i}` is the Pauli-Z
operator on that spin's qubit subspace.


Virtual rotations
-----------------

Virtual rotations are phase shifts applied in software rather than by applying a physical pulse. They do not consume
additional time and are commonly used to change the effective phase reference of subsequent pulses in a pulse sequence.

In Simphony, they are represented through per-spin :ref:`virtual phases <simphony_components_BaseSpin_virtual_phase>`
and the corresponding :ref:`virtual-phase operator <simphony_model_Model_virtual_phases_operator>`, acting as ideal
Z-rotations applied instantly in the qubit subspace.


Tensor product convention
-------------------------

In Simphony, tensor-product operators follow the standard Kronecker-product convention used in many quantum computing
frameworks, as in :ref:`tensorprod <simphony_utils_tensorprod>`: the rightmost operator acts on the first spin in the
register.

For example, in a two-spin model, :math:`A \otimes B` means that :math:`B` acts on the first spin and :math:`A` acts on
the second spin.
