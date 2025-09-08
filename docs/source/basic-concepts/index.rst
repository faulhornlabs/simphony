==============
Basic concepts
==============

Coordinate system
-----------------

In the case of the nitrogen-vacancy (NV) center, the z-axis is aligned parallel to the N-V axis, and typically, the static
magnetic field is also aligned with this axis. While there is freedom in choosing the orientation of the x-y plane, it
is essential to ensure that the relative positions of the nuclear spins (as reflected in the corresponding hyperfine
tensor) and the orientations of the driving fields are properly aligned.

Simphony provides a default NV model to which one can add carbon-13 nuclear spins using lattice indices. To identify the
choice of the coordinate system, see the documentation of the :ref:`default NV model <default_nv_model>`.

Spin-Hamiltonian
----------------

The central spin system can be described by the following Hamiltonian:

.. math::

    H = \sum_{i}{\gamma_i \boldsymbol{S_i}\left[\boldsymbol{B}_\text{static}+\boldsymbol{B}_\text{drive}(t)\right]}
      + \sum_{i}{\Delta_i S_{z,i}^2}
      + \sum_{i,j}{\boldsymbol{S_i A_{ij} S_j}},

where

    * :math:`\boldsymbol{S_i} = (S_{x,i}, S_{y,i}, S_{z,i})` are the spin operators,
    * :math:`\gamma_i` are the gyromagnetic ratios,
    * :math:`\boldsymbol{B}_\text{static}` is the static magnetic field,
    * :math:`\boldsymbol{B}_\text{drive}(t)` is the time-dependent driving magnetic field,
    * :math:`\Delta_i` are the zero-field splitting or nuclear quadrupole parameters,
    * :math:`\boldsymbol{A_{ij}}` describes the interaction (e.g., hyperfine or dipolar) between the i-th and j-th spin.

The first term represents the Zeeman interaction with static and driving fields. The second term accounts for
zero-field splitting (for spin-1) and nuclear quadrupole effects. The last term describes interactions between spins,
such as hyperfine coupling between an electron spin and nearby nuclear spins.

Note that the Hamiltonian above does not treat either spin as central; in practice, however, it is recommended to
consider the first spin as the central spin. Furthermore, in this Hamiltonian, *all Zeeman terms have a positive sign*,
unlike the usual convention where the terms corresponding to nuclear spins have a negative sign. As a result, this
introduces a **sign difference** in the gyromagnetic ratios for the nuclear spins.

The goal of Simphony is to make it easy to create such spin models, either by using the default one or by building a
custom model, and to set the time dependence of the driving magnetic field(s) through a sequence of pulses to coherently
control the coupled spin system.

Quantum numbers and qubit subspace
----------------------------------

The spins in the central spin register are typically spin-1/2 or spin-1. For quantum computing applications, it is
necessary to define a qubit subspace within the Hilbert space of a spin-1 system. Similarly, for spin-1/2 systems, the
ordering of basis states must be specified to correctly define the computational basis for the qubits.

Units
-----

Frequencies are expressed in MHz, while time is measured in :math:`\mu\text{s}`. Energies are given in terms of
frequency, which corresponds to the unit system :math:`h = 1` in the Hamiltonian. The strength of magnetic fields is
measured in Tesla (T). Gyromagnetic ratios are measured in MHz/T.

Bases and frames
----------------

By applying pulses to the spin system, nontrivial time evolution occurs. The goal is to generate gates by using
appropriate pulse sequences. A common question is in which basis and reference frame the time evolution realizes the
desired ideal gate.

Bases:
    * ``product``: Basis states of each spin and their tensor product states.
    * ``eigen``: Eigenstates of the full spin Hamiltonian, which include the effects of non-secular terms such as hyperfine interactions.

Frames:
    * ``lab``: The laboratory frame governed by the Schrödinger equation.
    * ``rotating``: A frame rotating with respect to the lab frame, often chosen to simplify systems with time-dependent Hamiltonians.

In Simphony, the ``product`` basis and the ``rotating`` frame are used as defaults.


Pulse segments
--------------

Simphony is designed to simulate pulse sequences that consist of microwave and radio-frequency pulses, usually acting
alternately. During simulation, Simphony determines the simulation segments, where each segment is separated by pulse
boundaries—that is, whenever a pulse starts or ends, a new simulation segment begins—and identifies which pulses are
active within each segment. The simulation then proceeds through these time segments, discretizing them and computing
the evolution by exponentiating the corresponding Hamiltonian for each piece.

In certain cases, the computation can be significantly accelerated. For example, if there is no active pulse, or if there
is a single pulse with a fixed frequency and constant envelope. In the latter case, Simphony applies an optimized method
that performs the time evolution only for a single sinusoidal wave and then computes the full time evolution for the
entire pulse from this result.


Rotating frame
--------------

The rotating frame is often introduced to simplify the interpretation of gates implemented by pulse sequences. Currently,
Simphony performs all computations in the lab frame, but the results can be analyzed in a rotating frame, which is
commonly used when defining and interpreting quantum gates.

In Simphony, the operator corresponding to the rotating frame is:

.. math::

    U_\text{rotating}(t) = \prod_{i \in \text{spins}} e^{i 2 \pi f_i t \sigma_{z,i} / 2},

where :math:`f_i` is the rotation frequency associated with spin :math:`i`, and :math:`\sigma_{z,i}` denotes the Pauli-Z
operator acting on the corresponding qubit subspace of spin :math:`i`.


Virtual rotations
-----------------

Virtual rotations are phase shifts applied in software rather than by applying a physical pulse. These rotations do
not require additional time and are commonly used to adjust the effective phase of subsequent pulses in a pulse sequence.

In Simphony, virtual rotations are represented as ideal Z-rotations applied instantly in the qubit subspace of the
corresponding spin.


Tensor product convention
-------------------------

In multi-spin systems, the ordering of operators in the tensor product follows a fixed convention. In Simphony, the
rightmost operator acts on the first spin in the register, consistent with the standard Kronecker product ordering used
in quantum computing frameworks.
