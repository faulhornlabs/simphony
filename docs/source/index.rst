.. title:: Simphony

Simphony
========

Introduction
------------

Simphony is an open-source Python package for simulating spin dynamics in central spin systems, with a focus on
nitrogen-vacancy (NV) centers coupled to nuclear spins. It provides a modular and efficient framework for building
quantum registers, designing pulse sequences, and analyzing system dynamics for quantum information and sensing
applications.

Key features
------------

* Build central spin registers by adding spins, interactions, and external fields.
* Simulate time evolution under pulse sequences to obtain the full unitary operator.
* Compute and visualize expectation values for chosen operators and initial states.
* Calculate process matrices in multiple bases and frames.
* Evaluate average gate fidelity against ideal operations.
* Include local quasi-static noise models to study error effects.
* Includes a predefined NV-center model with hyperfine interactions based on the
  `Ivády Group's hyperfine dataset <https://ivadygroup.elte.hu/hyperfine/nv/index.html>`_.

Technical specifications
------------------------

* Built on `qiskit-dynamics <https://qiskit.org/ecosystem/dynamics/>`_ and `jax <https://jax.readthedocs.io/>`_.
* Supports both CPU and GPU backends.
* Accelerated by Just-in-Time (JIT) compilation via XLA.
* Enables automatic differentiation for gradient-based optimization.

Future Plans
------------

* Add a predefined model for Silicon Carbide (SiC) with hyperfine interactions
* Support multiple NV centers
* Integrate photophysics dynamics, including initialization and readout
* Include Lindblad-type noise models

Documentation
-------------

.. toctree::
    :maxdepth: 1

    Basic Concepts <basic-concepts/index>
    API references <api-refs/index>
    Tutorials <tutorials/index>
    GitHub <https://github.com/faulhornlabs/simphony>
