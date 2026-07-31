# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-07-30

### Added

- New notebooks (4–10) covering single-sine-wave pulse methods, Knight-shift modeling, pulsed ODMR, and DC/AC magnetometry techniques for NV-center sensing.

### Changed

- Renumbered example notebooks.

### Fixed

- **GPU optimization**: Vectorized `compute_time_grid` jitter-free step-size construction across batch dimensions for significant speedup on GPU platforms.
- **GPU optimization**: Vectorized local-qubit Z rotating-frame operator computation across time points, reducing memory allocations.
- **GPU optimization**: Vectorized `single_sine_wave` propagator power reconstruction, enabling efficient GPU batch processing without `jnp.unique` overhead.
- **GPU platform defaults**: `Config.set_platform('gpu')` now defaults to `device_id=0` for transparent single-GPU resource isolation, avoiding cross-process GPU conflicts.
- **CircularDrivingField normalization**: Fixed quadrature coefficient normalization in `CircularDrivingField`, correcting pulse amplitude scaling in circular-polarization driving.
- **Autodiff pulse plotting**: Recovered concrete pulse envelopes for visualization after autodiff returns to avoid tracer artifacts.
- **Model.splitting() optional parameter**: Added optional `rest_quantum_nums` parameter to `Model.splitting()`, allowing custom quantum-number subsets for qubit-space splittings.

### Removed

- Removed now-unneeded pulse-rebuild workaround before autodiff plotting (unnecessary after tracer-recovery fix).

## [0.2.0] — 2026-06-10

### Changed

- Major rewrite and expansion of the package built on the v0.1.x foundation.
- Migrated packaging from `setup.py` to `pyproject.toml`.
- Renamed `simulationresult.py` to `analysis.py`, substantially expanding its API
  (`TimeEvolOperator`/`TimeEvolState`/`TimeEvolDensityMatrix`, basis-aware fidelity, leakage,
  and process-matrix analysis).
- Replaced the Qiskit Dynamics compatibility patch (`_qiskit_dynamics_patch.py`) with a native
  `solvers.py` module providing Magnus-based time-evolution solvers (NumPy/JAX/parallel-JAX).
- Consolidated the four unnumbered example notebooks into a numbered 5-notebook tutorial/example
  series (`1_tutorial_basic.ipynb` … `5_example_multiqubit_phase_gate_autodiff.ipynb`).
- Overhauled documentation: new Sphinx theme/branding, expanded API-reference autosummaries.

### Added

- Packaged NV hyperfine-coupling lattice database (`simphony/data/hyperfine/nv.csv`), removing
  the need for external DFT data files.
- GPU acceleration support (CUDA 12 extra).
- Autodiff-compatible pulse optimization via JAX.
- Rotating-frame and basis-aware analysis operators (process matrices, fidelity, leakage).
- Local quasi-static noise modeling (`local_quasistatic_noise` on spins, noise-aware solvers,
  shot-based noise analysis).
- Regression and release-smoke test suite (`tests/`).

## [0.1.2]

### Fixed

- SimulationResult.initial_state method bug fix.

## [0.1.1]

### Fixed

- Skip zero-duration segments in `add_discrete_pulse`.
- Bugfix in the Qiskit Dynamics compatibility patch.

### Changed

- Updated README.md.

## [0.1.0]

### Added

- Initial release of Simphony: a spin-dynamics simulation package for NV-center systems.
- Core modules: `Model` (`model.py`), spin/interaction/field/pulse components
  (`components.py`), global `Config` (`config.py`), default NV model builders and hyperfine
  database (`defaults.py`), simulation-result analysis (`simulationresult.py`).
- Qiskit Dynamics compatibility patch (`_qiskit_dynamics_patch.py`) for time-evolution solving.
- Four example notebooks: `basic-tutorial.ipynb`, `advanced-example-autodiff.ipynb`,
  `advanced-example-default-multiqubit.ipynb`, `advanced-example-noisy-GPU.ipynb`.
- Sphinx-based documentation.

[Unreleased]: https://github.com/faulhornlabs/simphony/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/faulhornlabs/simphony/releases/tag/v0.2.1
[0.2.0]: https://github.com/faulhornlabs/simphony/releases/tag/v0.2.0
[0.1.2]: https://github.com/faulhornlabs/simphony/releases/tag/v0.1.2
[0.1.1]: https://github.com/faulhornlabs/simphony/releases/tag/v0.1.1
[0.1.0]: https://github.com/faulhornlabs/simphony/releases/tag/v0.1.0
