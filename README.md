# Simphony

Simphony is an open-source Python package from [Qutility](https://www.qutility.io/) for simulating spin dynamics in central spin systems, with a focus on
nitrogen-vacancy (NV) centers coupled to nuclear spins. It provides a modular and efficient framework for building
quantum registers, designing pulse sequences, and analyzing system dynamics for quantum information and sensing
applications.

## Key features

* Build central spin registers by adding spins, interactions, and external fields.
* Simulate time evolution under pulse sequences to obtain the full unitary operator.
* Compute and visualize expectation values for chosen operators and initial states.
* Calculate process matrices in multiple bases and frames.
* Evaluate average gate fidelity against ideal operations.
* Include local quasi-static noise models to study error effects.
* Support simulations with multiple NV centers.
* Includes a predefined NV-center model with hyperfine interactions based on the
  [Ivády Group's hyperfine dataset](https://ivadygroup.elte.hu/hyperfine/nv/index.html).

## Technical specifications

* Uses internal time-evolution solvers based on NumPy and [JAX](https://jax.readthedocs.io/).
* Supports both CPU and GPU backends.
* Accelerated by Just-in-Time (JIT) compilation via XLA.
* Enables automatic differentiation for gradient-based pulse optimization.

## Future plans

* Add a predefined model for Silicon Carbide (SiC) with hyperfine interactions.
* Integrate photophysics dynamics, including initialization and readout.
* Include Lindblad-type noise models.

## Requirements

* Python >=3.9
* CUDA 12 for GPU support

How to check your Python and CUDA versions:
```bash
python3 --version
nvidia-smi | grep CUDA
```

## Installation

Simphony runs on CPU by default, but can achieve significant speedups with GPU acceleration. To enable GPU support, you
need an NVIDIA GPU and a CUDA 12 environment. For more details, see the [JAX GPU installation guide](https://docs.jax.dev/en/latest/installation.html#install-nvidia-gpu).

### Install directly from GitHub

You can install the package directly from GitHub:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/faulhornlabs/simphony.git"
```

With GPU (CUDA 12) support:
```bash
pip install "simphony[cuda12] @ git+https://github.com/faulhornlabs/simphony.git"
```

Installing with `pip install git+...` gives you the package only. The tutorial
notebooks and local documentation sources are not kept in your working
directory.

### Clone and install locally

If you want the full repository, including `jupyternbs/`, clone and install it
manually:

```bash
git clone https://github.com/faulhornlabs/simphony.git
cd simphony
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

Editable install for development:
```bash
pip install -e .
```

Optional extras:
- `cuda12`: GPU-enabled JAX support for NVIDIA CUDA 12 environments.
  ```bash
  pip install ".[cuda12]"
  ```
- `notebooks`: Dependencies for running the tutorial notebooks locally.
  ```bash
  pip install ".[notebooks]"
  ```
- `docs`: Dependencies for building the documentation locally.
  ```bash
  pip install ".[docs]"
  ```

Combine them as needed:
```bash
pip install ".[notebooks,cuda12]"
```

## Usage

Usage of `Jupyter Notebook` or `JupyterLab` is highly recommended to explore the
functionality of the package. Tutorial notebooks can be found within the
`jupyternbs` directory.
1. Start the JupyterLab by running: `jupyter-lab`
2. Select a notebook from the left panel within the pop-up browser window. For a
   first example, open `jupyternbs/tutorial_basic.ipynb`.

## Documentation

**Simphony** documentation is available [here](https://qutility.io/simphony/index.html)
