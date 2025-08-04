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
* Includes a predefined NV-center model with hyperfine interactions based on the
  [Ivády Group's hyperfine dataset](https://ivadygroup.elte.hu/hyperfine/nv/index.html).

## Technical specifications

* Built on [Qiskit Dynamics](https://qiskit.org/ecosystem/dynamics/) and [JAX](https://jax.readthedocs.io/).
* Supports both CPU and GPU backends.
* Accelerated by Just-in-Time (JIT) compilation via XLA.
* Enables automatic differentiation for gradient-based optimization.

## Future plans

* Add a predefined model for Silicon Carbide (SiC) with hyperfine interactions.
* Support multiple NV centers.
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

### Install directly via pip

You can install directly from GitHub using:
```bash
python3 -m venv env          # Create venv
source env/bin/activate      # Activate
pip install --upgrade pip    # Upgrade pip
pip install git+https://github.com/faulhornlabs/simphony.git
```

With GPU (CUDA 12) support:
```bash
pip install "simphony[cuda12] @ git+https://github.com/faulhornlabs/simphony.git"
```

Note:
* Tutorial notebooks are not included when installing this way.

### Manual installation with tutorial notebooks

1. Enter the installation directory: `cd <install_directory>`
2. Clone the repository:
    ```bash
    git clone git@github.com:faulhornlabs/simphony.git
    ```
3. Enter the repository: `cd simphony`
4. Make a python virtual environment: `python3 -m venv env`
5. Activate it: `source env/bin/activate`
6. Update pip: `pip install --upgrade pip`
7. Install the package:
   * Standard install: `pip install .`
   * Local editable install: `pip install . -e`
   * With GPU support: `pip install .[cuda12]`

## Usage

Usage of `Jupyter Notebook` or `JupyterLab` is highly recommended to explore the functionality of the package. Tutorial
notebooks can be found within the `jupyternbs` directory.
1. Start the JupyterLab by running: `jupyter-lab`
2. Select a notebook from the left panel within the pop-up browser window. (see first `jupyternbs/basic_tutorial.ipynb`)

## Documentation

**Simphony** documentation is available [here](https://qutility.io/simphony/index.html)
