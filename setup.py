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

from setuptools import setup

setup(name='simphony',
      version='0.1',
      description='NV center simulator package',
      author='Qutility @ Faulhorn Labs',
      url='https://github.com/faulhornlabs/simphony-internal',
      packages=['simphony'],
      package_data={},
      python_requires=">=3.9",
      install_requires=[
          "numpy",
          "scipy",
          "pandas",
          "matplotlib",
          "qiskit-dynamics==0.5.1",
          "qiskit<2",
          "jax[cpu]==0.4.30",
          "jupyterlab"
      ],
      extras_require={
        "cuda12": ["jax[cuda12]"]
      }
     )
