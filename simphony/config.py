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

"""Configurations."""

from .exceptions import SimphonyError, SimphonyWarning

from warnings import warn

import jax

import numpy as np
import jax.numpy as jnp

import types
import os


class Config:
    """Configurations.

    Warning:
        The simulation platform must be configured immediately after importing ``simphony``, without performing any
        other operations beforehand:

        .. code-block:: python

            import simphony
            simphony.Config.set_platform('gpu')

        The default simulation platform is the CPU.

    Note:
        To enable automatic differentiation mode, use:

        .. code-block:: python

            simphony.Config.set_autodiff_mode(True)

        By default, it is disabled.


    Hint:
        ``'retina'`` as matplotlib format is recommended:

        .. code-block:: python

            simphony.Config.set_matplotlib_format('retina')

    """
    _platform = None
    _autodiff_mode = None

    @staticmethod
    def get_platform():
        """Get the current platform of :mod:`jax`."""
        return Config._platform

    @staticmethod
    def set_platform(platform: str):
        """
        Set the simulation platform of :mod:`jax` to ``'cpu'`` or ``'gpu'``.

        Args:
            platform: Simulation platform of :mod:`jax`.
        """

        if platform == 'cpu':
            jax.config.update('jax_platform_name', 'cpu')
        elif platform == 'gpu':
            try:
                jax.devices('gpu')
                jax.config.update('jax_platform_name', 'gpu')
            except RuntimeError:
                warn("GPU backend not available in JAX. Falling back to CPU.", SimphonyWarning, stacklevel=2)
                jax.config.update('jax_platform_name', 'cpu')
                platform = 'cpu'
        else:
            raise SimphonyError("Invalid backend, must be 'cpu' or 'gpu'")

        Config._platform = platform

    @staticmethod
    def set_gpu(gpu_id: int):
        """
        Set the visible GPU device by setting the CUDA_VISIBLE_DEVICES environment variable.

        Warnings:

            Set it before calling ``Config.set_platform('gpu')``.

        Args:
            gpu_id: A single GPU ID.
        """
        if isinstance(gpu_id, int):
            gpu_str = str(gpu_id)
        else:
            raise SimphonyError("gpu_id must be an int")

        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    @staticmethod
    def set_matplotlib_format(matplotlib_format: str):
        """
        Set the format of :mod:`matplotlib`.

        Args:
            matplotlib_format: Format of :mod:`matplotlib`.
        """
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats(matplotlib_format)

    @staticmethod
    def set_autodiff_mode(enable: bool):
        """
        Enable or disable automatic differentiation mode.

        Note:

            If ``True``, sets ``unp`` to :mod:`jax.numpy`. If ``False``, sets ``unp`` to :mod:`numpy`.

        Args:
            enable: ``True`` or ``False``.
        """
        Config._autodiff_mode = enable
        module = jnp if enable else np
        unp.set(module)

    @staticmethod
    def get_autodiff_mode():
        """Returns whether automatic differentiation mode is enabled."""
        return Config._autodiff_mode


class _unpProxy(types.ModuleType):
    """A proxy to switch between numpy and jax.numpy dynamically."""

    def __init__(self):
        super().__init__('unp')
        self._unp = None

    def set(self, module):
        self._unp = module

    def __getattr__(self, name):
        if self._unp is None:
            raise SimphonyError("unp not set. Use Config.set_autodiff_mode() first.")
        return getattr(self._unp, name)


unp = _unpProxy()
