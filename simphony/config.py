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

from typing import ClassVar, Optional

from .exceptions import SimphonyError, SimphonyWarning

import warnings as _warnings
from warnings import warn

import jax
import numpy as np
import jax.numpy as jnp

import types
import os
import sys


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

    _platform: ClassVar[str] = 'cpu'
    _autodiff_mode: ClassVar[bool] = False
    _jax_expm_max_squarings: ClassVar[int] = 16
    _warnings_enabled: ClassVar[bool] = True

    @staticmethod
    def get_platform() -> str:
        """Get the current platform of :mod:`jax`.

        Returns:
            Current JAX platform, typically ``'cpu'`` or ``'gpu'``.
        """
        return Config._platform

    @staticmethod
    def set_platform(platform: str,
                     device_id: Optional[int] = None):
        """
        Set the simulation platform of :mod:`jax` to ``'cpu'`` or ``'gpu'``.

        Note:
            ``'cpu'`` is the safest default for small and medium-sized simulations, quick debugging,
            and workloads where JAX compilation overhead would dominate the actual runtime.

            ``'gpu'`` is typically preferable for larger simulations, batched propagations, and
            repeated time-evolution workloads where the higher arithmetic throughput of JAX on GPU can
            outweigh compilation and data-transfer costs.

            In practice, ``'gpu'`` tends to be most useful when the Hilbert-space dimension is large,
            many propagators must be evaluated, or the same computation is executed repeatedly after
            JIT compilation. For one-off calculations or small models, ``'cpu'`` may still be faster
            overall.

            If the requested GPU backend is unavailable, Simphony falls back to ``'cpu'``.

        Args:
            platform: Simulation platform of :mod:`jax`. Accepted case-insensitive values are
                ``'cpu'`` and ``'gpu'``.
            device_id: Optional GPU device id to expose via ``CUDA_VISIBLE_DEVICES``.
        """

        if not isinstance(platform, str):
            raise SimphonyError("platform must be 'cpu' or 'gpu'")

        platform = platform.strip().lower()

        if platform == 'cpu':
            jax.config.update('jax_platform_name', 'cpu')
        elif platform == 'gpu':
            if device_id is not None:
                if not isinstance(device_id, int):
                    raise SimphonyError("device_id must be an int")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            try:
                jax.devices('gpu')
                jax.config.update('jax_platform_name', 'gpu')
            except RuntimeError:
                warn("GPU backend not available in JAX. Falling back to CPU.", SimphonyWarning, stacklevel=1)
                jax.config.update('jax_platform_name', 'cpu')
                platform = 'cpu'
        else:
            raise SimphonyError("Invalid backend, must be 'cpu' or 'gpu'")

        Config._platform = platform
        _set_unp_roles()

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

            This setting controls how the role-specific proxies are resolved:
            ``unp_static`` remains backed by :mod:`numpy`. When autodiff mode is enabled,
            ``unp_dynamic`` and ``unp_analysis`` use :mod:`jax.numpy`. When autodiff mode is
            disabled, ``unp_analysis`` falls back to :mod:`numpy`, while ``unp_dynamic`` follows
            the selected execution backend and may still use :mod:`jax.numpy` on GPU.

        Args:
            enable: ``True`` or ``False``.
        """
        Config._autodiff_mode = enable
        _set_unp_roles()

    @staticmethod
    def get_autodiff_mode() -> bool:
        """Return whether automatic differentiation mode is enabled.

        Returns:
            ``True`` if automatic differentiation mode is enabled, otherwise ``False``.
        """
        return Config._autodiff_mode

    @staticmethod
    def set_jax_expm_max_squarings(max_squarings: int):
        r"""Set the maximum number of squarings used by the JAX matrix exponential.

        Args:
            max_squarings: Positive integer limit for the JAX matrix exponential.

        .. note::

            Simphony's JAX-based propagators ultimately call :func:`jax.scipy.linalg.expm`,
            which evaluates the matrix exponential using a scaling-and-squaring scheme together
            with a Padé approximation. The basic idea is to first reduce the matrix norm, compute
            an accurate approximation for the smaller matrix, and then undo the scaling.

            In simplified form, the method uses

            .. math::

                e^A = \left(e^{A / 2^s}\right)^{2^s},

            where :math:`s` is the number of squaring steps.

            For the scaled matrix :math:`A / 2^s`, JAX approximates the exponential by a rational
            Padé form

            .. math::

                e^{A / 2^s} \approx r_m(A / 2^s) = P_m(A / 2^s)\,Q_m(A / 2^s)^{-1},

            with matrix polynomials :math:`P_m` and :math:`Q_m`. In practice, the role of scaling
            is to make :math:`A / 2^s` small enough that this rational approximation is accurate
            and numerically stable.

            JAX estimates how much scaling is needed from the matrix norm and then performs the
            required number of repeated squarings. Roughly speaking,

            .. math::

                s \approx \max\left(0, \left\lceil \log_2 \|A\|_1 \right\rceil - c\right),

            where :math:`c` is an internal dtype-dependent threshold. The ``max_squarings``
            parameter is therefore not a direct accuracy knob for the Padé approximation itself;
            instead, it is an upper bound on how much scaling-and-squaring JAX is allowed to use.

            If the estimated number of squarings exceeds this bound, JAX may refuse the requested
            evaluation and return ``nan`` values rather than silently producing an unreliable
            result. Increasing ``max_squarings`` makes larger-norm matrices admissible, but also
            permits more work in the exponential evaluation.

            In Simphony this matters when the effective generator passed to ``expm`` becomes large,
            for example because of large time steps, strong couplings, large driving amplitudes,
            or Hamiltonians written in units that produce large matrix norms. If you encounter
            unexpected ``nan`` values in JAX-based propagation, this setting is one of the first
            things worth checking.
        """
        if not isinstance(max_squarings, int) or max_squarings <= 0:
            raise SimphonyError('max_squarings must be a positive integer.')
        Config._jax_expm_max_squarings = max_squarings

    @staticmethod
    def get_jax_expm_max_squarings() -> int:
        """Return the configured maximum number of squarings for the JAX matrix exponential.

        Returns:
            Maximum number of squarings used by the JAX matrix exponential.
        """
        return Config._jax_expm_max_squarings

    @staticmethod
    def set_warnings(enable: bool):
        """Enable or disable Simphony warnings.

        Args:
            enable: ``True`` to enable warnings, ``False`` to suppress them.
        """
        if not isinstance(enable, bool):
            raise SimphonyError('enable must be a bool.')
        Config._warnings_enabled = enable
        _apply_warning_filter()

    @staticmethod
    def get_warnings() -> bool:
        """Return whether Simphony warnings are enabled.

        Returns:
            ``True`` if Simphony warnings are enabled, otherwise ``False``.
        """
        return Config._warnings_enabled


def _apply_warning_filter():
    """Apply the current Simphony warning visibility setting."""
    action = 'default' if Config.get_warnings() else 'ignore'
    _warnings.filterwarnings(action, category=SimphonyWarning)


class _unpProxy(types.ModuleType):
    """A proxy to switch between numpy and jax.numpy dynamically."""

    def __init__(self, name='unp', role: Optional[str] = None):
        super().__init__(name)
        self._role = role
        self._roles = {}
        sys.modules[self.__name__] = self
        self.set()

    def set(self):
        platform = Config.get_platform()
        autodiff_state = Config.get_autodiff_mode()

        if autodiff_state:
            static_module = np
            dynamic_module = jnp
            analysis_module = jnp
        else:
            if platform == 'gpu':
                static_module = np
                dynamic_module = jnp
                analysis_module = np
            else:
                static_module = np
                dynamic_module = np
                analysis_module = np

        self._roles = {
            'static': static_module,
            'dynamic': dynamic_module,
            'analysis': analysis_module,
        }

        if self._role is not None:
            # For role-specific proxies (unp_static, etc.)
            self._module = self._roles[self._role]

    def __getitem__(self, role: str):
        try:
            return self._roles[role]
        except KeyError as exc:
            raise SimphonyError(f"Unknown unp role '{role}'.") from exc

    def __getattr__(self, name):
        if self._role is not None:
            return getattr(self._module, name)
        raise AttributeError(name)


# Registry of role-specific proxies
_PROXIES = []


def _register_unp_proxy(name: str, role: str) -> _unpProxy:
    proxy = _unpProxy(name, role)
    _PROXIES.append(proxy)
    return proxy


unp_static = _register_unp_proxy('unp_static', 'static')
unp_dynamic = _register_unp_proxy('unp_dynamic', 'dynamic')
unp_analysis = _register_unp_proxy('unp_analysis', 'analysis')


def _set_unp_roles():
    """Refresh all unp proxies."""
    for proxy in _PROXIES:
        proxy.set()


# Initialize defaults
_apply_warning_filter()
_set_unp_roles()
