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

# This section overrides specific parts of Qiskit Dynamics 0.5.1 to enable automatic differentiation for Simphony.

import sys
qiskit_dynamics = sys.modules.get('qiskit_dynamics')

########################################################################################################################

from typing import Callable, Optional, Tuple
from warnings import warn
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from qiskit_dynamics.arraylias import ArrayLike, requires_array_library

try:
    import jax
    from jax import vmap
    import jax.numpy as jnp
    from jax.lax import scan, cond, associative_scan
    from jax.scipy.linalg import expm as jexpm
except ImportError:
    pass

from qiskit_dynamics.solvers.solver_utils import trim_t_results

########################################################################################################################

from qiskit_dynamics.solvers.fixed_step_solvers import get_exponential_take_step, fixed_step_solver_template_jax

##### solvers/fixed_step_solvers.py ####################################################################################

@requires_array_library("jax")
def jax_expm_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
    magnus_order: int = 1,
):
    """Fixed-step size matrix exponential based solver implemented with ``jax``.
    Solves the specified problem by taking steps of size no larger than ``max_dt``.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.
        magnus_order: The expansion order in the Magnus method. Only orders 1, 2, and 3
            are supported.

    Returns:
        OdeResult: Results object.
    """
    def jexpm_64(A):
        return jexpm(A, max_squarings=64)
    take_step = get_exponential_take_step(magnus_order, expm_func=jexpm_64)

    return fixed_step_solver_template_jax(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )

qiskit_dynamics.solvers.fixed_step_solvers.jax_expm_solver = jax_expm_solver

########################################################################################################################

@requires_array_library("jax")
def jax_expm_parallel_solver(
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
    magnus_order: int = 1,
):
    """Parallel version of :func:`jax_expm_solver` implemented with JAX parallel operations.

    Args:
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.
        magnus_order: The expansion order in the Magnus method. Only orders 1, 2, and 3
            are supported.

    Returns:
        OdeResult: Results object.
    """
    def jexpm_64(A):
        return jexpm(A, max_squarings=64)
    take_step = get_exponential_take_step(magnus_order, expm_func=jexpm_64, just_propagator=True)

    return fixed_step_lmde_solver_parallel_template_jax(
        take_step, generator=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )

qiskit_dynamics.solvers.fixed_step_solvers.jax_expm_parallel_solver = jax_expm_parallel_solver

########################################################################################################################

def fixed_step_lmde_solver_parallel_template_jax(
    take_step: Callable,
    generator: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    max_dt: float,
    t_eval: Optional[ArrayLike] = None,
):
    """Parallelized and LMDE specific version of fixed_step_solver_template_jax.

    Assuming the structure of an LMDE:
    * Computes all propagators over each individual time-step in parallel using ``jax.vmap``.
    * Computes all propagators from t_span[0] to each intermediate time point in parallel
      using ``jax.lax.associative_scan``.
    * Applies results to y0 and extracts the desired time points from ``t_eval``.

    The above logic is slightly varied to save some operations is ``y0`` is square.

    The signature of ``take_step`` is assumed to be:
        - generator: A generator :math:`G(t)`.
        - t: The current time.
        - h: The size of the step to take.

    It returns:
        - y: The state of the DE at time t + h.

    Note that this differs slightly from the other template functions, in that
    ``take_step`` does not take take in ``y``, the state at time ``t``. The
    parallelization procedure described above uses the initial state being the identity
    matrix for each time step, and thus it is unnecessary to supply this to ``take_step``.

    Args:
        take_step: Fixed step integration rule.
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    # warn the user that the parallel solver will be very slow if run on a cpu
    if jax.default_backend() == "cpu":
        warn(
            """JAX parallel solvers will likely run slower on CPUs than non-parallel solvers.
            To make use of their capabilities it is recommended to use a GPU.""",
            stacklevel=2,
        )

    y0 = jnp.array(y0)

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    # set up time information for computing propagators in parallel
    all_times_list = []  # all stepping points
    all_h_list = []  # step sizes for each point above
    t_list_locations = [0]  # ordered list of locations in all_times that are in t_list

    for t, h, n_steps in zip(t_list, h_list, n_steps_list):
        all_times_list.append(t + h * jnp.arange(n_steps))
        all_h_list.append(h * jnp.ones(n_steps))
        t_list_locations.append(t_list_locations[-1] + n_steps)

    # Concatenate the lists into JAX arrays
    all_times = jnp.concatenate(all_times_list)
    all_h = jnp.concatenate(all_h_list)
    t_list_locations = jnp.array(t_list_locations)

    # compute propagators over each time step in parallel
    step_propagators = vmap(lambda t, h: take_step(generator, t, h))(all_times, all_h)

    # multiply propagators together in parallel
    ys = None

    def reverse_mul(A, B):
        return jnp.matmul(B, A)

    if y0.ndim == 2 and y0.shape[0] == y0.shape[1]:
        # if square, append y0 as the first step propagator, scan, and extract
        intermediate_props = associative_scan(
            reverse_mul, jnp.append(jnp.array([y0]), step_propagators, axis=0), axis=0
        )
        ys = intermediate_props[t_list_locations]
    else:
        # if not square, scan propagators, extract relevant time points, multiply by y0,
        # then prepend y0
        intermediate_props = associative_scan(reverse_mul, step_propagators, axis=0)
        # intermediate_props doesn't include t0, so shift t_list_locations when extracting
        intermediate_y = intermediate_props[t_list_locations[1:] - 1] @ y0
        ys = jnp.append(jnp.array([y0]), intermediate_y, axis=0)

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_eval)

qiskit_dynamics.solvers.fixed_step_solvers.fixed_step_lmde_solver_parallel_template_jax = fixed_step_lmde_solver_parallel_template_jax

########################################################################################################################

def get_fixed_step_sizes(
    t_span: ArrayLike, t_eval: ArrayLike, max_dt: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Merge ``t_span`` and ``t_eval``, and determine the number of time steps and
    and step sizes (no larger than ``max_dt``) required to fixed-step integrate between
    each time point.

    Args:
        t_span: Total interval of integration.
        t_eval: Time points within t_span at which the solution should be returned.
        max_dt: Max size step to take.

    Returns:
        Tuple[Array, Array, Array]: with merged time point list, list of step sizes to take
        between time points, and list of corresponding number of steps to take between time steps.
    """
    # time args are non-differentiable
    t_span = jnp.array(t_span)
    max_dt = jnp.array(max_dt)
    t_list = jnp.array(merge_t_args(t_span, t_eval))

    # set the number of time steps required in each interval so that
    # no steps larger than max_dt are taken
    delta_t_list = jnp.diff(t_list)
    n_steps_list = jnp.abs(delta_t_list / max_dt).astype(int)

    # correct potential rounding errors
    for idx, (delta_t, n_steps) in enumerate(zip(delta_t_list, n_steps_list)):
        if n_steps == 0:
            n_steps_list = n_steps_list.at[idx].set(1)
        # absolute value to handle backwards integration
        elif jnp.abs(delta_t / n_steps) / max_dt > 1 + 1e-15:
            n_steps_list = n_steps_list.at[idx].set(n_steps + 1)

    # step size in each interval
    h_list = jnp.array(delta_t_list / n_steps_list)

    return t_list, h_list, n_steps_list

qiskit_dynamics.solvers.fixed_step_solvers.get_fixed_step_sizes = get_fixed_step_sizes

##### solvers/solver_utils.py ##########################################################################################

def merge_t_args(t_span: ArrayLike, t_eval: Optional[ArrayLike] = None) -> jnp.ndarray:
    """Merge ``t_span`` and ``t_eval`` into a single array.

    Validition is similar to scipy ``solve_ivp``: ``t_eval`` must be contained in ``t_span``, and be
    increasing if ``t_span[1] > t_span[0]`` or decreasing if ``t_span[1] < t_span[0]``.

    Note: this is done explicitly with ``numpy``, and hence this is not differentiable or compilable
    using jax.

    If ``t_eval is None`` returns ``t_span`` with no modification.

    Args:
        t_span: Interval to solve over.
        t_eval: Time points to include in returned results.

    Returns:
        np.ndarray: Combined list of times.

    Raises:
        ValueError: If one of several validation checks fail.
    """

    if t_eval is None:
        return t_span

    t_span = jnp.array(t_span)

    t_min = jnp.min(t_span)
    t_max = jnp.max(t_span)
    t_direction = jnp.sign(t_span[1] - t_span[0])

    t_eval = jnp.array(t_eval)

    if t_eval.ndim > 1:
        raise ValueError("t_eval must be 1 dimensional.")

    if jnp.min(t_eval) < t_min or jnp.max(t_eval) > t_max:
        raise ValueError("t_eval entries must lie in t_span.")

    diff = jnp.diff(t_eval)

    if jnp.any(t_direction * diff < 0.0):
        raise ValueError("t_eval must be ordered according to the direction of integration.")

    # add endpoints
    t_eval = jnp.append(jnp.append(t_span[0], t_eval), t_span[1])

    return t_eval

qiskit_dynamics.solvers.solver_utils.merge_t_args = merge_t_args