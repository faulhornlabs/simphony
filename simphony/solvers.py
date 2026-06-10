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

"""Solvers classes and functions."""
from __future__ import annotations

from .config import unp_dynamic
from .components import Signal, SignalSum
from .exceptions import SimphonyError
from .utils import expm

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import jax.numpy as jnp
from jax import vmap
from jax.lax import scan, cond, associative_scan

uarray = Union[np.ndarray, jnp.ndarray]


class SolverResult:
    """Container for solver outputs that preserves array backend compatibility.

    The object stores both the evaluation grid and simulated states or operators while
    exposing helpers to concatenate segments and stack per-shot batches without
    breaking the underlying NumPy or JAX array type.

    Args:
        ts: Evaluation times.
        ys: Propagated states or operators.
        time_axis: Axis of ``ys`` corresponding to time.
        shot_axis: Optional axis of ``ys`` corresponding to shots.
        unp_inner: NumPy or :mod:`jax.numpy` namespace used to store the result.
    """

    def __init__(self,
                 ts: uarray,
                 ys: uarray,
                 *,
                 time_axis: int,
                 shot_axis: Optional[int] = None,
                 unp_inner: Any):
        self.unp_inner = unp_inner
        self.ts = unp_inner.asarray(ts)
        self.ys = unp_inner.asarray(ys)
        self.time_axis = time_axis
        self.shot_axis = shot_axis

    @classmethod
    def stack_shots(cls, shot_results: Sequence[SolverResult]) -> SolverResult:
        """Stack multiple solver runs along a unified shot axis.

        Args:
            shot_results: Solver results to combine.

        Returns:
            Solver result containing all shots stacked along a common shot axis.
        """
        first = shot_results[0]
        unp_inner = first.unp_inner
        ts = first.ts
        if first.shot_axis is not None:
            ys = unp_inner.concatenate([res.ys for res in shot_results], axis=first.shot_axis)
            return cls(ts, ys, time_axis=first.time_axis, shot_axis=first.shot_axis, unp_inner=unp_inner)
        ys = unp_inner.stack([res.ys for res in shot_results], axis=0)
        new_time_axis = first.time_axis + 1
        return cls(ts, ys, time_axis=new_time_axis, shot_axis=0, unp_inner=unp_inner)

    @classmethod
    def concat_segments(cls, segment_results: Sequence[SolverResult]) -> SolverResult:
        """Concatenate sequential segment solutions while avoiding duplicate time stamps.

        Args:
            segment_results: Segment-wise solver results in chronological order.

        Returns:
            Solver result spanning the full segment sequence.
        """
        first = segment_results[0]
        unp_inner = first.unp_inner
        tail_ts = [seg.ts[1:] for seg in segment_results[1:]]
        tail_ys = [seg.drop_first_time() for seg in segment_results[1:]]
        ts = unp_inner.concatenate([first.ts] + tail_ts)
        ys = unp_inner.concatenate([first.ys] + tail_ys, axis=first.time_axis)
        return cls(ts, ys, time_axis=first.time_axis, shot_axis=first.shot_axis, unp_inner=unp_inner)

    def drop_first_time(self) -> uarray:
        """Return the solution array with the first time step removed.

        Returns:
            Solution array without its first entry along the time axis.
        """
        slicer = [slice(None)] * self.ys.ndim
        slicer[self.time_axis] = slice(1, None)
        return self.ys[tuple(slicer)]

    def final_along_time(self) -> uarray:
        """Return the final value along the designated time axis.

        Returns:
            Final state or operator along the stored time axis.
        """
        slicer = [slice(None)] * self.ys.ndim
        slicer[self.time_axis] = -1
        return self.ys[tuple(slicer)]

    def final_shot(self) -> List[uarray]:
        """Return terminal states for each shot as a list.

        Returns:
            Final states or operators, one entry per shot.
        """
        finals = self.final_along_time()
        if self.shot_axis is None:
            return [finals]
        finals_shot_first = self.unp_inner.moveaxis(finals, self.shot_axis, 0)
        return [finals_shot_first[idx] for idx in range(finals_shot_first.shape[0])]


def compute_time_grid(t_span: uarray,
                      t_eval: Optional[uarray],
                      max_dt: float,
                      jitter: Optional[float] = 0.05,
                      *,
                      unp_inner,
                      rng=None) -> Tuple[uarray, List[uarray], uarray]:
    """Construct integration steps aligned with the evaluation grid.

    Args:
        t_span: Array containing the start and end times of the integration
            window.
        t_eval: Optional monotonic array of evaluation points inside ``t_span``.
        max_dt: Maximum allowed step size used to split each interval.
        jitter: Optional jitter amplitude applied to interior boundaries to
            avoid resonance artefacts. Set to ``None`` to disable.
        unp_inner: Array namespace (:mod:`numpy` or :mod:`jax.numpy`) used to
            build the grid.
        rng: Optional simulation-local NumPy random generator used to sample
            jittered interior step boundaries.

    Returns:
        Tuple ``(t_list, h_nested_list, n_list)`` where ``t_list`` is the
        merged evaluation grid, ``h_nested_list`` contains per-interval step
        sizes, and ``n_list`` records the number of steps taken inside each
        interval.
    """
    if max_dt <= 0:
        raise ValueError("max_dt must be positive.")
    if jitter is not None and not (0 < jitter < 0.5):
        raise ValueError("jitter must be between 0 and 0.5 (or None).")

    t_span_arr = unp_inner.asarray(t_span)
    if t_eval is None:
        t_list = t_span_arr
    else:
        t_eval_arr = unp_inner.asarray(t_eval)
        t_start, t_end = t_span_arr[0], t_span_arr[-1]
        parts = []
        if not unp_inner.isclose(t_eval_arr[0], t_start):
            parts.append(unp_inner.asarray([t_start]))
        parts.append(t_eval_arr)
        if not unp_inner.isclose(t_eval_arr[-1], t_end):
            parts.append(unp_inner.asarray([t_end]))
        t_list = unp_inner.concatenate(parts)

    delta_t = unp_inner.diff(t_list)
    n_list = unp_inner.ceil(unp_inner.abs(delta_t) / max_dt).astype(int)
    n_list = unp_inner.maximum(n_list, 1)

    h_nested_list = []
    for dt, n in zip(delta_t, n_list):
        if n == 1 or not jitter:
            h = unp_inner.full(n, dt / n)
        else:
            if rng is None:
                rng = np.random.default_rng()
            base = unp_inner.linspace(0.0, 1.0, n + 1)
            jitter_np = np.zeros(n + 1)
            jitter_np[1:-1] = (rng.random(n - 1) * 2 - 1) * jitter / n
            jitter_unp = base + unp_inner.asarray(jitter_np)
            h = unp_inner.diff(jitter_unp) * dt
        h_nested_list.append(h)

    return t_list, h_nested_list, n_list


def get_propagator(magnus_order: int) -> Callable:
    """Return a Magnus propagator closure of the requested order.

    Args:
        magnus_order: Magnus expansion order. Supported values are ``1``, ``2``,
            and ``3``.

    Returns:
        Callable ``propagator(G, t0, h)`` that evaluates the generator ``G``
        and returns the evolution operator over the time step ``h``.

    Raises:
        ValueError: If ``magnus_order`` is not one of the supported orders.
    """
    def matrix_commutator(A, B):
        return A @ B - B @ A

    if magnus_order == 1:
        def propagator(G, t0, h):
            return expm(G(t0 + (h / 2)) * h)
    elif magnus_order == 2:
        c1 = 0.5 - np.sqrt(3) / 6
        c2 = 0.5 + np.sqrt(3) / 6
        p2 = np.sqrt(3) / 12

        def propagator(G, t0, h):
            g1 = G(t0 + c1 * h)
            g2 = G(t0 + c2 * h)
            terms = h * (g1 + g2) / 2 + p2 * (h ** 2) * matrix_commutator(g2, g1)
            return expm(terms)
    elif magnus_order == 3:
        d1 = 0.5 - np.sqrt(15) / 10
        d2 = 0.5
        d3 = 0.5 + np.sqrt(15) / 10
        c0 = np.sqrt(15) / 3
        c1 = 10.0 / 3

        def propagator(G, t0, h):
            g1 = G(t0 + d1 * h)
            g2 = G(t0 + d2 * h)
            g3 = G(t0 + d3 * h)
            a1 = h * g2
            a2 = c0 * h * (g3 - g1)
            a3 = c1 * h * (g3 - 2 * g2 + g1)
            comm1 = matrix_commutator(a1, a2)
            comm2 = matrix_commutator(2 * a3 + comm1, a1) / 60
            terms = a1 + (a3 / 12) + matrix_commutator(-20 * a1 - a3 + comm1, a2 + comm2) / 240
            return expm(terms)
    else:
        raise ValueError("Only magnus_order 1, 2, and 3 are supported.")

    return propagator


def step_solver_numpy(propagator_fn: Callable,
                      generator: Callable,
                      t_span: uarray,
                      t_eval: Optional[uarray],
                      y0: uarray,
                      max_dt: float,
                      time_grid_jitter: Optional[float] = None,
                      *,
                      rng=None) -> SolverResult:
    """Integrate the time evolution using NumPy-based matrix exponentials.

    Args:
        propagator_fn: Callable returning the evolution operator for a time
            step when supplied with a generator.
        generator: Function of time that produces the system generator matrix.
        t_span: Two-element array containing the start and end times.
        t_eval: Optional evaluation grid contained within ``t_span``.
        y0: Initial state or operator to propagate.
        max_dt: Maximum allowed time-step length.
        time_grid_jitter: Optional jitter amplitude forwarded to
            :func:`compute_time_grid`.

    Returns:
        Solver result storing the evaluation grid and propagated states.
    """
    t_list, h_nested_list, _ = compute_time_grid(t_span, t_eval, max_dt, jitter=time_grid_jitter, unp_inner=np, rng=rng)

    y = np.asarray(y0)
    ys = [y]

    for t0, h_list in zip(t_list[:-1], h_nested_list):
        for h in h_list:
            U = propagator_fn(generator, t0, h)
            y = U @ y
            t0 += h
        ys.append(y)

    ys = np.stack(ys, axis=0)
    return SolverResult(t_list, ys, time_axis=0, unp_inner=np)


def step_solver_jax(propagator_fn: Callable,
                    generator: Callable,
                    t_span: uarray,
                    t_eval: Optional[uarray],
                    y0: uarray,
                    max_dt: float,
                    time_grid_jitter: Optional[float] = None,
                    *,
                    rng=None) -> SolverResult:
    """Integrate the time evolution using JAX primitives and scans.

    Args:
        propagator_fn: Callable returning the evolution operator for a time
            step when supplied with a generator.
        generator: Function of time that produces the system generator matrix.
        t_span: Two-element array containing the start and end times.
        t_eval: Optional evaluation grid contained within ``t_span``.
        y0: Initial state or operator to propagate.
        max_dt: Maximum allowed time-step length.
        time_grid_jitter: Optional jitter amplitude forwarded to
            :func:`compute_time_grid`.

    Returns:
        Solver result storing the evaluation grid and propagated states.
    """
    t_list, h_nested_list, n_list = compute_time_grid(t_span, t_eval, max_dt, jitter=time_grid_jitter, unp_inner=jnp, rng=rng)
    y0 = jnp.asarray(y0)
    max_steps = int(jnp.max(n_list))

    def _pad_steps(h_arr):
        pad_width = max_steps - h_arr.shape[0]
        return jnp.pad(h_arr, (0, pad_width)) if pad_width else h_arr

    h_padded = jnp.stack([_pad_steps(h_arr) for h_arr in h_nested_list], axis=0)

    step_indices = jnp.arange(max_steps, dtype=jnp.int32)

    def scan_interval_integrate(carry, x):
        current_t, h_arr, n_steps = x
        current_y = carry

        def scan_take_step(inner_carry, inputs):
            (t, y) = inner_carry
            step_idx, h = inputs

            def true_branch(args):
                (t_inner, y_inner), h_inner = args
                U = propagator_fn(generator, t_inner, h_inner)
                y_next = U @ y_inner
                return t_inner + h_inner, y_next

            def false_branch(args):
                (t_inner, y_inner), _ = args
                return t_inner, y_inner

            next_carry = cond(step_idx < n_steps,
                              true_branch,
                              false_branch,
                              ((t, y), h))
            return next_carry, None

        (_, final_y), _ = scan(
            scan_take_step,
            (current_t, current_y),
            (step_indices, h_arr),
        )
        return final_y, final_y

    ys_without_y0 = scan(
        scan_interval_integrate,
        init=y0,
        xs=(jnp.asarray(t_list[:-1]), h_padded, n_list),
    )[1]
    ys = jnp.append(jnp.expand_dims(y0, axis=0), ys_without_y0, axis=0)
    return SolverResult(t_list, ys, time_axis=0, unp_inner=jnp)


def step_solver_parallel_jax(propagator_fn: Callable,
                             generator: Callable,
                             t_span: uarray,
                             t_eval: Optional[uarray],
                             y0: uarray,
                             max_dt: float,
                             time_grid_jitter: Optional[float] = None,
                             *,
                             rng=None) -> SolverResult:
    """Integrate the time evolution using an associative scan over JAX matrices.

    Args:
        propagator_fn: Callable returning the evolution operator for a time
            step when supplied with a generator.
        generator: Function of time that produces the system generator matrix.
        t_span: Two-element array containing the start and end times.
        t_eval: Optional evaluation grid contained within ``t_span``.
        y0: Initial state or operator to propagate.
        max_dt: Maximum allowed time-step length.
        time_grid_jitter: Optional jitter amplitude forwarded to
            :func:`compute_time_grid`.

    Returns:
        Solver result storing the evaluation grid and propagated states.
    """
    t_list, h_nested_list, n_list = compute_time_grid(t_span, t_eval, max_dt, jitter=time_grid_jitter, unp_inner=jnp, rng=rng)

    y0 = jnp.asarray(y0)

    # all_times = jnp.concatenate([t + h * jnp.arange(n) for t, h, n in zip(t_list[:-1], h_list, n_list)])
    # all_h = jnp.concatenate([h * jnp.ones(n) for h, n in zip(h_list, n_list)])
    all_times = jnp.concatenate([t + jnp.concatenate([jnp.array([0.0]), jnp.cumsum(h_list[:-1])]) for t, h_list in
                                 zip(t_list[:-1], h_nested_list)])
    all_h = jnp.concatenate(h_nested_list)
    t_list_locations = jnp.concatenate([jnp.array([0], n_list.dtype), jnp.cumsum(n_list)])

    all_Us = vmap(lambda t, h: propagator_fn(generator, t, h))(all_times, all_h)

    def reverse_mul(A, B):
        return jnp.matmul(B, A)

    if y0.ndim == 2 and y0.shape[0] == y0.shape[1]:
        all_Us_and_y0 = jnp.concatenate([jnp.expand_dims(y0, axis=0), all_Us], axis=0)
        cumulative_Us = associative_scan(reverse_mul, all_Us_and_y0, axis=0)
        ys = cumulative_Us[t_list_locations]
    else:
        cumulative_Us = associative_scan(reverse_mul, all_Us, axis=0)
        ys_without_y0 = cumulative_Us[t_list_locations[1:] - 1] @ y0
        ys = jnp.concatenate([jnp.expand_dims(y0, axis=0), ys_without_y0], axis=0)

    return SolverResult(t_list, ys, time_axis=0, unp_inner=jnp)


class Solver:
    """Solver for the time-dependent Schrödinger equation.

    The solver converts the supplied static Hamiltonian, driving operators, and
    optional noise operators into generator matrices, then combines them with
    time-dependent :class:`Signal` / :class:`SignalSum` objects during
    propagation. It does so by discretizing each time segment into short steps
    and approximating each step with a matrix exponential of the local
    generator. Use :meth:`solve_batch` to propagate one or more initial states
    over a single time segment, and
    :func:`solve_time_segment_sequence` to stitch together multiple segments.

    Args:
        static_hamiltonian: Static Hamiltonian of the segment.
        driving_operators: Optional driving operators multiplying the
            time-dependent drive signals.
        noise_operators: Optional operators multiplying shot-dependent
            quasistatic-noise coefficients.
    """

    def __init__(self,
                 static_hamiltonian: uarray,
                 driving_operators: Optional[Sequence[uarray]] = None,
                 noise_operators: Optional[Sequence[uarray]] = None):
        minus1j_two_pi = -1j * 2 * unp_dynamic.pi

        self.static_generator = minus1j_two_pi * unp_dynamic.asarray(static_hamiltonian)

        if driving_operators is None:
            self.drive_generators = None
        else:
            self.drive_generators = minus1j_two_pi * unp_dynamic.asarray(driving_operators)

        if noise_operators is None:
            self.noise_generators = None
        else:
            self.noise_generators = minus1j_two_pi * unp_dynamic.asarray(noise_operators)

    def make_generator_without_noise(self,
                                     unp_inner,
                                     drive_signals: Optional[Sequence[Union[Signal, SignalSum]]]) -> Callable:
        """Build a generator factory that optionally injects noise coefficients.

        Args:
            unp_inner: Array namespace (:mod:`numpy` or :mod:`jax.numpy`) that
                matches the solver backend.
            drive_signals: Sequence of drive signals sampled during the solve or
                ``None`` if no drives are applied.

        Returns:
            Callable accepting shot-specific noise coefficients and returning a
            function ``generator(t)`` that yields the combined static, drive,
            and noise contributions.
        """

        static_generator = unp_inner.asarray(self.static_generator)
        drive_generators = unp_inner.asarray(self.drive_generators) if self.drive_generators is not None else None
        noise_generators = unp_inner.asarray(self.noise_generators) if self.noise_generators is not None else None
        drive_signals_tuple = tuple(drive_signals) if drive_signals is not None else None

        def add_noise(generator, noise_coeffs):
            if noise_generators is None or noise_coeffs is None:
                return generator

            coeffs = unp_inner.asarray(noise_coeffs)
            return generator + unp_inner.tensordot(coeffs, noise_generators, axes=(0, 0))

        if drive_generators is None or drive_signals_tuple is None:
            def generator_with_noise(noise_coeffs):
                generator = add_noise(static_generator, noise_coeffs)

                def generator_no_drive(t):
                    return generator

                return generator_no_drive

            return generator_with_noise

        def generator_with_noise(noise_coeffs):
            generator = add_noise(static_generator, noise_coeffs)

            def generator_with_drive(t):
                coeffs = unp_inner.stack([sig(t) for sig in drive_signals_tuple])
                coeffs = coeffs.astype(drive_generators.dtype, copy=False)
                return generator + unp_inner.tensordot(coeffs, drive_generators, axes=(0, 0))

            return generator_with_drive

        return generator_with_noise

    def solve_batch(self,
                    y0: Union[uarray, List[uarray]],
                    t_span: uarray,
                    t_eval: Optional[uarray] = None,
                    drive_signals: Optional[Sequence[Union[Signal, SignalSum]]] = None,
                    noise_coeffs: Optional[Sequence[Optional[Sequence[float]]]] = None,
                    *,
                    method: str,
                    max_dt: float,
                    time_grid_jitter: Optional[float] = None,
                    batch_size: Optional[int] = None,
                    rng=None) -> SolverResult:
        """Propagate one or more initial conditions through a single time span.

        Args:
            y0: Initial state or list of states/operators to evolve.
            t_span: Two-element array containing the start and end times.
            t_eval: Optional evaluation grid contained within ``t_span``.
            drive_signals: Sequence of drive signals evaluated at runtime.
            noise_coeffs: Optional per-shot noise coefficients matching
                ``self.noise_generators``.
            method: Backend identifier (``\"numpy_expm\"``, ``\"jax_expm\"``, or
                ``\"jax_expm_parallel\"``).
            max_dt: Maximum allowed time-step length.
            time_grid_jitter: Optional jitter amplitude forwarded to
                :func:`compute_time_grid`.
            batch_size: Optional batch size used when vectorising JAX solves.
            rng: Optional simulation-local NumPy random generator used for
                time-grid jitter.

        Returns:
            Solver result containing the propagated states for all shots.

        Raises:
            ValueError: If ``method`` is unsupported, or if ``batch_size`` is
                invalid for a JAX backend.
        """
        if isinstance(y0, (list, tuple)):
            y0s = list(y0)
        else:
            y0s = [y0]

        if noise_coeffs is None:
            noise_coeffs = [None] * len(y0s)

        if method == "numpy_expm":
            unp_inner = np
            solver_fn = step_solver_numpy
        elif method == "jax_expm":
            unp_inner = jnp
            solver_fn = step_solver_jax
        elif method == "jax_expm_parallel":
            unp_inner = jnp
            solver_fn = step_solver_parallel_jax
        else:
            raise ValueError(f"Unsupported method: {method}")

        propagator_fn = get_propagator(magnus_order=1)

        def solve_shot(generator_fn, y0):
            return solver_fn(
                propagator_fn,
                generator_fn,
                t_span=t_span,
                t_eval=t_eval,
                y0=y0,
                max_dt=max_dt,
                time_grid_jitter=time_grid_jitter,
                rng=rng,
            )

        generator_without_noise = self.make_generator_without_noise(unp_inner, drive_signals)

        if unp_inner is jnp and y0s:
            chunk_size = len(y0s) if batch_size is None else batch_size
            if chunk_size is None or chunk_size <= 0:
                raise ValueError("batch_size must be a positive integer or None when using JAX backends.")

            total_shots = len(y0s)
            chunk_results = []

            def solve_chunk(start_idx: int) -> SolverResult:
                end_idx = min(start_idx + chunk_size, total_shots)
                y0_chunk = y0s[start_idx:end_idx]
                noise_chunk = noise_coeffs[start_idx:end_idx]

                y0_array = jnp.stack([jnp.asarray(y) for y in y0_chunk], axis=0)

                noise_array = None
                if self.noise_generators is not None:
                    noise_generators_arr = jnp.asarray(self.noise_generators)
                    noise_length = noise_generators_arr.shape[0]
                    if noise_length > 0:
                        default_noise = jnp.zeros(noise_length, dtype=noise_generators_arr.dtype)
                        noise_batch = []
                        for coeff in noise_chunk:
                            if coeff is None:
                                noise_batch.append(default_noise)
                            else:
                                noise_batch.append(jnp.asarray(coeff, dtype=noise_generators_arr.dtype))
                        noise_array = jnp.stack(noise_batch, axis=0)

                base_generator = generator_without_noise(None) if noise_array is None else None

                def run_single_no_noise(y0_single):
                    result = solver_fn(
                        propagator_fn,
                        base_generator,
                        t_span=t_span,
                        t_eval=t_eval,
                        y0=y0_single,
                        max_dt=max_dt,
                        time_grid_jitter=time_grid_jitter,
                        rng=rng,
                    )
                    return result.ys, result.ts

                def run_single_with_noise(noise_single, y0_single):
                    result = solver_fn(
                        propagator_fn,
                        generator_without_noise(noise_single),
                        t_span=t_span,
                        t_eval=t_eval,
                        y0=y0_single,
                        max_dt=max_dt,
                        time_grid_jitter=time_grid_jitter,
                        rng=rng,
                    )
                    return result.ys, result.ts

                if noise_array is None:
                    ys_chunk, ts_chunk = vmap(run_single_no_noise, in_axes=0, out_axes=(0, 0))(y0_array)
                else:
                    ys_chunk, ts_chunk = vmap(run_single_with_noise, in_axes=(0, 0), out_axes=(0, 0))(noise_array,
                                                                                                      y0_array)

                return SolverResult(ts_chunk[0], ys_chunk, time_axis=1, shot_axis=0, unp_inner=jnp)

            for offset in range(0, total_shots, chunk_size):
                chunk_results.append(solve_chunk(offset))

            if len(chunk_results) == 1:
                return chunk_results[0]
            return SolverResult.stack_shots(chunk_results)

        results = []
        for y0_i, noise_coeffs_i in zip(y0s, noise_coeffs):
            generator = generator_without_noise(noise_coeffs_i)
            results.append(solve_shot(generator, y0_i))

        return SolverResult.stack_shots(results)


def solve_time_segment_sequence(solver: Solver,
                                y0s,
                                noise_coeffs,
                                time_segment_sequence,
                                method: str,
                                batch_size: Optional[int],
                                verbose: bool,
                                rng=None) -> SolverResult:
    """Solve a sequence of time segments and concatenate the results.

    Args:
        solver: Solver instance used for all segments.
        y0s: Initial states or operators, one per simulated shot.
        noise_coeffs: Shot-dependent noise strengths aligned with ``y0s``.
        time_segment_sequence: Prepared time segments to be simulated in order.
        method: Low-level integration method passed through to
            :meth:`Solver.solve_batch`.
        batch_size: Optional shot-batch size for batched JAX simulation.
        verbose: If ``True``, print a short summary for each simulated segment.
        rng: Optional simulation-local random number generator used for
            jittered time-grid construction.

    Returns:
        Solver result aggregating the trajectories across all segments.

    Raises:
        SimphonyError: If a segment declares an unknown simulation method.
    """
    segment_results = []

    for idx, time_segment in enumerate(time_segment_sequence):
        if time_segment.simulation.method == 'basic':
            segment_solver_fn = solve_time_segment_basic
        elif time_segment.simulation.method == 'single_sine_wave':
            segment_solver_fn = solve_time_segment_single_sine_wave
        else:
            raise SimphonyError('Unknown simulation_type')

        if verbose:
            duration = time_segment.duration
            prec = max(-int(np.floor(np.log10(duration))) + 2, 0)
            start, end = time_segment.simulation.t_span
            max_dt = time_segment.simulation.max_dt
            seg_type = time_segment.simulation.method
            print(
                f'simulate time segment [{start:.{prec}f}, {end:.{prec}f}] '
                f'with step size {max_dt:.4g} (type: {seg_type})'
            )

        segment_result = segment_solver_fn(
            solver=solver,
            time_segment=time_segment,
            noise_coeffs=noise_coeffs,
            method=method,
            y0s=y0s,
            batch_size=batch_size,
            rng=rng,
        )
        y0s = segment_result.final_shot()
        segment_results.append(segment_result)

    return SolverResult.concat_segments(segment_results)


def solve_time_segment_basic(solver: Solver,
                             y0s: List[uarray],
                             noise_coeffs: List[Sequence[float]],
                             time_segment,
                             method: str,
                             batch_size: Optional[int],
                             rng=None) -> SolverResult:
    """Solve a single time segment with uniform discretization.

    Builds per-shot signal lists, integrates with the configured method, and
    returns a :class:`SolverResult` containing the stacked solution array.

    Args:
        solver: Solver instance used to propagate the segment.
        y0s: Initial states or operators, one per simulated shot.
        noise_coeffs: Shot-dependent noise strengths aligned with ``y0s``.
        time_segment: Prepared time segment containing the simulation grid and
            drive signals.
        method: Low-level integration method passed through to
            :meth:`Solver.solve_batch`.
        batch_size: Optional shot-batch size for batched simulation.
        rng: Optional simulation-local random number generator used for
            jittered time-grid construction.

    Returns:
        Solver result on ``time_segment.simulation.t_eval`` with the propagated
        states or operators stacked over shots.
    """
    return solver.solve_batch(y0=y0s,
                              t_span=time_segment.simulation.t_span,
                              t_eval=time_segment.simulation.t_eval,
                              drive_signals=time_segment.simulation.signals,
                              noise_coeffs=noise_coeffs,
                              method=method,
                              max_dt=time_segment.simulation.max_dt,
                              time_grid_jitter=time_segment.simulation.time_grid_jitter,
                              batch_size=batch_size,
                              rng=rng)


def solve_time_segment_single_sine_wave(solver: Solver,
                                        y0s: List[uarray],
                                        noise_coeffs: List[Sequence[float]],
                                        time_segment,
                                        method: str,
                                        batch_size: Optional[int],
                                        rng=None) -> SolverResult:
    r"""Solve a constant-envelope single-sine segment.

    The method simulates one sine period on identity initial conditions, then
    reconstructs the full segment from powers of the single-cycle propagator.

    .. note::

        This method applies to segments with a single nonzero driving frequency
        and a constant envelope. Instead of simulating the full segment
        directly, it simulates only one sine period and uses that result to
        reconstruct the full propagator.

        If :math:`T = 1 / f` is the period of the drive, then the full
        evolution is reconstructed from the single-cycle propagator according
        to

        .. math::

            U(nT + \tau) = U(\tau) [U(T)]^n, \qquad 0 \leq \tau < T.

        In practice, Simphony projects the requested evaluation times into a
        single period, solves the dynamics there once on identity initial
        conditions, and then reconstructs the full-segment evolution by taking
        matrix powers of the single-cycle propagator. This can reduce both
        memory usage and runtime substantially for long constant-envelope
        pulses.

    Args:
        solver: Solver instance used to propagate the segment.
        y0s: Initial states or operators, one per simulated shot.
        noise_coeffs: Shot-dependent noise strengths aligned with ``y0s``.
        time_segment: Prepared time segment containing the projected single-cycle
            simulation data.
        method: Low-level integration method passed through to
            :meth:`Solver.solve_batch`.
        batch_size: Optional shot-batch size for batched simulation.
        rng: Optional simulation-local random number generator used for
            jittered time-grid construction.

    Returns:
        Solver result on ``time_segment.simulation.t_eval`` reconstructed from
        the simulated single-cycle propagator.
    """
    unp_inner = np if method == "numpy_expm" else jnp
    y0s_array = unp_inner.asarray(y0s)
    identity = unp_inner.identity(y0s_array.shape[-1], dtype=y0s_array.dtype)
    y0s_identity = [identity for _ in range(y0s_array.shape[0])]

    sol = solver.solve_batch(y0=y0s_identity,
                             t_span=time_segment.simulation.t_span_projected_sorted,
                             t_eval=time_segment.simulation.t_eval_projected_sorted,
                             drive_signals=time_segment.simulation.signals,
                             noise_coeffs=noise_coeffs,
                             method=method,
                             max_dt=time_segment.simulation.max_dt,
                             time_grid_jitter=time_segment.simulation.time_grid_jitter,
                             batch_size=batch_size,
                             rng=rng)

    solution_projected_sorted = sol.ys

    unsort_indices = unp_inner.asarray(time_segment.simulation.unsort_indices)
    solution_projected = solution_projected_sorted[:, unsort_indices]
    U_single_sine = solution_projected_sorted[:, -1]

    cycle_indices = unp_inner.asarray(time_segment.simulation.cycle_indices)
    U_single_sine_powers = unp_inner.stack(
        [unp_inner.linalg.matrix_power(U_single_sine, int(i)) for i in cycle_indices],
        axis=1,
    )

    solution = unp_inner.einsum('stij,stjk,skl->stil', solution_projected, U_single_sine_powers, y0s_array)
    solver_result = SolverResult(time_segment.simulation.t_eval,
                                 solution,
                                 time_axis=1,
                                 shot_axis=0,
                                 unp_inner=unp_inner)
    return solver_result
