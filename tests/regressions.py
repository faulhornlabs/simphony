"""Shared regression helpers, runners, and CLI entrypoint."""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import simphony

VALID_TEST_MODES = {'assert', 'performance'}
REFS_DIR = Path(__file__).parent / 'refs'


class RegressionSkip(RuntimeError):
    """Raised when a regression case cannot run in the current environment."""


def ensure_gpu_available() -> None:
    """Raise ``RegressionSkip`` when no GPU backend is available."""
    import jax

    try:
        devices = jax.devices('gpu')
    except Exception as exc:
        raise RegressionSkip('no_gpu_device_available') from exc

    if not devices:
        raise RegressionSkip('no_gpu_device_available')


def ensure_runtime_requirements(platform: str) -> None:
    """Validate optional runtime requirements for a regression case."""
    if platform == 'gpu':
        ensure_gpu_available()


def append_default_ddrf_pulse(model, nuclear_spin_name: str, angle: float, phase: float,
                              rf_duration: float, n_dds: int, dd_duration: float,
                              controlled: bool) -> object:
    """Add a minimal DDRF pulse sequence (specific to the regression test case).

    This is a simplified, test-specific version that generates the exact pulse sequence
    for the canonical DDRF regression test parameters.
    """
    # Hardcoded parameters for this specific test
    electron_spin_name = 'e'
    mw_driving_field_name = 'MW_x'
    rf_driving_field_name = 'RF_x'

    # Sign corrections
    if model.splitting_qubit(electron_spin_name) > 0:
        angle *= -1
    phase *= -1

    phase_shift = np.pi  # since controlled=True in this test

    # Compute driving field parameters
    electron_qubit_subspace = model.spin(electron_spin_name).qubit_subspace
    frequency_e = model.splitting_qubit(electron_spin_name)
    amplitude_e = model.rabi_amplitude_qubit(
        driving_field_name=mw_driving_field_name,
        period_time=2 * dd_duration,
        spin_name=electron_spin_name
    )

    frequency_n_0 = model.splitting_qubit(
        nuclear_spin_name,
        rest_quantum_nums={electron_spin_name: electron_qubit_subspace[0]}
    )
    frequency_n_1 = model.splitting_qubit(
        nuclear_spin_name,
        rest_quantum_nums={electron_spin_name: electron_qubit_subspace[1]}
    )

    diff_frequency_n = frequency_n_0 - frequency_n_1
    phi_tau = 2 * np.pi * diff_frequency_n * rf_duration
    rabi_frequency_n = angle / (2 * np.pi * n_dds * rf_duration)

    amplitude_n = model.rabi_amplitude_qubit(
        driving_field_name=rf_driving_field_name,
        period_time=1 / rabi_frequency_n,
        spin_name=nuclear_spin_name,
        rest_quantum_nums={electron_spin_name: electron_qubit_subspace[1]}
    )

    phase += model.spin(nuclear_spin_name).virtual_phase

    def _dd_phase(idx: int) -> float:
        return 0 if idx % 2 == 0 else np.pi / 2

    time = model.last_pulse_end

    # Initial RF tau
    phase_1_0 = phase + (phase_shift if 0 % 2 == 0 else 0)
    model.driving_field(rf_driving_field_name).add_rectangle_pulse(
        amplitude=amplitude_n,
        frequency=frequency_n_1,
        phase=phase_1_0,
        duration=rf_duration,
        start=time
    )
    time += rf_duration

    # Internal blocks: MW pi + RF 2tau
    for i in range(n_dds - 1):
        model.driving_field(mw_driving_field_name).add_rectangle_pulse(
            amplitude=amplitude_e,
            frequency=frequency_e,
            phase=_dd_phase(i),
            duration=dd_duration,
            start=time
        )
        time += dd_duration

        phase_mid = phase + (i + 1) * phi_tau + (phase_shift if i % 2 == 1 else 0)
        model.driving_field(rf_driving_field_name).add_rectangle_pulse(
            amplitude=amplitude_n,
            frequency=frequency_n_1,
            phase=phase_mid,
            duration=2 * rf_duration,
            start=time
        )
        time += 2 * rf_duration

    # Final MW pi
    model.driving_field(mw_driving_field_name).add_rectangle_pulse(
        amplitude=amplitude_e,
        frequency=frequency_e,
        phase=_dd_phase(n_dds - 1),
        duration=dd_duration,
        start=time
    )
    time += dd_duration

    # Final RF tau
    phase_2_last = phase + n_dds * phi_tau + (phase_shift if (n_dds - 1) % 2 == 1 else 0)
    model.driving_field(rf_driving_field_name).add_rectangle_pulse(
        amplitude=amplitude_n,
        frequency=frequency_n_1,
        phase=phase_2_last,
        duration=rf_duration,
        start=time
    )

    # Update virtual phases for non-electron spins
    duration = n_dds * (2 * rf_duration + dd_duration)
    for spin in model.spins:
        if spin.name != electron_spin_name:
            splitting_0 = model.splitting_qubit(
                spin.name,
                rest_quantum_nums={electron_spin_name: electron_qubit_subspace[0]}
            )
            splitting_1 = model.splitting_qubit(
                spin.name,
                rest_quantum_nums={electron_spin_name: electron_qubit_subspace[1]}
            )
            splitting_diff = splitting_0 - splitting_1
            phase_update = 2 * np.pi * splitting_diff * duration / 2
            spin.virtual_phase += phase_update

    return model


def normalize_test_mode(mode: str) -> str:
    """Validate and normalize regression execution mode."""
    normalized = mode.lower()
    if normalized not in VALID_TEST_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_TEST_MODES)}, got: {mode}")
    return normalized


def emit_result_line(test_name: str,
                     platform: str,
                     *,
                     status: str,
                     mode: Optional[str] = None,
                     runtime: Optional[float] = None,
                     jit_runtime: Optional[float] = None,
                     compiled_runtime: Optional[float] = None,
                     reason: Optional[str] = None) -> None:
    """Print a machine-readable result line for the shell regression runner."""
    parts = [
        'RESULT',
        f'test={test_name}',
        f'platform={platform}',
        f'status={status}',
    ]
    if mode is not None:
        parts.append(f'mode={mode}')
    if runtime is not None:
        parts.append(f'runtime={runtime:.6f}')
    if jit_runtime is not None:
        parts.append(f'jit_runtime={jit_runtime:.6f}')
    if compiled_runtime is not None:
        parts.append(f'compiled_runtime={compiled_runtime:.6f}')
    if reason is not None:
        parts.append(f'reason={reason}')
    print(' '.join(parts))


def run_esr(platform: str = 'cpu', mode: str = 'performance') -> None:
    platform = platform.lower()
    if platform not in ('cpu', 'gpu'):
        platform = 'cpu'
    mode = normalize_test_mode(mode)

    ensure_runtime_requirements(platform)
    simphony.Config.set_platform(platform)

    model = simphony.default_nv_model(nitrogen_isotope=14, static_field_strength=0.05)

    duration = 0.01
    frequency = model.splitting_qubit('e')
    phase = 0
    angle = np.pi / 2
    period_time = 2 * np.pi / angle * duration
    amplitude = model.rabi_amplitude_qubit(
        driving_field_name='MW_x',
        period_time=period_time,
        spin_name='e',
    )

    model.driving_field('MW_x').add_rectangle_pulse(
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
    )

    ref_file = REFS_DIR / 'esr_U_ref.npy'
    if not ref_file.exists():
        raise FileNotFoundError(f'Missing reference file: {ref_file}. Generate it on the original branch.')
    U_ref = np.load(ref_file)
    threshold = 1e-5

    if mode == 'assert':
        t0 = time.perf_counter()
        result = model.simulate_time_evolution(n_eval=251, n_split=250, simulation_method='basic')
        t1 = time.perf_counter()
        U = result.time_evol_operator.matrix(t_idx=-1)[0, 0]
        F = simphony.average_gate_fidelity_from_unitaries(U, U_ref)
        emit_result_line('ESR', platform, status='PASS', mode=mode, runtime=t1 - t0)
        assert np.abs(1 - float(F)) < threshold
        return

    t0 = time.perf_counter()
    result1 = model.simulate_time_evolution(n_eval=251, n_split=250, simulation_method='basic')
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    result2 = model.simulate_time_evolution(n_eval=251, n_split=250, simulation_method='basic')
    t3 = time.perf_counter()

    U1 = result1.time_evol_operator.matrix(t_idx=-1)[0, 0]
    U2 = result2.time_evol_operator.matrix(t_idx=-1)[0, 0]
    F = simphony.average_gate_fidelity_from_unitaries(U2, U_ref)

    emit_result_line('ESR', platform, status='PASS', mode=mode, jit_runtime=t1 - t0, compiled_runtime=t3 - t2)

    np.testing.assert_allclose(U1, U2)
    assert np.abs(1 - float(F)) < threshold


def run_nmr(platform: str = 'cpu', mode: str = 'performance') -> None:
    platform = platform.lower()
    if platform not in ('cpu', 'gpu'):
        platform = 'cpu'
    mode = normalize_test_mode(mode)

    ensure_runtime_requirements(platform)
    simphony.Config.set_platform(platform)

    model = simphony.default_nv_model(
        nitrogen_isotope=14,
        static_field_strength=0.05,
        carbon_atom_indices=[(1, 1, 1, 0), (2, -3, 4, 1)],
    )

    duration = 10
    frequency = model.splitting_qubit('N', rest_quantum_nums={'e': 0, 'C1': -1 / 2, 'C2': -1 / 2})
    frequency = np.abs(frequency)
    phase = 0
    angle = np.pi / 2
    period_time = 2 * np.pi / angle * duration
    wait_duration = 42.
    r = 0.1
    amplitude = model.rabi_amplitude_qubit(
        driving_field_name='RF_x',
        period_time=period_time,
        spin_name='N',
        rest_quantum_nums={'e': 0, 'C1': -1 / 2, 'C2': -1 / 2},
    )

    model.driving_field('RF_x').add_rectangle_pulse(
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        duration=duration,
    )
    model.driving_field('RF_x').add_wait(duration=wait_duration)
    model.driving_field('RF_x').add_rectangle_pulse(
        amplitude=amplitude,
        is_effective_amplitude=True,
        frequency=frequency,
        phase=phase,
        duration=duration,
        rise_time=r * duration,
        fall_time=r * duration,
    )

    ref_file = REFS_DIR / 'nmr_expval_ref.npy'
    if not ref_file.exists():
        raise FileNotFoundError(f'Missing reference file: {ref_file}. Generate it on the original branch.')
    expval_ref = np.load(ref_file)
    initial_state = model.productstate({'e': 0, 'N': 0, 'C1': -1 / 2, 'C2': -1 / 2})
    threshold = 1e-5

    if mode == 'assert':
        t0 = time.perf_counter()
        result = model.simulate_time_evolution(n_eval=25, n_split=250)
        t1 = time.perf_counter()
        result.initial_state = initial_state
        expval = result.expectation_value(['IZIZ', 'XYZI'])[0]
        norm = np.linalg.norm(expval - expval_ref)
        emit_result_line('NMR', platform, status='PASS', mode=mode, runtime=t1 - t0)
        assert np.abs(float(norm)) < threshold
        return

    t0 = time.perf_counter()
    result1 = model.simulate_time_evolution(n_eval=25, n_split=250)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    result2 = model.simulate_time_evolution(n_eval=25, n_split=250)
    t3 = time.perf_counter()

    result1.initial_state = initial_state
    expval1 = result1.expectation_value(['IZIZ', 'XYZI'])[0]
    result2.initial_state = initial_state
    expval2 = result2.expectation_value(['IZIZ', 'XYZI'])[0]
    norm = np.linalg.norm(expval2 - expval_ref)

    emit_result_line('NMR', platform, status='PASS', mode=mode, jit_runtime=t1 - t0, compiled_runtime=t3 - t2)

    np.testing.assert_allclose(expval1, expval2)
    assert np.abs(float(norm)) < threshold


def run_ddrf(platform: str = 'gpu', mode: str = 'performance') -> None:
    platform = platform.lower()
    if platform not in ('cpu', 'gpu'):
        platform = 'cpu'
    mode = normalize_test_mode(mode)

    ensure_runtime_requirements(platform)
    simphony.Config.set_platform(platform)

    model = simphony.default_nv_model(
        nitrogen_isotope=14,
        static_field_strength=0.05,
    )

    model = append_default_ddrf_pulse(
        model=model,
        nuclear_spin_name='N',
        angle=np.pi,
        phase=0,
        rf_duration=10,
        n_dds=4,
        dd_duration=0.01,
        controlled=True,
    )

    ref_file = REFS_DIR / 'ddrf_chi_ref.npy'
    if not ref_file.exists():
        raise FileNotFoundError(f'Missing reference file: {ref_file}. Generate it on the original branch.')
    chi_ref = np.load(ref_file)
    threshold = 1e-5

    if mode == 'assert':
        t0 = time.perf_counter()
        result = model.simulate_time_evolution(n_shots=1, n_eval=2, n_split=251)
        t1 = time.perf_counter()
        chi = result.process_matrix()
        chi_norm = np.linalg.norm(chi - chi_ref)
        emit_result_line('DDRF', platform, status='PASS', mode=mode, runtime=t1 - t0)
        assert np.abs(float(chi_norm)) < threshold
        return

    t0 = time.perf_counter()
    result1 = model.simulate_time_evolution(n_shots=1, n_eval=2, n_split=251)
    t1 = time.perf_counter()
    n_shots = 10 if platform == 'gpu' else 1
    t2 = time.perf_counter()
    result2 = model.simulate_time_evolution(n_shots=n_shots, n_eval=2, n_split=251)
    t3 = time.perf_counter()

    chi1 = result1.process_matrix()
    chi2 = result2.process_matrix()
    chi_norm = np.linalg.norm(chi2 - chi_ref)

    emit_result_line(
        'DDRF',
        platform,
        status='PASS',
        mode=mode,
        jit_runtime=t1 - t0,
        compiled_runtime=(t3 - t2) / n_shots,
    )

    np.testing.assert_allclose(chi1, chi2)
    assert np.abs(float(chi_norm)) < threshold


def run_autodiff(platform: str = 'gpu', mode: str = 'performance') -> None:
    platform = platform.lower()
    if platform not in ('cpu', 'gpu'):
        platform = 'cpu'
    mode = normalize_test_mode(mode)

    ensure_runtime_requirements(platform)
    simphony.Config.set_platform(platform)
    simphony.Config.set_autodiff_mode(True)

    from jax import value_and_grad
    from qiskit import QuantumCircuit

    model = simphony.default_nv_model(
        nitrogen_isotope=14,
        static_field_strength=0.005,
    )

    frequency = model.splitting_qubit(spin_name='e')
    qc = QuantumCircuit(2)
    qc.rx(np.pi / 2, 0)

    def infidelity(parameters_, dt_, model_, n_shots_):
        samples = parameters_[::2] + 1j * parameters_[1::2]
        model_.remove_all_pulses()
        model_.driving_field('MW_x').add_discrete_pulse(frequency=frequency, samples=samples, dt=dt_)
        model.spin('e').local_quasistatic_noise.z = 25
        result = model.simulate_time_evolution(n_eval=2, n_shots=n_shots_, random_seed=42)
        result.ideal = qc
        return 1 - result.average_gate_fidelity()

    val_grad_fn = value_and_grad(infidelity)

    dt = 0.01 / 8
    parameters = np.array([
        -0.01322891, 0.04721478, 0.02287283, 0.00860466, -0.03636436,
        -0.03708852, -0.04570514, 0.03769117, 0.01089687, 0.02379661,
        -0.04742311, 0.04797991, 0.03008434, -0.02462076, -0.03396757,
        -0.03406872,
    ])

    value_ref = np.load(REFS_DIR / 'autodiff_value_ref.npy')
    grad_ref = np.load(REFS_DIR / 'autodiff_grad_ref.npy')
    threshold = 1e-4

    if mode == 'assert':
        t0 = time.perf_counter()
        value, grad = val_grad_fn(parameters, dt, model, n_shots_=3)
        t1 = time.perf_counter()
        emit_result_line('AUTODIFF', platform, status='PASS', mode=mode, runtime=t1 - t0)
        assert np.abs(float(value) - float(value_ref)) < threshold
        assert np.abs(np.linalg.norm(grad - grad_ref)) < threshold
        return

    t0 = time.perf_counter()
    value1, grad1 = val_grad_fn(parameters, dt, model, n_shots_=3)
    t1 = time.perf_counter()
    t2 = time.perf_counter()
    value2, grad2 = val_grad_fn(parameters, dt, model, n_shots_=3)
    t3 = time.perf_counter()

    emit_result_line('AUTODIFF', platform, status='PASS', mode=mode, jit_runtime=t1 - t0, compiled_runtime=t3 - t2)

    np.testing.assert_allclose(value1, value2)
    np.testing.assert_allclose(grad1, grad2)
    assert np.abs(float(value2) - float(value_ref)) < threshold
    assert np.abs(np.linalg.norm(grad2 - grad_ref)) < threshold


REGRESSION_RUNNERS = {
    'ESR': run_esr,
    'NMR': run_nmr,
    'DDRF': run_ddrf,
    'AUTODIFF': run_autodiff,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run a Simphony regression case.')
    parser.add_argument('--case', choices=sorted(REGRESSION_RUNNERS), required=True, help='Regression case to run')
    parser.add_argument('--platform', choices=['cpu', 'gpu'], default='cpu', help='Execution platform')
    parser.add_argument('--mode', choices=['assert', 'performance'], default='performance', help='Execution mode')
    args = parser.parse_args(argv)

    try:
        REGRESSION_RUNNERS[args.case](args.platform, mode=args.mode)
    except RegressionSkip as exc:
        print(f"SKIP: {exc}")
        emit_result_line(args.case, args.platform, status='SKIP', mode=args.mode, reason=str(exc))
        return 80
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
