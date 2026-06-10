import numpy as np
import simphony


def test_release_smoke() -> None:
    db = simphony.default_nv_hyperfine_database()
    assert len(db) > 0
    assert db[0].type in {'N', 'V', 'C'}

    model = simphony.default_nv_model()
    assert model.n_spins > 0
    assert 'MW_x' in model.driving_field_names

    duration = 0.01
    frequency = model.splitting_qubit('e')
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
        phase=0.0,
        duration=duration,
    )

    result = model.simulate_time_evolution(start=0.0, end=duration, n_eval=2, simulation_method='basic')
    assert result.n_ts >= 1
    assert result.time_evol_operator is not None
