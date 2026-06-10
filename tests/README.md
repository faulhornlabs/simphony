# Tests

This directory contains both small unit tests and larger numerical regression
checks.

## Test Types

- `test_*.py`: pytest-discovered tests
- `esr.py`, `nmr.py`, `ddrf.py`, `autodiff.py`: numerical regression scripts
  that can be run directly or through pytest wrappers

## Default CPU Regression Run

Run the default CPU regression suite with pytest:

```bash
pytest -q tests/test_regressions.py -m cpu
```

This uses `assert` mode, so each regression case runs once and checks only
correctness against the stored references.

## Release Smoke Test

After installing Simphony into a fresh environment, run the release smoke test
to verify the packaged import path, the packaged NV database, and a minimal
time-evolution call:

```bash
pytest -q tests/test_release_smoke.py
```

This test is intentionally small and should run quickly.

## Performance Runner

Run the full shell-based regression matrix with:

```bash
SIMPHONY_TEST_MODE=performance ./tests/run_regression_benchmark.sh
```

This runner is intended for performance-oriented regression checks. If
`SIMPHONY_TEST_MODE` is not set, it defaults to `performance`.

## GPU Tests

The optional GPU pytest wrapper is:

```bash
SIMPHONY_RUN_GPU_TESTS=1 pytest -q tests/test_regressions.py
```

If no GPU backend is available, GPU regression cases are skipped cleanly.

## DDRF and Quasar

The DDRF regression depends on the optional `quasar` package. If `quasar` is
not importable, the DDRF regression is skipped instead of failing the full test
run.

## Reference Files

Regression baselines are stored under:

```text
tests/refs/
```

These `.npy` files are part of the regression suite and are required for the
numerical checks.

## Logs

`tests/run_regression_benchmark.sh` writes per-test logs and a run summary under:

```text
tests/logs/<run-id>/
```

Each script also emits a machine-readable `RESULT ...` line that the shell
runner uses to build the final summary.
