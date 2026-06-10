import os

import pytest

from regressions import (
    RegressionSkip,
    run_autodiff,
    run_ddrf,
    run_esr,
    run_nmr,
)


REGRESSION_CASES = [
    pytest.param("ESR", run_esr, "cpu", marks=pytest.mark.cpu, id="esr-cpu"),
    pytest.param("ESR", run_esr, "gpu", marks=pytest.mark.gpu, id="esr-gpu"),
    pytest.param("NMR", run_nmr, "cpu", marks=pytest.mark.cpu, id="nmr-cpu"),
    pytest.param("NMR", run_nmr, "gpu", marks=pytest.mark.gpu, id="nmr-gpu"),
    pytest.param("DDRF", run_ddrf, "cpu", marks=pytest.mark.cpu, id="ddrf-cpu"),
    pytest.param("DDRF", run_ddrf, "gpu", marks=pytest.mark.gpu, id="ddrf-gpu"),
    pytest.param("AUTODIFF", run_autodiff, "cpu", marks=pytest.mark.cpu, id="autodiff-cpu"),
    pytest.param("AUTODIFF", run_autodiff, "gpu", marks=pytest.mark.gpu, id="autodiff-gpu"),
]


@pytest.mark.parametrize(("name", "function", "platform"), REGRESSION_CASES)
def test_regression_case(name: str, function, platform: str) -> None:
    if platform == "gpu" and os.environ.get("SIMPHONY_RUN_GPU_TESTS") != "1":
        pytest.skip("Set SIMPHONY_RUN_GPU_TESTS=1 to run optional GPU regression tests.")

    try:
        function(platform, mode="assert")
    except RegressionSkip as exc:
        pytest.skip(str(exc))
