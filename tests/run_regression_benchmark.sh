#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CPU_COUNT="${SIMPHONY_TEST_CPU_COUNT:-1}"
TEST_MODE="${SIMPHONY_TEST_MODE:-performance}"
if ! [[ "$CPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SIMPHONY_TEST_CPU_COUNT must be a positive integer, got: $CPU_COUNT" >&2
    exit 2
fi
if [[ "$TEST_MODE" != "assert" && "$TEST_MODE" != "performance" ]]; then
    echo "SIMPHONY_TEST_MODE must be 'assert' or 'performance', got: $TEST_MODE" >&2
    exit 2
fi

if [[ "$TEST_MODE" == "assert" ]]; then
    echo "Use pytest for correctness checks:"
    echo "  pytest -q tests/test_regressions.py -m cpu"
    echo "Optional GPU coverage:"
    echo "  SIMPHONY_RUN_GPU_TESTS=1 pytest -q tests/test_regressions.py"
    exit 0
fi

# Keep CPU-backed numerical libraries from using every available core.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_COUNT}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_COUNT}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_COUNT}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_COUNT}"

XLA_CPU_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$CPU_COUNT"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }$XLA_CPU_FLAGS"

# Determine active git branch and set up logging
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
SAFE_BRANCH_NAME=${BRANCH_NAME//\//_}
SAFE_BRANCH_NAME=${SAFE_BRANCH_NAME//:/_}
SAFE_BRANCH_NAME=${SAFE_BRANCH_NAME// /_}
RUN_ID="$(date +%Y%m%d_%H%M%S)_$SAFE_BRANCH_NAME"
LOG_DIR="$ROOT_DIR/tests/logs"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_FILE="$RUN_LOG_DIR/summary.txt"
SUITE_LOG_FILE="$RUN_LOG_DIR/suite.log"
mkdir -p "$RUN_LOG_DIR"
exec > >(tee "$SUITE_LOG_FILE") 2>&1
echo "Logging output to: $RUN_LOG_DIR"
echo "CPU thread limit: $CPU_COUNT"
echo "Test mode: $TEST_MODE"
echo "Performance regression runner"

declare -a SUMMARY_LINES=()
TOTAL_CASES=0
PASS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

run_case() {
    local name="$1"
    local case_name="$2"
    local platform="$3"
    local log_file="$RUN_LOG_DIR/${name,,}_${platform}.log"
    local exit_code result_line status reason mode runtime jit_runtime compiled_runtime line part key value

    TOTAL_CASES=$((TOTAL_CASES + 1))
    echo "== ${name} (${platform}) ==================================="

    set +e
    PYTHONPATH=$PWD python tests/regressions.py --case "$case_name" --platform "$platform" --mode "$TEST_MODE" > >(tee "$log_file") 2>&1
    exit_code=$?
    set -e

    result_line=$(grep '^RESULT ' "$log_file" | tail -n 1 || true)
    status=""
    reason=""
    mode=""
    runtime=""
    jit_runtime=""
    compiled_runtime=""
    if [[ -n "$result_line" ]]; then
        for part in ${result_line#RESULT }; do
            key=${part%%=*}
            value=${part#*=}
            case "$key" in
                status) status="$value" ;;
                reason) reason="$value" ;;
                mode) mode="$value" ;;
                runtime) runtime="$value" ;;
                jit_runtime) jit_runtime="$value" ;;
                compiled_runtime) compiled_runtime="$value" ;;
            esac
        done
    fi

    if [[ -z "$status" ]]; then
        if [[ "$exit_code" -eq 0 ]]; then
            status="PASS"
        elif [[ "$exit_code" -eq 80 ]]; then
            status="SKIP"
            reason="skip_without_result_line"
        else
            status="FAIL"
            reason="exit_${exit_code}"
        fi
    fi

    if [[ -z "$mode" ]]; then
        mode="$TEST_MODE"
    fi

    case "$status" in
        PASS)
            PASS_COUNT=$((PASS_COUNT + 1))
            if [[ -n "$compiled_runtime" && -n "$jit_runtime" ]]; then
                line=$(printf '%-9s %-4s %-4s jit=%ss compiled=%ss' "$name" "$platform" "$status" "$jit_runtime" "$compiled_runtime")
            else
                line=$(printf '%-9s %-4s %-4s runtime=%ss' "$name" "$platform" "$status" "$runtime")
            fi
            ;;
        SKIP)
            SKIP_COUNT=$((SKIP_COUNT + 1))
            line=$(printf '%-9s %-4s %-4s %s' "$name" "$platform" "$status" "$reason")
            ;;
        *)
            FAIL_COUNT=$((FAIL_COUNT + 1))
            line=$(printf '%-9s %-4s %-4s %s' "$name" "$platform" "FAIL" "$reason")
            ;;
    esac

    SUMMARY_LINES+=("$line")
    echo "$line"
    echo "Log file: $log_file"
}

run_case "ESR" "ESR" "cpu"
run_case "ESR" "ESR" "gpu"
run_case "NMR" "NMR" "cpu"
run_case "NMR" "NMR" "gpu"
run_case "DDRF" "DDRF" "cpu"
run_case "DDRF" "DDRF" "gpu"
run_case "AUTODIFF" "AUTODIFF" "cpu"
run_case "AUTODIFF" "AUTODIFF" "gpu"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    OVERALL_STATUS="FAIL"
elif [[ "$SKIP_COUNT" -gt 0 ]]; then
    OVERALL_STATUS="PASS_WITH_SKIPS"
else
    OVERALL_STATUS="PASS"
fi

{
    echo "Test run: $RUN_ID"
    echo "Logs: $RUN_LOG_DIR"
    echo "Mode: $TEST_MODE"
    echo
    for line in "${SUMMARY_LINES[@]}"; do
        echo "$line"
    done
    echo
    echo "Totals: total=$TOTAL_CASES pass=$PASS_COUNT skip=$SKIP_COUNT fail=$FAIL_COUNT"
    echo "Result: $OVERALL_STATUS"
} | tee "$SUMMARY_FILE"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi
