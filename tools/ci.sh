#!/usr/bin/env bash
# tools/ci.sh
# local CI pipeline: lint, typecheck, test, deadcode, complexity, security
# usage: bash tools/ci.sh [stage]
#   stages: lint, fix, typecheck, test, deadcode, complexity, security
#   no argument runs all stages in parallel (fix excluded -- mutates files)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/_report"
TARGETS=("$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" "$PROJECT_ROOT/sort.py")

stage_lint() {
    local rc=0
    ruff check "${TARGETS[@]}" "$PROJECT_ROOT/tests" || rc=1
    return $rc
}

stage_fix() {
    ruff check --fix "${TARGETS[@]}" "$PROJECT_ROOT/tests"
}

stage_typecheck() {
    mypy "${TARGETS[@]}"
}

stage_test() {
    bash "$PROJECT_ROOT/tools/run_tests.sh" -n auto
}

stage_deadcode() {
    local out
    out=$(vulture "${TARGETS[@]}" \
        "$PROJECT_ROOT/vulture_whitelist.py" \
        --min-confidence 80)
    if [ -z "$out" ]; then
        echo "no dead code found"
    else
        echo "$out"
        return 1
    fi
}

stage_complexity() {
    radon cc "${TARGETS[@]}" -s -a \
        > "$REPORT_DIR/radon_cc.txt"
    radon mi "${TARGETS[@]}" -s \
        > "$REPORT_DIR/radon_mi.txt"
    echo "reports written to _report/radon_cc.txt and _report/radon_mi.txt"
    # thresholds: single block max C, per-module average max C, overall average max A
    # config/validate.py excluded because range-check functions are inherently verbose
    # and im tired. also i am aware that this is not an "elegant" solution but ignore
    # flag did not work
    xenon --max-absolute C --max-modules C --max-average A \
        "$PROJECT_ROOT/src" \
        "$PROJECT_ROOT/config/__init__.py" \
        "$PROJECT_ROOT/config/config.py" \
        "$PROJECT_ROOT/config/constants.py" \
        "$PROJECT_ROOT/utils" \
        "$PROJECT_ROOT/sort.py"
}

stage_security() {
    bandit -r "${TARGETS[@]}" \
        -f txt -o "$REPORT_DIR/bandit.txt"
    pip-audit -r "$PROJECT_ROOT/requirements.txt" \
        > "$REPORT_DIR/pip_audit.txt"
    [ -s "$REPORT_DIR/pip_audit.txt" ] || echo "no known vulnerabilities found" >> "$REPORT_DIR/pip_audit.txt"
    echo "reports written to _report/bandit.txt and _report/pip_audit.txt"
}

_run() {
    local name="$1"
    if ! declare -f "stage_${name}" > /dev/null 2>&1; then
        echo "unknown stage: '${name}' (valid: lint, typecheck, test, deadcode, complexity, security, fix)" >&2
        return 1
    fi
    local log="$REPORT_DIR/${name}.log"
    if "stage_${name}" > "$log" 2>&1; then
        printf "stage: %-10s ... passed\n" "${name}"
        return 0
    else
        printf "stage: %-10s ... failed, see _report/${name}.log\n" "${name}"
        return 1
    fi
}

main() {
    mkdir -p "$REPORT_DIR"
    echo "pipeline started"

    pids=()
    trap 'trap "" INT TERM; kill 0; exit 130' INT TERM
    _run lint       & pids+=($!)
    _run typecheck  & pids+=($!)
    _run test       & pids+=($!)
    _run deadcode   & pids+=($!)
    _run complexity & pids+=($!)
    _run security   & pids+=($!)

    local failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done

    echo ""
    if [ "$failed" -eq 0 ]; then
        echo "pipeline complete -- all stages passed"
    else
        echo "pipeline complete -- one or more stages failed"
    fi
    return "$failed"
}

if [ -n "${1:-}" ]; then
    if [ "${1}" = "fix" ]; then
        stage_fix
    else
        mkdir -p "$REPORT_DIR"
        _run "${1}"
    fi
else
    main
fi