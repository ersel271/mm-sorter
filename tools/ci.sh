#!/usr/bin/env bash
# tools/ci.sh
# local CI pipeline: lint, typecheck, test, deadcode, complexity, security
# usage: bash tools/ci.sh [stage]
#   stages: lint, fix, typecheck, test, deadcode, complexity, security
#   no argument runs all stages in order (fix excluded -- mutates files)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/_report"

stage_lint() {
    local rc=0
    ruff check "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" "$PROJECT_ROOT/tests" || rc=1
    # ruff format --check "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" "$PROJECT_ROOT/tests" || rc=1
    return $rc
}

stage_fix() {
    ruff check --fix "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" "$PROJECT_ROOT/tests"
    # ruff format "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" "$PROJECT_ROOT/tests"
}

stage_typecheck() {
    mypy "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils"
}

stage_test() {
    bash "$PROJECT_ROOT/tools/run_tests.sh" -m "not hw and not slow"
}

stage_deadcode() {
    local out
    out=$(vulture "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" \
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
    radon cc "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" -s -a \
        > "$REPORT_DIR/radon_cc.txt"
    radon mi "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" -s \
        > "$REPORT_DIR/radon_mi.txt"
    echo "reports written to _report/radon_cc.txt and _report/radon_mi.txt"
    # thresholds: single block max C, per-module average max B, overall average max A
    xenon --max-absolute C --max-modules B --max-average A \
        "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils"
}

stage_security() {
    bandit -r "$PROJECT_ROOT/src" "$PROJECT_ROOT/config" "$PROJECT_ROOT/utils" \
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
    printf "stage: %-10s ... " "${name}"
    if "stage_${name}" > "$log" 2>&1; then
        echo "passed"
        return 0
    else
        echo "failed, see _report/${name}.log"
        return 1
    fi
}

main() {
    mkdir -p "$REPORT_DIR"
    echo "pipeline started"
    local failed=0
    
    _run lint       || failed=1
    _run typecheck  || failed=1
    _run test       || failed=1
    _run deadcode   || failed=1
    _run complexity || failed=1
    _run security   || failed=1

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
