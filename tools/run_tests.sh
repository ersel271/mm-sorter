#!/usr/bin/env bash
# tools/run_tests.sh
# run the full test suite with verbose output and coverage report

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

REPORT_DIR="${PROJECT_ROOT}/_report"
COV_HTML_DIR="${REPORT_DIR}/htmlcov"
PYTEST_REPORT="${REPORT_DIR}/report.html"

python3 -m pytest \
    -c tests/pytest.ini \
    --cov-report=html:"$COV_HTML_DIR" \
    --html="$PYTEST_REPORT" \
    --self-contained-html