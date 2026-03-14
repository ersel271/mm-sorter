#!/usr/bin/env bash
# tools/setup.sh
# set up the development environment: check for virtualenv,
# install dependencies, and prepare runtime directories
# usage: bash tools/setup.sh [dev|prod]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/testvenv"
MODE="${1:-prod}"

if [ "$MODE" = "dev" ]; then
    REQ_FILE="$PROJECT_ROOT/requirements-dev.txt"
else
    REQ_FILE="$PROJECT_ROOT/requirements.txt"
fi

if [ ! -f "$REQ_FILE" ]; then
    echo "error: $REQ_FILE not found"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "no active virtualenv detected"
    if [ ! -d "$VENV_DIR" ]; then
        echo "creating virtualenv in $VENV_DIR ..."
        python3 -m venv "$VENV_DIR"
    fi
    echo "activate it first, then re-run this script:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  bash $0"
    exit 1
fi

echo "installing dependencies ($MODE) ..."
pip install -r "$REQ_FILE" > /dev/null

chmod +x "$PROJECT_ROOT"/tools/*.sh

echo "creating runtime directories ..."
mkdir -p "$PROJECT_ROOT"/data/{logs,events,samples}

if [ "$MODE" = "dev" ]; then
    echo "creating dev environment specific directories ..."
    mkdir -p "$PROJECT_ROOT/_report"
fi

echo "setup complete"