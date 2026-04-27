#!/usr/bin/env bash
# fixture.sh — add and verify reference audio fixtures
#
# The script handles everything: venv setup, analysis, and running tests.
# Audio files (.wav/.mp3) are gitignored; the generated .reference.json
# files are tracked so tests can run on CI without the audio.
#
# Usage:
#   fixture.sh <file>            copy <file> to fixtures/, analyze, run tests
#   fixture.sh                   re-analyze all fixtures and run tests
#   fixture.sh --analyze-only    (re-)generate JSON without running tests
#   fixture.sh --test-only       run tests against existing JSON
#   fixture.sh --fixtures <dir>  use a different fixtures directory
#
# Fixtures default to ./fixtures relative to the current working directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
ANALYZE=true
RUN_TESTS=true
INPUT_FILE=""
FIXTURES_DIR="$REPO_ROOT/v2/fixtures"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --analyze-only) RUN_TESTS=false ; shift ;;
        --test-only)    ANALYZE=false   ; shift ;;
        --fixtures)     FIXTURES_DIR="$(cd "$2" 2>/dev/null && pwd || echo "$2")" ; shift 2 ;;
        --help|-h)      sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ; exit 0 ;;
        -*) echo "Unknown option: $1" >&2 ; exit 1 ;;
        *)  INPUT_FILE="$1" ; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
setup_venv() {
    if [[ ! -x "$PYTHON" ]]; then
        echo "Setting up Python environment…"
        python3 -m venv "$VENV_DIR"
    fi
    local stamp="$VENV_DIR/.requirements_installed"
    if [[ ! -f "$stamp" || "$SCRIPT_DIR/requirements.txt" -nt "$stamp" ]]; then
        echo "Installing dependencies…"
        "$PYTHON" -m pip install --quiet --upgrade pip
        "$PYTHON" -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
        touch "$stamp"
    fi
}

copy_input() {
    local src="$1"
    [[ -f "$src" ]] || { echo "Error: file not found: $src" >&2; exit 1; }
    local dest="$FIXTURES_DIR/$(basename "$src")"
    if [[ "$(realpath "$src")" != "$(realpath "$dest" 2>/dev/null || echo "")" ]]; then
        echo "Copying $(basename "$src") → fixtures/"
        cp "$src" "$dest"
    fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$FIXTURES_DIR"
setup_venv

[[ -n "$INPUT_FILE" ]] && copy_input "$INPUT_FILE"

if $ANALYZE; then
    echo ""
    "$PYTHON" "$SCRIPT_DIR/analyze_reference.py" "$FIXTURES_DIR"
fi

if $RUN_TESTS; then
    echo ""
    echo "Running fixture tests…"
    cargo test \
        --manifest-path "$REPO_ROOT/v2/Cargo.toml" \
        --package kaidee-analysis \
        reference_json \
        -- --include-ignored --nocapture
fi
