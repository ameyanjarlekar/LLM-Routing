#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s <T> <comma-separated-models> <comma-separated-compute-values>\n' "$0" >&2
    printf 'Example: %s 100 "model-a,model-b" "120,80"\n' "$0" >&2
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

T="$1"
MODELS_CSV="$2"
COMPUTE_VALUES_CSV="$3"

if [[ ! "$T" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Error: T must be a positive integer.\n' >&2
    exit 2
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a COMPUTE_VALUES <<< "$COMPUTE_VALUES_CSV"

if [[ ${#MODELS[@]} -eq 0 || ${#MODELS[@]} -ne ${#COMPUTE_VALUES[@]} ]]; then
    printf 'Error: provide one compute value for every model.\n' >&2
    exit 2
fi

for model in "${MODELS[@]}"; do
    if [[ -z "${model//[[:space:]]/}" ]]; then
        printf 'Error: model IDs must not be empty.\n' >&2
        exit 2
    fi
done

python - "${COMPUTE_VALUES[@]}" <<'PY'
import math
import sys

try:
    values = [float(value) for value in sys.argv[1:]]
except ValueError:
    raise SystemExit("Error: compute values must be numeric.")

if not values or any(not math.isfinite(value) or value <= 0 for value in values):
    raise SystemExit("Error: compute values must be finite and greater than zero.")
PY

if [[ -z "${LUMEN_API_KEY:-}" ]]; then
    printf 'Error: set LUMEN_API_KEY before running the routing pipeline.\n' >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/llm_routing_poc_data"

python run_routing.py \
    --models "$MODELS_CSV" \
    --compute-values "$COMPUTE_VALUES_CSV" \
    --n-queries "$T" \
    --output-dir "$DATA_DIR"