#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source "$(dirname "$0")/_detect_python.sh"

RESULTS_DIR="$PROJECT_DIR/results_sonnet"
DATA_DIR="$PROJECT_DIR/data_splits"

echo "=== Sonnet Eval: $(date) ==="
for SUBDIR in "$RESULTS_DIR"/*/; do
    [ -d "$SUBDIR" ] || continue
    DIRNAME=$(basename "$SUBDIR")
    [ "$DIRNAME" = "trajectories" ] && continue

    # Skip if eval pkl already exists
    if ls "$SUBDIR"/*_eval.pkl &>/dev/null; then
        echo "SKIP (already has eval): $DIRNAME"
        continue
    fi
    # Skip if no results pkl
    if ! ls "$SUBDIR"/*_results.pkl &>/dev/null; then
        echo "SKIP (no results pkl): $DIRNAME"
        continue
    fi

    PATHOLOGY="${DIRNAME%%_ZeroShot_*}"
    PATIENT_DATA="$DATA_DIR/$PATHOLOGY/test.pkl"
    if [ ! -f "$PATIENT_DATA" ]; then
        echo "SKIP (no test.pkl): $PATHOLOGY"
        continue
    fi

    echo ""
    echo "--- Evaluating: $PATHOLOGY ---"
    "$PYTHON" "$PROJECT_DIR/scripts/evaluate_run.py" \
        --results_dir "$SUBDIR" \
        --pathology "$PATHOLOGY" \
        --patient_data "$PATIENT_DATA" || echo "  WARNING: eval failed for $PATHOLOGY"
done

echo ""
echo "=== Sonnet Eval COMPLETE: $(date) ==="
