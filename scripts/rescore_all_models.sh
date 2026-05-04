#!/usr/bin/env bash
# Re-score every model directory under the results root using the updated
# (re-parsed) .predictions.jsonl files. Overwrites scores.json and
# per_sample_scores.jsonl in place.

set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-<repo-root>/results/robustbench/results}"
EVAL_SCRIPT="${EVAL_SCRIPT:-<repo-root>/scripts/eval_results.py}"

echo "Scoring all model directories under: ${RESULTS_ROOT}"
echo

for mdir in "${RESULTS_ROOT}"/*/; do
    [ -d "$mdir" ] || continue
    name="$(basename "$mdir")"
    # Skip anything without .predictions.jsonl
    if ! ls "$mdir"*.predictions.jsonl 1>/dev/null 2>&1; then
        echo "SKIP $name (no .predictions.jsonl)"
        continue
    fi
    echo "=== $name ==="
    python3 "$EVAL_SCRIPT" \
        --predictions "$mdir" \
        --output "${mdir}scores.json" \
        --per-sample-output "${mdir}per_sample_scores.jsonl" \
        2>&1 | tail -8
    echo
done

echo "Done."
