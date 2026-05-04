#!/usr/bin/env bash
# Evaluate openai/o4-mini on RobustBench-TC via the public OpenAI API.
#
# Usage (you run this yourself — don't paste the key here):
#   export OPENAI_API_KEY=sk-...
#   USE_FC=1 LIMIT=5 bash scripts/run_o4mini_eval.sh   # smoke test, 5 samples
#   USE_FC=1 bash scripts/run_o4mini_eval.sh           # full run
#   unset OPENAI_API_KEY
#
# Notes:
#   - o4-mini is a reasoning model; OpenAI's API requires max_completion_tokens
#     (not max_tokens) and accepts only the default temperature. Both handled
#     in run_eval.py via the is_openai_reasoning detection branch.
#   - Default mode is FC (USE_FC=1) — o4-mini natively supports OpenAI tool
#     calling (tools=[...]) and prompt-mode would underutilize it. Set
#     USE_FC=0 to force prompt mode for an apples-to-apples vs ToolRL family.
#   - Max output tokens 8192 — reasoning models burn many tokens on the
#     hidden think trace; 4096 was occasionally too tight at the GPT-5
#     parity tier so we bump it for o4-mini.
#   - Outputs land at <repo-root>/results/robustbench/results/openai__o4-mini/
#
set -euo pipefail

# Defaults — can be overridden via env (USE_FC=0 bash run_o4mini_eval.sh)
USE_FC="${USE_FC:-1}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
LIMIT="${LIMIT:-}"

FC_FLAG=""
[[ "${USE_FC}" == "1" ]] && FC_FLAG="--use-fc"
LIMIT_FLAG=""
[[ -n "${LIMIT}" ]] && LIMIT_FLAG="--limit ${LIMIT}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: export OPENAI_API_KEY=sk-... before running this script."
    exit 2
fi

# Activate the cluster's base conda env (login nodes have no bare 'python').
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    source /sw/user/python/miniforge3-pytorch-2.10.0/etc/profile.d/conda.sh
    conda activate base
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 still not on PATH after conda activate. Source it manually."
    exit 3
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${ROOT}/hf_data/datasets/api_eval"
OUTPUT_DIR="<repo-root>/results/robustbench/results/openai__o4-mini"
mkdir -p "${OUTPUT_DIR}"

MODEL="o4-mini"
ENDPOINT="https://api.openai.com/v1"

# Single-mode pass (clean + 16 static perturbations).
python3 "${ROOT}/scripts/run_eval.py" \
    --model-name "${MODEL}" \
    --endpoint "${ENDPOINT}" \
    --api-key "${OPENAI_API_KEY}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode single \
    --max-tokens "${MAX_TOKENS}" \
    --skip-multi-turn \
    ${FC_FLAG} ${LIMIT_FLAG}

# Transition pass (6 variants). run_eval.py's --transition-type takes the
# short form (timeout, rate_limit, ...) not the JSONL filename prefix.
for T in timeout rate_limit auth_error server_error malformed_response schema_drift; do
    python3 "${ROOT}/scripts/run_eval.py" \
        --model-name "${MODEL}" \
        --endpoint "${ENDPOINT}" \
        --api-key "${OPENAI_API_KEY}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}/transition_${T}" \
        --mode transition \
        --transition-type "${T}" \
        --max-tokens "${MAX_TOKENS}" \
        --skip-multi-turn \
        ${FC_FLAG} ${LIMIT_FLAG}
done

# Score the single-mode predictions (clean + 16 static perturbations).
python3 "${ROOT}/scripts/eval_results.py" \
    --predictions "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/scores.json" \
    --per-sample-output "${OUTPUT_DIR}/per_sample_scores.jsonl"

# Each transition_<T>/ directory holds its own clean.predictions.jsonl;
# score them and merge into the top-level scores.json afterward.
for T in timeout rate_limit auth_error server_error malformed_response schema_drift; do
    DIR="${OUTPUT_DIR}/transition_${T}"
    if [[ -d "${DIR}" ]]; then
        python3 "${ROOT}/scripts/eval_results.py" \
            --predictions "${DIR}" \
            --output "${DIR}/scores.json" \
            --per-sample-output "${DIR}/per_sample_scores.jsonl"
    fi
done

# Re-aggregate top-level scores.json by re-running the scorer over EVERY
# predictions file (single + transitions) so render_figures.py picks them
# all up. The scorer auto-discovers transition_*/clean.predictions.jsonl.
python3 -c "
import json, glob, os
from pathlib import Path
out_dir = Path('${OUTPUT_DIR}')
agg = {'overall': {'overall': {'n':0,'correct':0}, 'by_perturbation': {}, 'by_mdp_category': {}}}
for sj in [out_dir/'scores.json'] + sorted(out_dir.glob('transition_*/scores.json')):
    if not sj.exists(): continue
    s = json.loads(sj.read_text())
    for k, v in s['overall'].get('by_perturbation', {}).items():
        agg['overall']['by_perturbation'][k] = v
        agg['overall']['overall']['n']       += v.get('n',0)
        agg['overall']['overall']['correct'] += v.get('correct', round(v.get('accuracy',0)*v.get('n',0)))
    for k, v in s['overall'].get('by_mdp_category', {}).items():
        if k not in agg['overall']['by_mdp_category']:
            agg['overall']['by_mdp_category'][k] = {'n':0,'correct':0}
        agg['overall']['by_mdp_category'][k]['n']       += v.get('n',0)
        agg['overall']['by_mdp_category'][k]['correct'] += v.get('correct', round(v.get('accuracy',0)*v.get('n',0)))
n = agg['overall']['overall']['n']; c = agg['overall']['overall']['correct']
agg['overall']['overall']['accuracy'] = c/n if n else 0
for k, v in agg['overall']['by_mdp_category'].items():
    v['accuracy'] = v['correct']/v['n'] if v['n'] else 0
(out_dir/'scores.json').write_text(json.dumps(agg, indent=2))
print(f'Aggregated scores: n={n} acc={c/n:.4f}')
"

echo
echo "✅ o4-mini eval complete. Scores at:"
echo "   ${OUTPUT_DIR}/scores.json"
echo "   Re-run scripts/render_figures.py to add o4-mini to the heatmap."
