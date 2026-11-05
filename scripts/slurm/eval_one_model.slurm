#!/bin/bash
#SBATCH --job-name=rbtc_eval
#SBATCH --partition=ghx4
#SBATCH --account=<allocation>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --output=<repo-root>/scripts/logs/rbtc_%x_%j.out
#SBATCH --error=<repo-root>/scripts/logs/rbtc_%x_%j.err
#
# AgentRobustBench — single-model end-to-end evaluator on <HPC-cluster>.
# Starts a local vLLM server on the allocated GH200 GPU(s), runs our
# scripts/run_eval.py (single-mode + transition-mode), then
# scripts/eval_results.py, then tears down the vLLM server.
#
# Submit with env vars controlling which model to evaluate:
#
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
#   GPUS=1  DTYPE=bfloat16  MAX_MODEL_LEN=8192  \
#   EXTRA_VLLM_FLAGS="" \
#   sbatch --gres=gpu:nvidia_gh200_120gb:1 \
#          --job-name=rbtc_qwen2.5_3b \
#          scripts/slurm/eval_one_model.slurm
#
# For 32B (MUA-RL) you need 2 GPUs:
#   MODEL_ID=zzwkk/MUA-RL-32B  GPUS=2  \
#   EXTRA_VLLM_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" \
#   sbatch --gres=gpu:nvidia_gh200_120gb:2 --job-name=rbtc_mua32b \
#          scripts/slurm/eval_one_model.slurm
#
# Helper ``scripts/slurm/submit_all.sh`` submits this for all 13 models
# from model_manifest.py.

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# 1. Required / optional env vars
# ────────────────────────────────────────────────────────────────
: "${MODEL_ID:?MODEL_ID env var must be set (HF repo id, e.g. Qwen/Qwen3.5-9B)}"
: "${GPUS:=1}"
: "${DTYPE:=bfloat16}"
: "${MAX_MODEL_LEN:=16384}"
: "${EXTRA_VLLM_FLAGS:=}"
: "${VLLM_PORT:=8765}"
: "${REPO_ROOT:=<repo-root>}"
: "${DATA_DIR:=${REPO_ROOT}/hf_data/datasets/api_eval}"
: "${RESULT_ROOT:=<repo-root>/results/robustbench/results}"
: "${HF_HOME:=<repo-root>/cache/hf/.cache/huggingface}"
: "${RUN_TRANSITION:=1}"            # set 0 to skip transition mode
# Default is "all" → run all 6 transient-error variants. Override to a
# single name (timeout|rate_limit|auth_error|server_error|malformed_response
# |schema_drift) for partial / backfill runs.
: "${TRANSITION_TYPE:=all}"
: "${MAX_WORKERS:=16}"              # parallel requests to vLLM
: "${MAX_TOKENS:=1024}"             # generation budget per call (4096 for reasoning models)
: "${TEMPERATURE:=0.0}"             # 0.0 default (deterministic / reproducible). Override to 0.6 for DeepSeek-R1.
: "${NO_SYSTEM_PROMPT:=0}"          # set to 1 for DeepSeek-R1 — folds system into user message
# Qwen3-family thinking-mode switch. "disable" → pass chat_template_kwargs
# enable_thinking=False (REQUIRED for MUA-RL/LoopTool/Qwen3-Instruct to
# match the non-thinking eval used in their papers). "enable" → True
# (rarely needed). Empty → omit the kwarg (model uses its default).
: "${THINKING_MODE:=}"              # enable | disable | ""
# Optional: delete this model's HF cache after eval completes (saves disk
# when running many new 14B+/32B models that we only need once). Set to 1
# for new model downloads; leave unset (0) for re-runs that use already
# cached weights and want to keep them.
: "${CLEAN_CACHE:=0}"
# Native function-calling mode: pass `--use-fc` to run_eval.py so it sends
# tools via the OpenAI `tools=` API field and reads structured tool_calls.
# Required for Qwen3 family (MUA-RL/LoopTool/Qwen3/Qwen3.5) to match their
# paper eval; leave at 0 for ToolRL/TL-CodeLLaMA/Qwen2.5-Instruct (prompt
# mode, which is how their papers evaluate).
: "${USE_FC:=0}"
: "${LIMIT:=}"                      # optional: --limit N for quick sanity

# Sanitize model id for filesystem use
MODEL_SAFE="${MODEL_ID//\//__}"
OUTPUT_DIR="${RESULT_ROOT}/${MODEL_SAFE}"
mkdir -p "${OUTPUT_DIR}"

export HF_HOME
mkdir -p "${HF_HOME}"

echo "=============================================="
echo " AgentRobustBench eval"
echo " Job ID       : ${SLURM_JOB_ID:-local}"
echo " Node         : $(hostname)"
echo " Start        : $(date)"
echo " MODEL_ID     : ${MODEL_ID}"
echo " GPUS         : ${GPUS}"
echo " DTYPE        : ${DTYPE}"
echo " MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo " VLLM_PORT    : ${VLLM_PORT}"
echo " DATA_DIR     : ${DATA_DIR}"
echo " OUTPUT_DIR   : ${OUTPUT_DIR}"
echo " HF_HOME      : ${HF_HOME}"
echo " RUN_TRANSITION: ${RUN_TRANSITION} (type=${TRANSITION_TYPE})"
echo " TEMPERATURE : ${TEMPERATURE}"
echo " MAX_TOKENS  : ${MAX_TOKENS}"
echo " THINKING    : ${THINKING_MODE:-<default>}"
echo " USE_FC      : ${USE_FC}"
echo " CLEAN_CACHE : ${CLEAN_CACHE}"
echo "=============================================="

# ────────────────────────────────────────────────────────────────
# 2. Environment setup (<HPC-cluster>)
# ────────────────────────────────────────────────────────────────
module reset
module load python/miniforge3_pytorch/2.10.0 2>/dev/null || true
module load cuda/12.9 2>/dev/null || true

# <HPC-cluster>'s miniforge3-pytorch base env already bundles vllm+torch+openai+tqdm
# (verified 2026-04-22 on <login-node>: vllm=0.19.0, torch=2.10.0+cu129,
#  openai=2.26.0, tqdm=4.67.3). So we just activate base.
: "${CONDA_ENV:=base}"
: "${MINIFORGE_ROOT:=/sw/user/python/miniforge3-pytorch-2.10.0}"

echo "[debug] MINIFORGE_ROOT=${MINIFORGE_ROOT}"
echo "[debug] CONDA_ENV=${CONDA_ENV}"
ls -ld "${MINIFORGE_ROOT}" 2>&1 | head -2
echo "[debug] conda.sh exists? $(test -f ${MINIFORGE_ROOT}/etc/profile.d/conda.sh && echo yes || echo no)"

source "${MINIFORGE_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}" || {
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 2
}

# Some HPC environments install vllm into user-local site-packages while
# torch lives in the shared base env. Ensure user-local is explicitly on
# PYTHONPATH for sbatch jobs (srun inherits it automatically; sbatch's
# --export=ALL,VAR=X handling can drop user-site under some SLURM configs).
USER_SITE="$(python -c 'import site; print(site.getusersitepackages())')"
if [[ -d "${USER_SITE}" ]]; then
    export PYTHONPATH="${USER_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[debug] after conda activate:"
echo "  which python: $(which python)"
echo "  python --version: $(python --version 2>&1)"
echo "  CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "  PYTHONPATH=${PYTHONPATH:-unset}"

python -c "import vllm, openai; print('vllm', vllm.__version__, '| openai', openai.__version__)"

# ────────────────────────────────────────────────────────────────
# 3. Start vLLM server (background)
# ────────────────────────────────────────────────────────────────
VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"
echo "[$(date '+%H:%M:%S')] Launching vLLM (log: ${VLLM_LOG})"

VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL_ID}"
    --served-model-name "${MODEL_ID}"
    --host 0.0.0.0
    --port "${VLLM_PORT}"
    --tensor-parallel-size "${GPUS}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --download-dir "${HF_HOME}/hub"
    --no-enable-log-requests
)
# Append extra flags split by whitespace (e.g. --tool-call-parser hermes)
if [[ -n "${EXTRA_VLLM_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=(${EXTRA_VLLM_FLAGS})
    VLLM_CMD+=("${EXTRA_ARR[@]}")
fi

"${VLLM_CMD[@]}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "[$(date '+%H:%M:%S')] vLLM PID=${VLLM_PID}"

cleanup() {
    echo "[$(date '+%H:%M:%S')] Shutting down vLLM (PID ${VLLM_PID})"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until API is healthy (max 20 min for big models)
ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1"
READY=0
for i in $(seq 1 240); do
    if curl -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
        READY=1
        echo "[$(date '+%H:%M:%S')] vLLM ready after ${i}×5s"
        break
    fi
    sleep 5
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Tail of log:"
        tail -50 "${VLLM_LOG}" || true
        exit 3
    fi
done
if [[ ${READY} -ne 1 ]]; then
    echo "ERROR: vLLM did not become ready within 20 min. Tail of log:"
    tail -50 "${VLLM_LOG}" || true
    exit 4
fi

# ────────────────────────────────────────────────────────────────
# 4. Run single-mode inference across all 17 api_eval files
# ────────────────────────────────────────────────────────────────
RUN_EVAL_ARGS=(
    --model-name "${MODEL_ID}"
    --endpoint "${ENDPOINT}"
    --data-dir "${DATA_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --max-workers "${MAX_WORKERS}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --resume
)
if [[ "${NO_SYSTEM_PROMPT}" == "1" ]]; then
    RUN_EVAL_ARGS+=(--no-system-prompt)
fi
if [[ "${THINKING_MODE}" == "disable" ]]; then
    RUN_EVAL_ARGS+=(--disable-thinking)
elif [[ "${THINKING_MODE}" == "enable" ]]; then
    RUN_EVAL_ARGS+=(--enable-thinking)
fi
if [[ "${USE_FC}" == "1" ]]; then
    RUN_EVAL_ARGS+=(--use-fc)
fi
if [[ -n "${LIMIT}" ]]; then
    RUN_EVAL_ARGS+=(--limit "${LIMIT}")
fi

echo "[$(date '+%H:%M:%S')] run_eval.py single mode"
python "${REPO_ROOT}/scripts/run_eval.py" "${RUN_EVAL_ARGS[@]}"

# ────────────────────────────────────────────────────────────────
# 5. Run transition mode on clean.jsonl (optional)
# ────────────────────────────────────────────────────────────────
if [[ "${RUN_TRANSITION}" == "1" ]]; then
    # TRANSITION_TYPE="all" → run all 6 variants in sequence (reuses same
    # vLLM server, saves ~2 min startup × 5 variants).
    # Otherwise run just the single TRANSITION_TYPE specified.
    if [[ "${TRANSITION_TYPE}" == "all" ]]; then
        TRANSITION_TYPES=("timeout" "rate_limit" "auth_error" \
                          "server_error" "malformed_response" "schema_drift")
    else
        TRANSITION_TYPES=("${TRANSITION_TYPE}")
    fi
    for TR_TYPE in "${TRANSITION_TYPES[@]}"; do
        echo "[$(date '+%H:%M:%S')] run_eval.py transition mode (type=${TR_TYPE})"
        TR_ARGS=(
            --model-name "${MODEL_ID}"
            --endpoint "${ENDPOINT}"
            --data-dir "${DATA_DIR}"
            --output-dir "${OUTPUT_DIR}"
            --max-workers "${MAX_WORKERS}"
            --max-tokens "${MAX_TOKENS}"
            --temperature "${TEMPERATURE}"
            --resume
            --mode transition --transition-type "${TR_TYPE}"
        )
        if [[ "${NO_SYSTEM_PROMPT}" == "1" ]]; then
            TR_ARGS+=(--no-system-prompt)
        fi
        if [[ "${THINKING_MODE}" == "disable" ]]; then
            TR_ARGS+=(--disable-thinking)
        elif [[ "${THINKING_MODE}" == "enable" ]]; then
            TR_ARGS+=(--enable-thinking)
        fi
        if [[ "${USE_FC}" == "1" ]]; then
            TR_ARGS+=(--use-fc)
        fi
        if [[ -n "${LIMIT}" ]]; then
            TR_ARGS+=(--limit "${LIMIT}")
        fi
        python "${REPO_ROOT}/scripts/run_eval.py" "${TR_ARGS[@]}"
    done
fi

# ────────────────────────────────────────────────────────────────
# 6. Score + aggregate
# ────────────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] eval_results.py"
python "${REPO_ROOT}/scripts/eval_results.py" \
    --predictions "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/scores.json" \
    --per-sample-output "${OUTPUT_DIR}/per_sample_scores.jsonl"

echo "[$(date '+%H:%M:%S')] Done. Summary:"
python -c "
import json, sys
with open('${OUTPUT_DIR}/scores.json') as f: data = json.load(f)
ov = data['overall']['overall']
print(f\"  overall: n={ov['n']} acc={ov['accuracy']:.4f}\")
for k, v in sorted(data['overall']['by_mdp_category'].items()):
    print(f\"  {k:14s}: n={v['n']:4d} acc={v['accuracy']:.4f}\")
"


# ────────────────────────────────────────────────────────────────
# 7. Optional post-eval HF cache cleanup
# ────────────────────────────────────────────────────────────────
if [[ "${CLEAN_CACHE}" == "1" ]]; then
    # HF cache dir name follows: models--<org>--<name>
    # Derive from MODEL_ID = "<org>/<name>"
    CACHE_MODEL_DIR="${HF_HOME}/hub/models--${MODEL_ID//\//--}"
    if [[ -d "${CACHE_MODEL_DIR}" ]]; then
        CACHE_SZ=$(du -sh "${CACHE_MODEL_DIR}" 2>/dev/null | cut -f1)
        echo "[$(date '+%H:%M:%S')] CLEAN_CACHE=1 — removing ${CACHE_MODEL_DIR} (${CACHE_SZ})"
        rm -rf "${CACHE_MODEL_DIR}"
    else
        echo "[$(date '+%H:%M:%S')] CLEAN_CACHE=1 but ${CACHE_MODEL_DIR} not found; skipping."
    fi
fi

echo "=============================================="
echo " End          : $(date)"
echo "=============================================="
