#!/bin/bash
# Submit AgentRobustBench eval SLURM jobs for the 13 models in model_manifest.py.
#
# Usage:
#   # Submit everything (default: all models, with transition=timeout):
#   bash scripts/slurm/submit_all.sh
#
#   # Subset by category (e.g. just the frontier open-weights):
#   bash scripts/slurm/submit_all.sh --category frontier_open
#
#   # Subset by model id glob:
#   bash scripts/slurm/submit_all.sh --models 'Qwen/*' 'deepseek-ai/*'
#
#   # Dry run: print sbatch commands but don't submit
#   bash scripts/slurm/submit_all.sh --dry-run
#
# Environment overrides passed through to eval_one_model.slurm:
#   DATA_DIR, RESULT_ROOT, HF_HOME, RUN_TRANSITION, TRANSITION_TYPE, LIMIT, etc.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-<repo-root>}"
SLURM_DIR="${REPO_ROOT}/scripts/slurm"

DRY_RUN=0
CATEGORY_FILTER=""
MODELS_FILTER=()
CLEAN_CACHE_FLAG=0     # 0=keep caches (default), 1=delete each model's HF cache after its eval

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --clean-cache) CLEAN_CACHE_FLAG=1; shift ;;
        --category) CATEGORY_FILTER="$2"; shift 2 ;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MODELS_FILTER+=("$1"); shift
            done ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# Need Python ≥ 3.7 for the manifest (dataclasses). The login node's
# default python3 is 3.6, so prefer /usr/bin/python3.12 or the conda
# env's python if loaded.
if command -v python3.12 >/dev/null 2>&1; then
    PYBIN="python3.12"
elif [[ -x /usr/bin/python3.12 ]]; then
    PYBIN="/usr/bin/python3.12"
else
    PYBIN="python3"
fi

# Pull manifest entries as TSV: model_id \t gpus \t dtype \t max_model_len
#   \t extra_flags \t category \t thinking_mode
mapfile -t MANIFEST < <("${PYBIN}" - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/scripts")
from model_manifest import MODELS
for mid, cfg in MODELS.items():
    if cfg.category == "frontier_closed":
        continue  # no local vLLM
    # Use sentinel "-" for empty extra_flags so bash tab-read doesn't
    # collapse the field and shift everything after it.
    extra = " ".join(cfg.extra_flags) if cfg.extra_flags else "-"
    if cfg.enable_thinking is False:
        thinking = "disable"
    elif cfg.enable_thinking is True:
        thinking = "enable"
    else:
        thinking = "-"
    use_fc = "1" if cfg.use_fc else "0"
    print(f"{mid}\t{cfg.gpus}\t{cfg.dtype}\t{cfg.max_model_len}\t{extra}\t{cfg.category}\t{thinking}\t{use_fc}")
PYEOF
)

echo "Loaded ${#MANIFEST[@]} manifest entries (excluding frontier_closed)"
echo

submit_one() {
    local mid="$1" gpus="$2" dtype="$3" max_len="$4" extra="$5" category="$6" thinking="$7" use_fc="$8"

    if [[ -n "${CATEGORY_FILTER}" && "${category}" != "${CATEGORY_FILTER}" ]]; then
        return 0
    fi
    if [[ ${#MODELS_FILTER[@]} -gt 0 ]]; then
        local match=0
        for pat in "${MODELS_FILTER[@]}"; do
            # shellcheck disable=SC2053
            if [[ "${mid}" == ${pat} ]]; then match=1; break; fi
        done
        [[ ${match} -eq 0 ]] && return 0
    fi

    local safe="${mid//\//__}"
    # Wide random range (10000–60000) to avoid collisions when multiple
    # jobs land on the same GH200 node. vLLM engine-core distributed-init
    # will pick adjacent ports, so jobs too close in port space can clash.
    local port=$(( 10000 + RANDOM % 50000 ))

    # Non-empty thinking field translates to THINKING_MODE env var
    local thinking_env=""
    if [[ "${thinking}" != "-" ]]; then
        thinking_env=",THINKING_MODE=${thinking}"
    fi

    # Optional post-eval HF cache cleanup
    local clean_env=""
    if [[ "${CLEAN_CACHE_FLAG}" -eq 1 ]]; then
        clean_env=",CLEAN_CACHE=1"
    fi

    # FC mode (native tool-calling): pass USE_FC=1 to the slurm wrapper
    local fc_env=""
    if [[ "${use_fc}" == "1" ]]; then
        fc_env=",USE_FC=1"
    fi

    local CMD=(
        sbatch
        --job-name="rbtc_${safe}"
        --gres="gpu:nvidia_gh200_120gb:${gpus}"
        --export="ALL,MODEL_ID=${mid},GPUS=${gpus},DTYPE=${dtype},MAX_MODEL_LEN=${max_len},EXTRA_VLLM_FLAGS=${extra},VLLM_PORT=${port}${thinking_env}${clean_env}${fc_env}"
        "${SLURM_DIR}/eval_one_model.slurm"
    )

    echo "[${category}] ${mid}  gpus=${gpus} dtype=${dtype} thinking=${thinking} use_fc=${use_fc} clean_cache=${CLEAN_CACHE_FLAG}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '  DRY: '
        printf '%q ' "${CMD[@]}"
        printf '\n'
    else
        "${CMD[@]}"
    fi
}

for row in "${MANIFEST[@]}"; do
    IFS=$'\t' read -r MID GPUS DTYPE MAX_LEN EXTRA CAT THINKING USE_FC <<< "${row}"
    # De-sentinel: "-" means empty extra_flags
    [[ "${EXTRA}" == "-" ]] && EXTRA=""
    submit_one "${MID}" "${GPUS}" "${DTYPE}" "${MAX_LEN}" "${EXTRA}" "${CAT}" "${THINKING}" "${USE_FC}"
done

echo
echo "Submitted. Monitor with: squeue -u \$USER"
