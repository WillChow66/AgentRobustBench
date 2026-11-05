# AgentRobustBench: Running the NeurIPS Evaluation Pipeline on <HPC-cluster>

Evaluate a tool-calling checkpoint against the unified
`AgentRobustBench` benchmark (16 perturbation
files + clean baseline + 6 runtime transition variants). Scoring is fully
deterministic — **no OpenAI API key required for scoring**.

```
┌──────────────────────────────────────────────────────────────────────┐
│  HF api_eval/*.jsonl (17 files)                                      │
│        │                                                             │
│        ▼                                                             │
│   run_eval.py  ── HTTP ──►  vLLM server (your ckpt)                  │
│        │                                                             │
│        ▼                                                             │
│   *.predictions.jsonl                                                │
│        │                                                             │
│        ▼                                                             │
│   eval_results.py  ──►  scores.json                                  │
└──────────────────────────────────────────────────────────────────────┘
```

## Files

| Path | Role |
|------|------|
| `../run_eval.py` | Inference runner (single + transition modes) |
| `../eval_results.py` | Deterministic scorer + aggregator |
| `../model_manifest.py` | 13-model registry: HF id, GPU count, vLLM flags |
| `eval_one_model.slurm` | End-to-end SLURM job for ONE model |
| `submit_all.sh` | Submit `eval_one_model` for ALL or filtered models |
| `../requirements.txt` | Python deps: `vllm`, `openai`, `tqdm`, `huggingface_hub` |

## One-time setup (<HPC-cluster>)

```bash
# Activate Python env + install vLLM
module load python/miniforge3_pytorch/2.10.0
conda create -n rbtc-vllm --clone base       # run once
conda activate rbtc-vllm
pip install -r <repo-root>/scripts/requirements.txt

# Point HF cache somewhere with disk space (home dir is usually too small)
export HF_HOME=<repo-root>/cache/hf/.cache/huggingface
mkdir -p "${HF_HOME}"

# Pull the eval dataset once (21.6 MB)
hf download AgentRobustBench/AgentRobustBench \
    --repo-type dataset --include '*.jsonl' \
    --local-dir <repo-root>/hf_data/datasets/api_eval
```

## Evaluate one model (SLURM end-to-end)

```bash
MODEL_ID=Qwen/Qwen3.5-9B \
GPUS=1 DTYPE=bfloat16 MAX_MODEL_LEN=16384 \
sbatch --gres=gpu:nvidia_gh200_120gb:1 \
       --job-name=rbtc_qwen3.5_9b \
       scripts/slurm/eval_one_model.slurm
```

The SLURM job:
1. Boots vLLM on the allocated GH200 GPU(s)
2. Waits for the `/v1/models` endpoint to be healthy
3. Runs `run_eval.py` across all 17 `api_eval` jsonl files (single mode)
4. Runs `run_eval.py --mode transition --transition-type timeout` on `clean.jsonl`
5. Runs `eval_results.py` to produce `scores.json`
6. Tears down vLLM

Results → `<repo-root>/results/robustbench/results/<model-safe-name>/`

### Env-var knobs

| Var | Default | Notes |
|-----|---------|-------|
| `MODEL_ID` | *required* | HF repo id, e.g. `Qwen/Qwen3.5-9B` |
| `GPUS` | `1` | Tensor-parallel size for vLLM |
| `DTYPE` | `bfloat16` | Use `float32` for ToolRL-1.5B |
| `MAX_MODEL_LEN` | `16384` | Lower for OOM, higher for long-context benchmarks |
| `EXTRA_VLLM_FLAGS` | `""` | e.g. `"--enable-auto-tool-choice --tool-call-parser hermes"` for MUA-RL |
| `VLLM_PORT` | `8765` | Random if submit_all.sh is used |
| `RUN_TRANSITION` | `1` | Set to `0` to skip 2-pass transition phase |
| `TRANSITION_TYPE` | `timeout` | one of {timeout, rate_limit, auth_error, server_error, malformed_response, schema_drift} |
| `LIMIT` | unset | `LIMIT=50` for quick sanity (50 samples per file) |
| `MAX_WORKERS` | `16` | Parallel inflight requests to vLLM |
| `CONDA_ENV` | `rbtc-vllm` | Conda env to activate |
| `DATA_DIR` | HF download dir | `api_eval/*.jsonl` location |
| `RESULT_ROOT` | `<repo-root>/results/robustbench/results` | Per-model output parent |
| `HF_HOME` | `<repo-root>/cache/hf/.cache/huggingface` | Model cache |

### Model-specific notes

| Model | Notable config |
|-------|----------------|
| `ToolRL-Qwen2.5-1.5B` | `DTYPE=float32` (bf16 → inf/nan) |
| `MUA-RL-32B` | `GPUS=2`, `EXTRA_VLLM_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes"` |
| `Gemma-4-26B-A4B-it` | MoE (26B total / 3.8B active); fits 1 GH200 |
| `Qwen3.5-9B` | Native 262K ctx — cap `MAX_MODEL_LEN=16384` |
| `DeepSeek-R1-Distill-Qwen-14B` | Emits `<think>...</think>`; parser handles it |
| `TL-CodeLLaMA-2` | Non-standard chat template; may need `--chat-template` arg |

All are captured in `model_manifest.py`.

## Evaluate all 13 open models

```bash
# Submit all
bash scripts/slurm/submit_all.sh

# Preview sbatch commands without submitting
bash scripts/slurm/submit_all.sh --dry-run

# Filter by category or glob
bash scripts/slurm/submit_all.sh --category frontier_open
bash scripts/slurm/submit_all.sh --category proposed_rl
bash scripts/slurm/submit_all.sh --models 'Qwen/*' 'google/*'
```

Categories defined in `model_manifest.py`:
- `tool_rl_small` / `tool_rl` / `tool_rl_large` / `tool_sft` — open-source tool models
- `proposed_rl` — our perturbed-RL-trained ckpts (2; HF ids anonymized)
- `base_pair` — Qwen2.5/Llama base models for RL-effect ablation (3)
- `frontier_open` — Qwen3.5-9B / DeepSeek-R1-Distill (2)
- `frontier_closed` — `o4-mini` (skipped by submit_all; use OpenAI API)

`o4-mini` is closed-source and called separately via the OpenAI API,
not through this local-vLLM pipeline.

## Manual (non-SLURM) invocation

If you already have a vLLM endpoint up somewhere (e.g. an interactive
`ghx4-interactive` session), skip the SLURM job and call the Python
scripts directly:

```bash
# 1. Inference across the 17 perturbation/clean files
python scripts/run_eval.py \
    --model-name Qwen/Qwen3.5-9B \
    --endpoint http://<node>:8000/v1 \
    --data-dir <repo-root>/hf_data/datasets/api_eval \
    --output-dir <repo-root>/results/robustbench/results/qwen3.5_9b \
    --max-workers 16

# 2. Transition (2-pass, default timeout variant)
python scripts/run_eval.py \
    --model-name Qwen/Qwen3.5-9B \
    --endpoint http://<node>:8000/v1 \
    --output-dir <repo-root>/results/robustbench/results/qwen3.5_9b \
    --mode transition --transition-type timeout

# 3. Score everything (recursive glob, picks up transition_*/)
python scripts/eval_results.py \
    --predictions <repo-root>/results/robustbench/results/qwen3.5_9b \
    --output <repo-root>/results/robustbench/results/qwen3.5_9b/scores.json \
    --per-sample-output <repo-root>/results/robustbench/results/qwen3.5_9b/per_sample_scores.jsonl
```

## Output structure

```
${RESULT_ROOT}/<model-safe>/
├── vllm_server.log                          # vLLM stdout/stderr
├── clean.predictions.jsonl                  # inference outputs
├── realistic_typos.predictions.jsonl
├── query_paraphrase.predictions.jsonl
├── ...                                      # 17 files total (single mode)
├── transition_timeout/
│   └── clean.predictions.jsonl              # 2-pass transition output
├── _run_summary.json
├── scores.json                              # aggregated
└── per_sample_scores.jsonl                  # one score per sample
```

`scores.json` structure:
```jsonc
{
  "overall": {
    "overall":        {"n": ..., "accuracy": ..., "mean_score": ...},
    "by_benchmark":   {"bfcl_v3": {...}, "apibank": {...}, ...},
    "by_perturbation":{"clean": {...}, "realistic_typos": {...}, ...},
    "by_mdp_category":{"clean": {...}, "observation": {...},
                       "action": {...}, "reward": {...},
                       "transition": {...}},
    "by_benchmark_x_perturbation": { "bfcl_v3::clean": {...}, ... }
  },
  "per_file": { "<filename>.predictions.jsonl": {...} }
}
```

## Scoring logic (deterministic, no LLM judge)

| Original `eval_config.method` | Sample source | Scorer |
|---|---|---|
| `exact_match` | BFCL V3, API-Bank | name + numeric-tolerant param match |
| `rule_based` | RoTBench | TS / PI / CF 3-tier |
| `gpt_judge` + `relevance` | ToolEyes | name-only match |
| `gpt_judge` + `complete_accuracy` | ToolAlpaca | same as `exact_match` |

The `"gpt_judge"` label in HF data is NOT followed — scoring is
deterministic by design. Rationale: reproducibility, no API costs, no
key distribution.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: vllm` | On login node or wrong env. `conda activate rbtc-vllm` after `module load python/miniforge3_pytorch/2.10.0`. |
| vLLM server exits during boot | `${OUTPUT_DIR}/vllm_server.log` — usually HF cache perms, dtype mismatch, or OOM (lower `MAX_MODEL_LEN`) |
| `raw_output` is empty for every sample | Model may need a non-default chat template; inspect with `curl http://...:PORT/v1/models` and pass `--chat-template` to vLLM |
| Scores way below paper baseline | Look at `*.predictions.jsonl` → `raw_output` field. Is tool-call format being emitted? Does our parser match? `_strip_thinking()` handles `<think>` prefixes. |
| `skipped_multi_turn=191` | Expected: BFCL multi-turn requires a stateful backend not supported by the unified runner. 191 / 2527 samples. |

## Sanity check before running everything

For a quick 10-minute end-to-end test on a small model:

```bash
MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
LIMIT=30 RUN_TRANSITION=0 \
sbatch --gres=gpu:nvidia_gh200_120gb:1 \
       --time=01:00:00 \
       --job-name=rbtc_sanity \
       scripts/slurm/eval_one_model.slurm
```

This runs 30 samples per file (~500 inference calls), no transition phase.
Total end-to-end should finish in under an hour including model download.
Use this to verify vLLM launches, parser extracts tool calls, and scoring
runs before committing GPU hours to the full 13-model run.
