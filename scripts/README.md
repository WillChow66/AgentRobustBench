# AgentRobustBench: Evaluation Scripts

This directory contains the runnable parts of the AgentRobustBench evaluation pipeline. There is intentionally **no figure-rendering or paper-analysis code here**; those are paper-internal and live in our private development repo.

## Files

| Path | Role |
|------|------|
| `run_eval.py` | Inference runner. Reads `hf_data/datasets/api_eval/*.jsonl`, dispatches per-benchmark prompt + parser, writes `*.predictions.jsonl`. Supports `--mode single` (all 16 perturbations + clean) and `--mode transition --transition-type <T>` (2-pass runtime injection for the 6 Transition variants). |
| `reparse_predictions.py` | Re-runs the per-benchmark `parse_tool_calls()` on existing `*.predictions.jsonl` (creates `*.bak` on first run). Useful if you extend a parser without wanting to re-run inference. |
| `model_manifest.py` | `ModelConfig` dataclass + an example registry. Add your own checkpoints here (HF id, GPU count, dtype, vLLM flags, FC vs prompt mode). |
| `requirements.txt` | Python deps for the local eval runner: `vllm`, `openai`, `tqdm`, `huggingface_hub`. |
| `slurm/eval_one_model.slurm` | End-to-end SLURM job: boot vLLM → single + transition(×6) → score → teardown. |
| `slurm/submit_all.sh` | Reads `model_manifest.py`, submits one job per model. Flags: `--models 'pattern'`, `--category <cat>`, `--dry-run`. |
| `slurm/tl_codellama_chat_template.jinja` | Custom chat template for `Junjie-Ye/TL-CodeLLaMA-2` (its tokenizer_config.json has none). |

Scoring is performed by **`leaderboard/scoring/eval_results.py`** in this repo (also the function the public leaderboard runs server-side on submitted predictions). All scoring is deterministic — there is no LLM judge.

## Quick start (no SLURM)

```bash
# 1. Point a vLLM server at your checkpoint, e.g.
vllm serve <model-id> --port 8000

# 2. Inference across all 17 jsonl files (16 perturbations + clean)
python scripts/run_eval.py \
    --model-name <model-id> \
    --endpoint http://localhost:8000/v1 \
    --data-dir hf_data/datasets/api_eval \
    --output-dir results/<safe-name> \
    --mode single \
    --skip-multi-turn

# 3. The 6 Transition variants (2-pass runtime injection)
for T in timeout rate_limit auth_error server_error malformed_response schema_drift; do
  python scripts/run_eval.py \
      --model-name <model-id> \
      --endpoint http://localhost:8000/v1 \
      --output-dir results/<safe-name> \
      --mode transition --transition-type "$T"
done

# 4. Score (deterministic; no API key required)
python leaderboard/scoring/eval_results.py \
    --predictions results/<safe-name> \
    --output      results/<safe-name>/scores.json \
    --per-sample-output results/<safe-name>/per_sample_scores.jsonl
```

## SLURM (preferred for full-lineup eval)

```bash
# Single model, end-to-end (single + 6 transition variants + score):
MODEL_ID=Qwen/Qwen3-8B GPUS=1 DTYPE=bfloat16 MAX_MODEL_LEN=16384 \
USE_FC=1 THINKING_MODE=disable \
EXTRA_VLLM_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes" \
sbatch scripts/slurm/eval_one_model.slurm

# Or submit all configured models in model_manifest.py:
bash scripts/slurm/submit_all.sh --dry-run         # preview
bash scripts/slurm/submit_all.sh                   # submit
bash scripts/slurm/submit_all.sh --models 'Qwen/*' # filter
```

See `slurm/README.md` for environment variables and per-model knobs.

## Submitting to the public leaderboard

After you have a `<safe-name>/` directory with `*.predictions.jsonl` (and optionally `transition_*/`), submit it via the leaderboard Space (URL in the paper / repo top-level `README.md`). The Space runs the same `eval_results.py` server-side and persists the row to a HuggingFace Dataset.
