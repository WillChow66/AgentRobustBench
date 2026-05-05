# RobustBench-TC

Production-grounded sim-to-real robustness benchmark and domain-randomized
RL recipe for tool-use language agents (NeurIPS 2026 D&B-track submission,
under double-blind review).

This repository contains the **runnable, reviewer-facing release** of the
benchmark and leaderboard pipeline. Paper-internal analysis and figure code
are not included.

## Layout

| Directory | Contents |
|-----------|----------|
| `hf_data/datasets/api_eval/` | 17 JSONL files: 16 static perturbation cells + the clean baseline (3{,}721 predictions per model in total). The 6 Transition variants are produced at runtime by `scripts/run_eval.py`, not stored as separate files. |
| `scripts/` | Inference runner (`run_eval.py`), reparser (`reparse_predictions.py`), model manifest, and SLURM templates. See `scripts/README.md`. |
| `leaderboard/` | HuggingFace Space implementation: Gradio app, deterministic scorer, leaderboard CSV. See `leaderboard/README.md`. |
| `training/` | Reproduction recipe for ToolRL-DR-Full and ToolRL-DR-Mixed: the parquets actually fed to the GRPO trainer (`train.parquet`, `mixed_train.parquet`, `test.parquet`), the reference jsonl→parquet converter, our drop-in reward-score patch (`rlla_perturbed.py`), the GRPO launcher, and a hyperparameter manifest. See `training/README.md`. |

## Trained checkpoints

ToolRL-DR-Full and ToolRL-DR-Mixed weights are hosted on HuggingFace and
will be released alongside the camera-ready version of the paper.
Identifiers are withheld during double-blind review. The training data
and recipe needed to reproduce them from scratch are in `training/`.

## Quick start

```bash
# 1. Run inference against any vLLM endpoint:
python scripts/run_eval.py \
    --model-name <model-id> \
    --endpoint <vllm-endpoint>/v1 \
    --data-dir hf_data/datasets/api_eval \
    --output-dir results/<safe-name> \
    --mode single \
    --skip-multi-turn

# 2. Score the predictions (deterministic; no API key required):
python leaderboard/scoring/eval_results.py \
    --predictions results/<safe-name> \
    --output      results/<safe-name>/scores.json

# 3. Or submit `results/<safe-name>/` to the public leaderboard Space
#    (the Space runs the same scorer server-side).
```

The full SLURM end-to-end pipeline (boot vLLM → 17 single-mode files →
6 Transition variants → score → teardown) is in `scripts/slurm/`.

## Reproducing ToolRL-DR (optional)

The parquets fed to the GRPO trainer are already in `training/data/`,
so reviewers do not need to run any conversion. After the one-time setup
described in `training/README.md` (clone public ToolRL, drop in the
`rlla_perturbed.py` reward patch, set `PYTHONPATH`):

```bash
# DR-Full
TRAIN_FILE=training/data/train.parquet \
VAL_FILE=training/data/test.parquet \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-full \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh

# DR-Mixed: change TRAIN_FILE to training/data/mixed_train.parquet
```

## License

Code: Apache-2.0. Data: see per-source license documented in the paper.
