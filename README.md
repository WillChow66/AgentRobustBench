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

## Trained checkpoints

ToolRL-DR-Full and ToolRL-DR-Mixed weights are hosted on HuggingFace and
will be released alongside the camera-ready version of the paper.
Identifiers are withheld during double-blind review.

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

## License

Code: Apache-2.0. Data: see per-source license documented in the paper.
