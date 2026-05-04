# RobustBench-TC

Production-grounded sim-to-real robustness benchmark and domain-randomized
RL recipe for tool-use language agents.

This repository contains:

- `scripts/`              — eval pipeline, figure/table renderer, audit pipeline
- `scripts/slurm/`        — SLURM templates for end-to-end one-model eval jobs
- `leaderboard/`          — HuggingFace Space implementation (live URL withheld for double-blind review)
- `hf_data/datasets/api_eval/` — 17 perturbation JSONLs + clean baseline (3{,}721 predictions per model)

## Trained checkpoints

ToolRL-DR-Full and ToolRL-DR-Mixed weights are hosted on HuggingFace and will
be released alongside the camera-ready version of the paper. Identifiers are
withheld during double-blind review.

## Quick start

```bash
# Run the eval pipeline against any vLLM endpoint:
python scripts/run_eval.py \
    --model-name <model-id> \
    --endpoint <vllm-endpoint>/v1 \
    --data-dir hf_data/datasets/api_eval \
    --output-dir results/<model-safe-name> \
    --mode single \
    --skip-multi-turn

# Score the predictions: predictions JSONs are scored server-side by the
# leaderboard Space; submit them via the live submission portal listed in
# the paper.
#
# Render figures + tables for the paper:
python scripts/render_figures.py
python scripts/fig_hero.py
python scripts/fig2_scaling.py
```

## License

Code: Apache-2.0. Data: see per-source license documented in the paper.
