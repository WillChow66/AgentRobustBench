# Training: reproduce ToolRL-DR-Full / ToolRL-DR-Mixed

This directory contains everything needed to retrain the two ToolRL-DR
checkpoints described in §5 of the paper, on top of the public ToolRL
recipe with our perturbation-augmented training data substituted in.

The training itself uses the public ToolRL repository (which embeds a
fork of veRL's GRPO trainer) **without modifications**. We do not ship a
copy of veRL; instead we ship the data and the launcher needed to plug
into the upstream public release.

## Layout

| Path | Role |
|------|------|
| `data/clean.jsonl` | The original ToolRL `rlla_4k` 4{,}000 training samples reformatted into RobustBench-TC schema. Used as a baseline; reproduces the public **ToolRL-Clean** checkpoint when the launcher points at it. |
| `data/perturbed.jsonl` | Our 3{,}984 perturbation-augmented samples. **Source for ToolRL-DR-Full.** Distribution: 1{,}362 Observation + 519 Action + 2{,}103 Reward. |
| `data/mixed.jsonl` | A 50/50 mix (2{,}000 clean + 2{,}000 perturbed). **Source for ToolRL-DR-Mixed.** Same perturbation breakdown, downsampled. |
| `data/metadata.json` | Sample counts, perturbation breakdown, and seed metadata. |
| `convert_to_parquet.py` | Converts any of the three JSONLs into the `prompt`/`response` parquet format expected by veRL's GRPO trainer. |
| `grpo_launch.sh` | End-to-end GRPO launcher. Hyperparameters are identical to the public ToolRL recipe; only `data.train_files` changes between DR-Full and DR-Mixed. |
| `grpo_hyperparams.yaml` | Human-readable summary of every hyperparameter the launcher sets. |

## Reproducing a checkpoint

The pipeline is **two steps after a one-time setup**.

### Setup (once)

```bash
# 1. Clone the public ToolRL repo (which embeds the GRPO trainer).
#    The exact upstream is cited in §5.2 of the paper. After clone,
#    install its requirements per its own README (vllm, verl, etc.).
git clone <public-toolrl-url> /path/to/toolrl
cd /path/to/toolrl && pip install -r requirements.txt   # follow upstream

# 2. Make the verl module importable for grpo_launch.sh.
export PYTHONPATH="/path/to/toolrl:${PYTHONPATH}"
```

### Step 1 — convert the chosen JSONL to parquet

```bash
# DR-Full
python training/convert_to_parquet.py \
    --input  training/data/perturbed.jsonl \
    --output-dir training/parquet/dr_full

# DR-Mixed
python training/convert_to_parquet.py \
    --input  training/data/mixed.jsonl \
    --output-dir training/parquet/dr_mixed
```

Output of each call:

```
training/parquet/<run>/
├── train.parquet                   # 98% of samples, used for RL
├── val.parquet                     # 2% held-out validation
└── raw_jsonl_for_inspection.jsonl  # the same data flat-jsonl for sanity
```

### Step 2 — launch GRPO

```bash
# DR-Full
DATA_DIR=training/parquet/dr_full \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-full \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh

# DR-Mixed
DATA_DIR=training/parquet/dr_mixed \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-mixed \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh
```

Each run takes approximately 9.3 hours on a single 2× RTX 6000 Ada
workstation. Checkpoints land under veRL's standard output directory
(controlled by `trainer.default_local_dir` in upstream config).

## Hyperparameters at a glance

```
Algorithm:            GRPO (advantage estimator = grpo)
KL controller coef:   0.001
Actor learning rate:  1e-6
Rollout (n):          4 samples per prompt
Train batch size:     512
PPO mini-batch:       128
Max prompt length:    2048
Max response length:  1024
Total epochs:         15
Reward:               public ToolRL structured reward (unmodified)
                      tool-name + parameter correctness; binary 0/1
```

See `grpo_hyperparams.yaml` for the full machine-readable manifest.

## Data schema

Each line of `clean.jsonl` / `perturbed.jsonl` / `mixed.jsonl` is one
training sample with these fields:

```jsonc
{
  "id": "toolrl_3939",
  "benchmark": "toolrl",
  "category": "toolrl_train",
  "conversation": [{"role": "user", "content": "..."}],
  "tools": [{"name": "...", "description": "...", "parameters": {...}}, ...],
  "golden_answers": [{"name": "...", "parameters": {...}}, ...],
  "perturbation": {"type": "query_paraphrase", "mdp_category": "observation"},
  "source": {"dataset": "toolrl_rlla_4k", "original_index": 3939}
}
```

For perturbed / mixed samples, `perturbation.type` is one of:

| MDP component | Subtypes |
|---|---|
| Observation | `realistic_typos`, `query_paraphrase`, `paraphrase_tool_description`, `paraphrase_parameter_description` |
| Action | `same_name_A`, `same_name_B`, `same_name_C`, `same_name_D`, `same_name_E`, `redundant` |
| Reward | `CD`, `TD`, `CD_NT`, `TD_NT`, `CD_AB`, `TD_AB` |

Transition perturbations are intentionally absent from training data.
They are runtime properties of tool execution and are evaluated only at
inference time; see §5.1 of the paper for the rationale.

## Notes on randomness

`convert_to_parquet.py` sets `random.seed(args.seed)` (default 42)
before the train/val shuffle. The veRL GRPO trainer's own randomness
(rollout sampling, optimizer init) is controlled by veRL's seed flags
documented upstream.
