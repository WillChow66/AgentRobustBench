# Training: reproduce ToolRL-DR-Full / ToolRL-DR-Mixed

This directory contains everything needed to retrain the two ToolRL-DR
checkpoints described in §5 of the paper. We reuse the public ToolRL
GRPO recipe (which embeds a fork of veRL) **without modifications**
to the trainer; we only substitute the training data.

The actual files fed to the GRPO trainer are the parquets in `data/`.
They were produced from our intermediate JSONL sources (not redistributed
here) by `convert_to_toolrl_parquet.py`, which is included as reference.

## Layout

| Path | Role |
|------|------|
| `data/train.parquet` | **DR-Full** training input. 3,826 perturbed samples in ToolRL parquet schema (`data_source`, `prompt`, `reward_model`, `extra_info`). Each `prompt` is a list of `{role, content}` messages; each `reward_model.ground_truth` is the ToolRL-format `<think>...</think>\n<tool_call>...</tool_call>` reference output. |
| `data/mixed_train.parquet` | **DR-Mixed** training input. 4,000 samples (~50/50 clean / perturbed). Same schema. |
| `data/test.parquet` | 79-sample validation split used during training (referenced by `data.val_files` in the GRPO launcher; trainer evaluates on it every 5 epochs). |
| `convert_to_toolrl_parquet.py` | Reference script that converts source JSONLs in our schema into the ToolRL parquet schema above. The shipped parquets were produced by this script; reviewers do **not** need to rerun it (the JSONL sources are not in this release). |
| `rlla_perturbed.py` | Reward-function file. **Drop-in replacement** for `verl/utils/reward_score/rlla.py` in the public ToolRL/veRL repo. The only behavioral change vs. upstream is that response-only samples receive a non-zero correctness signal instead of a flat 0.0. |
| `grpo_launch.sh` | End-to-end GRPO launcher. Hyperparameters identical to the public ToolRL recipe (lr=1e-6, KL=0.001, rollout n=4, 15 epochs); only the data path differs between DR-Full and DR-Mixed. |
| `grpo_hyperparams.yaml` | Human-readable summary of every flag the launcher sets. |

## Published checkpoints

- [ToolRL-DR-Full](https://huggingface.co/willchow66/ToolRL-DR-Full-Qwen2.5-3B)
- [ToolRL-DR-Mixed](https://huggingface.co/willchow66/ToolRL-DR-Mixed-Qwen2.5-3B)

## Reproducing a checkpoint

The pipeline is **two steps after a one-time setup**.

### Setup (once)

The reference implementation is [qiancheng0/ToolRL](https://github.com/qiancheng0/ToolRL). Use a separate training environment following its [installation instructions at this reference checkout](https://github.com/qiancheng0/ToolRL/blob/8cee13ec0ca72f0461da372a93a6fd8140dbb840/README.md#-installation); it uses an older vLLM stack than the evaluation environment.

```bash
git clone https://github.com/qiancheng0/ToolRL.git ~/ToolRL
cd ~/ToolRL
git checkout 8cee13ec0ca72f0461da372a93a6fd8140dbb840
# Complete the upstream environment installation linked above, then:
python -m pip install -e .
cd ~/AgentRobustBench
cp training/rlla_perturbed.py ~/ToolRL/verl/utils/reward_score/rlla.py
export PYTHONPATH="$HOME/ToolRL:${PYTHONPATH}"
```

### Launch — DR-Full

```bash
TRAIN_FILE=training/data/train.parquet \
VAL_FILE=training/data/test.parquet \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-full \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh
```

### Launch — DR-Mixed

```bash
TRAIN_FILE=training/data/mixed_train.parquet \
VAL_FILE=training/data/test.parquet \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
EXPERIMENT_NAME=toolrl-dr-mixed \
N_GPUS=2 ROLLOUT_TP_SIZE=1 \
bash training/grpo_launch.sh
```

Each run takes approximately 9.3 hours on a 2× RTX 6000 Ada workstation.
Checkpoints land under veRL's standard output directory (controlled by
`trainer.default_local_dir` in upstream config).

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
Reward:               public ToolRL structured reward, with the small
                      response-only patch in rlla_perturbed.py
                      (tool-name + parameter correctness; 0/1 binary)
```

See `grpo_hyperparams.yaml` for the full machine-readable manifest.

## Parquet schema

Each row of `data/{train,mixed_train,test}.parquet`:

```python
{
  "data_source": "rlla",
  "prompt": [
    {"role": "system", "content": <ToolRL system prompt with tools listed>},
    {"role": "user",   "content": <dialogue history with <user>...</user> tags>},
  ],
  "ability": "math",
  "reward_model": {
    "style": "rule",
    "ground_truth": "<think> ... </think>\n<tool_call>\n{json}\n</tool_call>",
  },
  "extra_info": {
    "split": "train",
    "index": <int>,
    "instruction": <same as prompt[0].content>,
    "input":       <same as prompt[1].content>,
    "output":      <same as reward_model.ground_truth>,
    "perturbation": {"type": "...", "mdp_category": "..."},
    "original_id": <id from AgentRobustBench source jsonl>,
  },
}
```

## Notes

- **Transition perturbations are intentionally absent from training data.**
  They are runtime properties of tool execution and are evaluated only at
  inference time; see §5.1 of the paper for the rationale.
- `convert_to_toolrl_parquet.py` reads `np.random.seed(42)` for the
  train/test split. The shipped `train.parquet` / `test.parquet` were
  produced with this seed; rerunning with a different value will yield
  a different split.
